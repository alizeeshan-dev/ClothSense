from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import collect_logits
from .uncertainty import FittedUncertainty


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_calibration_outputs(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    destination: str | Path,
    *,
    seed: int,
    config_hash: str,
    checkpoint_sha256: str,
    expected_samples: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    path = Path(destination)
    if path.is_file() and not force:
        with np.load(path, allow_pickle=False) as saved:
            required = {"logits", "labels", "seed", "config_hash", "checkpoint_sha256"}
            compatible = (
                set(saved.files) == required
                and int(saved["seed"]) == seed
                and str(saved["config_hash"]) == config_hash
                and str(saved["checkpoint_sha256"]) == checkpoint_sha256
                and saved["logits"].shape == (expected_samples, 10)
                and saved["labels"].shape == (expected_samples,)
            )
            if compatible:
                return saved["logits"].astype(np.float32), saved["labels"].astype(np.int64), True

    logits, labels, _ = collect_logits(model, calibration_loader, device)
    if logits.shape != (expected_samples, 10) or labels.shape != (expected_samples,):
        raise ValueError("Calibration inference returned unexpected shapes")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logits=logits.astype(np.float32),
        labels=labels.astype(np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        config_hash=np.asarray(config_hash),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
    )
    return logits, labels, False


def _alpha_key(alpha: float) -> str:
    return format(float(alpha), ".12g")


def save_uncertainty_artifact(
    fitted: FittedUncertainty,
    destination: str | Path,
    *,
    seed: int,
    config_hash: str,
    checkpoint_sha256: str,
    calibration_outputs: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "seed": seed,
        "config_hash": config_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_outputs": str(calibration_outputs),
        "temperature": fitted.temperature,
        "standard_conformal_thresholds": {
            _alpha_key(alpha): threshold
            for alpha, threshold in fitted.standard_thresholds.items()
        },
        "class_conditional_conformal_thresholds": {
            _alpha_key(alpha): [
                thresholds[class_id] for class_id in range(len(thresholds))
            ]
            for alpha, thresholds in fitted.class_conditional_thresholds.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_uncertainty_artifact(
    path: str | Path,
    *,
    expected_seed: int | None = None,
    expected_config_hash: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> tuple[FittedUncertainty, dict[str, object]]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported uncertainty artifact format")
    expectations = (
        ("seed", expected_seed),
        ("config_hash", expected_config_hash),
        ("checkpoint_sha256", expected_checkpoint_sha256),
    )
    for key, expected in expectations:
        if expected is not None and metadata.get(key) != expected:
            raise ValueError(f"Uncertainty artifact {key} is incompatible")
    standard = {
        float(alpha): float(threshold)
        for alpha, threshold in metadata["standard_conformal_thresholds"].items()
    }
    conditional = {
        float(alpha): {
            class_id: float(threshold)
            for class_id, threshold in enumerate(thresholds)
        }
        for alpha, thresholds in metadata["class_conditional_conformal_thresholds"].items()
    }
    fitted = FittedUncertainty(float(metadata["temperature"]), standard, conditional)
    return fitted, metadata

