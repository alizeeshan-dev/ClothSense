from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics


@dataclass(frozen=True)
class RuntimeMetrics:
    evaluation_seconds: float
    sample_count: int

    @property
    def mean_inference_seconds_per_image(self) -> float:
        return self.evaluation_seconds / self.sample_count

    def as_dict(self) -> dict[str, float | int]:
        return {
            "evaluation_seconds": self.evaluation_seconds,
            "mean_inference_seconds_per_image": self.mean_inference_seconds_per_image,
            "sample_count": self.sample_count,
        }


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float32))
            label_batches.append(labels.numpy().astype(np.int64))
    if not logits_batches:
        raise ValueError("Cannot evaluate an empty DataLoader")
    logits_array = np.concatenate(logits_batches)
    labels_array = np.concatenate(label_batches)
    predictions = logits_array.argmax(axis=1).astype(np.int64)
    return logits_array, labels_array, predictions


def collect_logits_with_runtime(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RuntimeMetrics]:
    start = time.perf_counter()
    logits, labels, predictions = collect_logits(model, loader, device)
    runtime = RuntimeMetrics(time.perf_counter() - start, len(labels))
    return logits, labels, predictions, runtime


def save_clean_evaluation(
    logits: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, Any],
    *,
    seed: int,
    config_hash: str,
    checkpoint_path: str | Path,
    outputs_path: str | Path,
    metrics_path: str | Path,
    runtime: RuntimeMetrics | None = None,
) -> None:
    output_destination = Path(outputs_path)
    output_destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "logits": logits.astype(np.float32),
        "labels": labels.astype(np.int64),
        "predictions": predictions.astype(np.int64),
    }
    if runtime is not None:
        arrays["evaluation_seconds"] = np.asarray(runtime.evaluation_seconds, dtype=np.float64)
        arrays["mean_inference_seconds_per_image"] = np.asarray(
            runtime.mean_inference_seconds_per_image, dtype=np.float64
        )
    np.savez_compressed(output_destination, **arrays)
    metrics_destination = Path(metrics_path)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "config_hash": config_hash,
        "checkpoint_path": str(checkpoint_path),
        "sample_count": int(len(labels)),
        "metrics": metrics,
    }
    if runtime is not None:
        payload["runtime"] = runtime.as_dict()
    metrics_destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
