from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from clothsense.abstention import (
    confidence_threshold_decision,
    hybrid_decision,
    singleton_set_decision,
)
from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.data import build_dataloaders
from clothsense.reproducibility import select_device
from clothsense.storage import update_model_temperature
from clothsense.training import configuration_hash, load_checkpoint, seed_artifact_paths
from clothsense.uncertainty import fit_uncertainty
from clothsense.uncertainty_artifacts import (
    file_sha256,
    load_or_create_calibration_outputs,
    save_uncertainty_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit ClothSense calibration and conformal parameters")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--seed", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--smoke-clean-samples", type=int, default=0)
    return parser.parse_args()


def smoke_clean_subset(
    fitted,
    clean_outputs: Path,
    sample_count: int,
    alpha: float,
    threshold: float,
) -> str:
    with np.load(clean_outputs, allow_pickle=False) as saved:
        logits = torch.from_numpy(saved["logits"][:sample_count])
    outputs = fitted.apply(logits, alpha)
    confidence_decisions = [
        confidence_threshold_decision(int(prediction), float(confidence), threshold)
        for prediction, confidence in zip(
            outputs.calibrated.predicted_classes,
            outputs.calibrated.confidence,
        )
    ]
    singleton_decisions = [singleton_set_decision(value) for value in outputs.standard_sets]
    hybrid_decisions = [
        hybrid_decision(int(prediction), float(confidence), conformal_set, threshold)
        for prediction, confidence, conformal_set in zip(
            outputs.calibrated.predicted_classes,
            outputs.calibrated.confidence,
            outputs.standard_sets,
        )
    ]
    accepted = tuple(
        sum(decision.accepted for decision in decisions)
        for decisions in (confidence_decisions, singleton_decisions, hybrid_decisions)
    )
    return (
        f"smoke_samples={sample_count} threshold={threshold:g} "
        f"accepted_confidence/singleton/hybrid={accepted}"
    )


def main() -> None:
    args = parse_args()
    if args.smoke_clean_samples < 0:
        raise ValueError("Smoke sample count cannot be negative")
    config = load_config(args.config)
    device = select_device(config.device)
    loaders = build_dataloaders(config, download=not args.no_download)
    run_hash = configuration_hash(config)
    seeds = config.seeds.experiments if args.all else (args.seed,)

    for seed_value in seeds:
        seed = int(seed_value)
        artifacts = seed_artifact_paths(config, seed)
        model, checkpoint = load_checkpoint(artifacts.checkpoint, device)
        if int(checkpoint["seed"]) != seed or checkpoint["config_hash"] != run_hash:
            raise ValueError(f"Checkpoint for seed {seed} is incompatible with current configuration")
        checkpoint_hash = file_sha256(artifacts.checkpoint)
        logits, labels, reused = load_or_create_calibration_outputs(
            model,
            loaders.calibration,
            device,
            artifacts.calibration_outputs,
            seed=seed,
            config_hash=run_hash,
            checkpoint_sha256=checkpoint_hash,
            expected_samples=config.data.calibration_size,
            force=args.force_inference,
        )
        fitted = fit_uncertainty(
            torch.from_numpy(logits),
            torch.from_numpy(labels),
            config.uncertainty.alpha_values,
        )
        save_uncertainty_artifact(
            fitted,
            artifacts.uncertainty,
            seed=seed,
            config_hash=run_hash,
            checkpoint_sha256=checkpoint_hash,
            calibration_outputs=artifacts.calibration_outputs.relative_to(config.root),
        )
        model_id = update_model_temperature(
            config.paths.database,
            seed=seed,
            config_hash=run_hash,
            temperature=fitted.temperature,
        )
        message = (
            f"seed={seed} model_id={model_id} temperature={fitted.temperature:.6f} "
            f"calibration_logits={'reused' if reused else 'created'}"
        )
        if args.smoke_clean_samples:
            message += " " + smoke_clean_subset(
                fitted,
                artifacts.clean_outputs,
                args.smoke_clean_samples,
                config.uncertainty.alpha_values[1],
                config.uncertainty.abstention_thresholds[3],
            )
        print(message, flush=True)


if __name__ == "__main__":
    main()
