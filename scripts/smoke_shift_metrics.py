from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from clothsense.abstention import confidence_threshold_decision
from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.data import build_dataloaders, load_raw_clean_test
from clothsense.evaluation import collect_logits_with_runtime
from clothsense.metrics import (
    classification_metrics,
    conformal_metrics,
    probability_metrics,
    selective_prediction_metrics,
)
from clothsense.reproducibility import select_device
from clothsense.shifts import build_shift_scenarios, find_scenario
from clothsense.training import configuration_hash, load_checkpoint, seed_artifact_paths
from clothsense.uncertainty_artifacts import file_sha256, load_uncertainty_artifact


def evaluate_subset(name, model, loader, fitted, config, device, alpha, threshold) -> None:
    logits, labels, predictions, runtime = collect_logits_with_runtime(model, loader, device)
    outputs = fitted.apply(torch.from_numpy(logits), alpha)
    classification = classification_metrics(labels, predictions)
    calibration = probability_metrics(
        outputs.calibrated.probabilities,
        labels,
        config.uncertainty.calibration_bin_edges,
    )
    conformal = conformal_metrics(
        outputs.standard_sets,
        labels,
        target_coverage=1.0 - alpha,
    )
    decisions = [
        confidence_threshold_decision(int(prediction), float(confidence), threshold)
        for prediction, confidence in zip(
            outputs.calibrated.predicted_classes,
            outputs.calibrated.confidence,
        )
    ]
    selective = selective_prediction_metrics(labels, decisions)
    print(
        f"{name}: accuracy={classification['accuracy']:.4f} "
        f"ece={calibration['aggregate']['expected_calibration_error']:.4f} "
        f"coverage={conformal['aggregate']['empirical_coverage']:.4f} "
        f"acceptance={selective['acceptance_rate']:.4f} "
        f"seconds={runtime.evaluation_seconds:.4f} "
        f"seconds_per_image={runtime.mean_inference_seconds_per_image:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test shifted uncertainty evaluation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("Sample count must be positive")

    config = load_config(args.config)
    device = select_device(config.device)
    artifacts = seed_artifact_paths(config, args.seed)
    model, checkpoint = load_checkpoint(artifacts.checkpoint, device)
    run_hash = configuration_hash(config)
    checkpoint_hash = file_sha256(artifacts.checkpoint)
    if int(checkpoint["seed"]) != args.seed or checkpoint["config_hash"] != run_hash:
        raise ValueError("Checkpoint is incompatible with the requested seed/configuration")
    fitted, _ = load_uncertainty_artifact(
        artifacts.uncertainty,
        expected_seed=args.seed,
        expected_config_hash=run_hash,
        expected_checkpoint_sha256=checkpoint_hash,
    )

    loaders = build_dataloaders(config, download=not args.no_download)
    raw_clean = load_raw_clean_test(config, download=False)
    scenarios = build_shift_scenarios(raw_clean, config, seed=args.seed)
    noise = find_scenario(scenarios, "gaussian_noise", "mild")
    clean_loader = DataLoader(
        Subset(loaders.clean_test.dataset, range(args.samples)),
        batch_size=min(config.training.batch_size, args.samples),
        shuffle=False,
    )
    shifted_loader = DataLoader(
        Subset(noise.dataset, range(args.samples)),
        batch_size=min(config.training.batch_size, args.samples),
        shuffle=False,
    )
    alpha = config.uncertainty.alpha_values[1]
    threshold = config.uncertainty.abstention_thresholds[3]
    evaluate_subset("clean", model, clean_loader, fitted, config, device, alpha, threshold)
    evaluate_subset("gaussian_noise/mild", model, shifted_loader, fitted, config, device, alpha, threshold)


if __name__ == "__main__":
    main()

