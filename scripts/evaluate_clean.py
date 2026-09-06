from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.data import build_dataloaders
from clothsense.evaluation import (
    classification_metrics,
    collect_logits_with_runtime,
    save_clean_evaluation,
)
from clothsense.plotting import plot_confusion_matrix
from clothsense.reproducibility import select_device
from clothsense.training import configuration_hash, load_checkpoint, seed_artifact_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved CNNs on clean Fashion-MNIST")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--seed", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(config.device)
    loaders = build_dataloaders(config, download=not args.no_download)
    seeds = config.seeds.experiments if args.all else (args.seed,)
    for seed_value in seeds:
        seed = int(seed_value)
        artifacts = seed_artifact_paths(config, seed)
        model, checkpoint = load_checkpoint(artifacts.checkpoint, device)
        expected_hash = configuration_hash(config)
        if checkpoint["config_hash"] != expected_hash or int(checkpoint["seed"]) != seed:
            raise ValueError(f"Checkpoint metadata does not match seed {seed} and current config")
        logits, labels, predictions, runtime = collect_logits_with_runtime(
            model, loaders.clean_test, device
        )
        metrics = classification_metrics(labels, predictions)
        save_clean_evaluation(
            logits,
            labels,
            predictions,
            metrics,
            seed=seed,
            config_hash=expected_hash,
            checkpoint_path=artifacts.checkpoint,
            outputs_path=artifacts.clean_outputs,
            metrics_path=artifacts.clean_metrics,
            runtime=runtime,
        )
        plot_confusion_matrix(metrics, artifacts.confusion_matrix)
        print(
            f"seed={seed} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
