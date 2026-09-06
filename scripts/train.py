from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.data import build_dataloaders
from clothsense.evaluation import (
    classification_metrics,
    collect_logits_with_runtime,
    save_clean_evaluation,
)
from clothsense.plotting import plot_confusion_matrix, plot_training_curves
from clothsense.reproducibility import select_device
from clothsense.training import EpochMetrics, train_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed ClothSense Fashion-MNIST CNNs")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--seed", type=int, help="Train one seed")
    selection.add_argument("--all", action="store_true", help="Train all configured experiment seeds")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def report_epoch(
    epoch: int,
    train: EpochMetrics,
    validation: EpochMetrics,
    seconds: float,
) -> None:
    print(
        f"epoch={epoch:02d} train_loss={train.loss:.4f} train_acc={train.accuracy:.4f} "
        f"val_loss={validation.loss:.4f} val_acc={validation.accuracy:.4f} "
        f"seconds={seconds:.1f}",
        flush=True,
    )


def run_seed(config, seed: int, *, download: bool) -> None:
    device = select_device(config.device)
    loaders = build_dataloaders(config, download=download, loader_seed=seed)
    print(f"training seed={seed} device={device}", flush=True)
    trained = train_seed(config, seed, loaders, device=device, on_epoch=report_epoch)
    plot_training_curves(trained.fit.history, trained.artifacts.training_curves)

    logits, labels, predictions, runtime = collect_logits_with_runtime(
        trained.model, loaders.clean_test, device
    )
    metrics = classification_metrics(labels, predictions)
    save_clean_evaluation(
        logits,
        labels,
        predictions,
        metrics,
        seed=seed,
        config_hash=trained.config_hash,
        checkpoint_path=trained.artifacts.checkpoint,
        outputs_path=trained.artifacts.clean_outputs,
        metrics_path=trained.artifacts.clean_metrics,
        runtime=runtime,
    )
    plot_confusion_matrix(metrics, trained.artifacts.confusion_matrix)
    print(
        f"complete seed={seed} best_epoch={trained.fit.best_epoch} "
        f"val_loss={trained.fit.validation_loss:.4f} val_acc={trained.fit.validation_accuracy:.4f} "
        f"test_acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"model_id={trained.model_id}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seeds = config.seeds.experiments if args.all else (args.seed,)
    for seed in seeds:
        run_seed(config, int(seed), download=not args.no_download)


if __name__ == "__main__":
    main()
