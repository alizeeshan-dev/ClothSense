from __future__ import annotations

import argparse

from clothsense.config import DEFAULT_CONFIG_PATH, load_config
from clothsense.data import (
    build_dataloaders,
    compute_normalization,
    dataset_summary,
    load_datasets,
    save_sample_grid,
)
from clothsense.reproducibility import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the ClothSense Fashion-MNIST data")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML configuration path")
    parser.add_argument("--no-download", action="store_true", help="Require already-downloaded data")
    parser.add_argument(
        "--recompute-splits",
        action="store_true",
        help="Regenerate the configured seed's split file deterministically",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_directories()
    seed_everything(config.seeds.split, config.device.deterministic_algorithms)

    loaders = build_dataloaders(
        config,
        download=not args.no_download,
        force_recompute_splits=args.recompute_splits,
    )
    raw_training, _, clean_test = load_datasets(config, download=False)
    mean, std = compute_normalization(raw_training, loaders.splits.train)
    configured_mean = config.data.normalization_mean[0]
    configured_std = config.data.normalization_std[0]
    if abs(mean - configured_mean) > 5e-5 or abs(std - configured_std) > 5e-5:
        raise ValueError(
            "Configured normalization does not match the training split: "
            f"computed mean={mean:.8f}, std={std:.8f}"
        )

    grid_path = save_sample_grid(
        raw_training,
        loaders.splits.train,
        config.paths.plots / "fashion_mnist_samples.png",
    )
    print(dataset_summary(raw_training.targets, loaders.splits, clean_test.targets))
    print(f"normalization_mean={mean:.8f}")
    print(f"normalization_std={std:.8f}")
    print(f"split_file={config.split_file}")
    print(f"sample_grid={grid_path}")
    print(f"device={select_device(config.device)}")


if __name__ == "__main__":
    main()
