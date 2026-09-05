from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import ProjectConfig
from .reproducibility import dataloader_generator, seed_worker


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    seed: int
    dataset_size: int


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    calibration: DataLoader
    clean_test: DataLoader
    splits: SplitIndices


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    train_size: int,
    validation_size: int,
    calibration_size: int,
    seed: int,
) -> SplitIndices:
    labels_array = np.asarray(labels, dtype=np.int64)
    total = train_size + validation_size + calibration_size
    if labels_array.ndim != 1 or len(labels_array) != total:
        raise ValueError(f"Expected exactly {total} one-dimensional labels")

    all_indices = np.arange(total, dtype=np.int64)
    train, held_out = train_test_split(
        all_indices,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=labels_array,
    )
    validation, calibration = train_test_split(
        held_out,
        train_size=validation_size,
        test_size=calibration_size,
        random_state=seed,
        shuffle=True,
        stratify=labels_array[held_out],
    )
    splits = SplitIndices(
        train=np.sort(train),
        validation=np.sort(validation),
        calibration=np.sort(calibration),
        seed=seed,
        dataset_size=total,
    )
    validate_splits(splits, train_size, validation_size, calibration_size)
    return splits


def validate_splits(
    splits: SplitIndices,
    train_size: int,
    validation_size: int,
    calibration_size: int,
) -> None:
    expected_sizes = (train_size, validation_size, calibration_size)
    arrays = (splits.train, splits.validation, splits.calibration)
    if tuple(len(array) for array in arrays) != expected_sizes:
        raise ValueError("Saved split sizes do not match configuration")
    combined = np.concatenate(arrays)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Training, validation, and calibration splits overlap")
    if len(combined) != splits.dataset_size or not np.array_equal(
        np.sort(combined), np.arange(splits.dataset_size)
    ):
        raise ValueError("Splits do not partition the official training dataset")


def save_split_indices(splits: SplitIndices, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        train=splits.train,
        validation=splits.validation,
        calibration=splits.calibration,
        seed=np.asarray(splits.seed, dtype=np.int64),
        dataset_size=np.asarray(splits.dataset_size, dtype=np.int64),
    )


def load_split_indices(path: str | Path) -> SplitIndices:
    with np.load(Path(path), allow_pickle=False) as saved:
        required = {"train", "validation", "calibration", "seed", "dataset_size"}
        if set(saved.files) != required:
            raise ValueError("Split file has an unexpected schema")
        return SplitIndices(
            train=saved["train"].astype(np.int64),
            validation=saved["validation"].astype(np.int64),
            calibration=saved["calibration"].astype(np.int64),
            seed=int(saved["seed"]),
            dataset_size=int(saved["dataset_size"]),
        )


def get_or_create_splits(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    config: ProjectConfig,
    *,
    force_recompute: bool = False,
) -> SplitIndices:
    path = config.split_file
    if path.exists() and not force_recompute:
        splits = load_split_indices(path)
        if splits.seed != config.seeds.split:
            raise ValueError("Saved splits use a different split seed")
        validate_splits(
            splits,
            config.data.train_size,
            config.data.validation_size,
            config.data.calibration_size,
        )
        if splits.dataset_size != len(labels):
            raise ValueError("Saved splits target a different dataset size")
        return splits

    splits = stratified_split_indices(
        labels,
        config.data.train_size,
        config.data.validation_size,
        config.data.calibration_size,
        config.seeds.split,
    )
    save_split_indices(splits, path)
    return splits


def evaluation_transform(config: ProjectConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.data.normalization_mean, config.data.normalization_std),
        ]
    )


def training_transform(config: ProjectConfig) -> transforms.Compose:
    return evaluation_transform(config)


def load_datasets(config: ProjectConfig, *, download: bool = True) -> tuple[Dataset, Dataset, Dataset]:
    raw_training = datasets.FashionMNIST(root=config.paths.data, train=True, download=download)
    transformed_training = datasets.FashionMNIST(
        root=config.paths.data,
        train=True,
        download=False,
        transform=training_transform(config),
    )
    clean_test = datasets.FashionMNIST(
        root=config.paths.data,
        train=False,
        download=download,
        transform=evaluation_transform(config),
    )
    if len(raw_training) != sum(
        (config.data.train_size, config.data.validation_size, config.data.calibration_size)
    ):
        raise ValueError("Official Fashion-MNIST training set must contain 60,000 images")
    if len(clean_test) != config.data.test_size:
        raise ValueError("Official Fashion-MNIST test set must contain 10,000 images")
    return raw_training, transformed_training, clean_test


def load_raw_clean_test(config: ProjectConfig, *, download: bool = True) -> Dataset:
    dataset = datasets.FashionMNIST(root=config.paths.data, train=False, download=download)
    if len(dataset) != config.data.test_size:
        raise ValueError("Official Fashion-MNIST test set must contain 10,000 images")
    return dataset


def build_dataloaders(
    config: ProjectConfig,
    *,
    download: bool = True,
    force_recompute_splits: bool = False,
    loader_seed: int | None = None,
) -> DataLoaders:
    raw_training, transformed_training, clean_test = load_datasets(config, download=download)
    splits = get_or_create_splits(
        raw_training.targets,
        config,
        force_recompute=force_recompute_splits,
    )

    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": config.training.num_workers > 0,
    }
    return DataLoaders(
        train=DataLoader(
            Subset(transformed_training, splits.train.tolist()),
            shuffle=True,
            generator=dataloader_generator(
                config.seeds.dataloader if loader_seed is None else loader_seed
            ),
            **common,
        ),
        validation=DataLoader(
            Subset(transformed_training, splits.validation.tolist()), shuffle=False, **common
        ),
        calibration=DataLoader(
            Subset(transformed_training, splits.calibration.tolist()), shuffle=False, **common
        ),
        clean_test=DataLoader(clean_test, shuffle=False, **common),
        splits=splits,
    )


def compute_normalization(
    raw_training: Dataset,
    train_indices: Sequence[int] | np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[float, float]:
    if not hasattr(raw_training, "data"):
        raise TypeError("Dataset must expose raw pixel data")
    indices = np.asarray(train_indices, dtype=np.int64)
    pixel_sum = 0.0
    squared_pixel_sum = 0.0
    pixel_count = 0
    for start in range(0, len(indices), chunk_size):
        batch = raw_training.data[indices[start : start + chunk_size]].to(torch.float64) / 255.0
        pixel_sum += batch.sum().item()
        squared_pixel_sum += batch.square().sum().item()
        pixel_count += batch.numel()
    mean = pixel_sum / pixel_count
    variance = max(squared_pixel_sum / pixel_count - mean * mean, 0.0)
    return mean, variance**0.5


def dataset_summary(
    labels: Sequence[int] | torch.Tensor,
    splits: SplitIndices,
    test_labels: Sequence[int] | torch.Tensor | None = None,
) -> str:
    labels_array = np.asarray(labels, dtype=np.int64)
    rows = ["split,count," + ",".join(CLASS_NAMES)]
    for name, indices in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("calibration", splits.calibration),
    ):
        counts = np.bincount(labels_array[indices], minlength=len(CLASS_NAMES))
        rows.append(f"{name},{len(indices)}," + ",".join(str(int(value)) for value in counts))
    if test_labels is not None:
        test_labels_array = np.asarray(test_labels, dtype=np.int64)
        counts = np.bincount(test_labels_array, minlength=len(CLASS_NAMES))
        rows.append(
            f"clean_test,{len(test_labels_array)},"
            + ",".join(str(int(value)) for value in counts)
        )
    return "\n".join(rows)


def save_sample_grid(
    raw_training: Dataset,
    train_indices: Sequence[int] | np.ndarray,
    destination: str | Path,
) -> Path:
    labels = np.asarray(raw_training.targets, dtype=np.int64)
    train_indices_array = np.asarray(train_indices, dtype=np.int64)
    selected = []
    for class_id in range(len(CLASS_NAMES)):
        candidates = train_indices_array[labels[train_indices_array] == class_id]
        if len(candidates) == 0:
            raise ValueError(f"Training split has no sample for class {class_id}")
        selected.append(int(candidates[0]))

    figure, axes = plt.subplots(2, 5, figsize=(10, 5.2))
    for axis, index, class_name in zip(axes.flat, selected, CLASS_NAMES):
        image, _ = raw_training[index]
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(class_name, fontsize=9)
        axis.axis("off")
    figure.suptitle("Fashion-MNIST training split samples")
    figure.subplots_adjust(top=0.88, bottom=0.05, hspace=0.32, wspace=0.08)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output
