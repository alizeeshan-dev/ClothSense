from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from clothsense.data import (
    DataLoaders,
    SplitIndices,
    get_or_create_splits,
    load_split_indices,
    stratified_split_indices,
)
from clothsense.reproducibility import seed_everything


def balanced_labels(samples_per_class: int = 6000) -> np.ndarray:
    return np.repeat(np.arange(10), samples_per_class)


def test_split_sizes_disjointness_and_stratification() -> None:
    labels = balanced_labels()
    splits = stratified_split_indices(labels, 45000, 5000, 10000, seed=2027)

    assert (len(splits.train), len(splits.validation), len(splits.calibration)) == (
        45000,
        5000,
        10000,
    )
    assert not set(splits.train) & set(splits.validation)
    assert not set(splits.train) & set(splits.calibration)
    assert not set(splits.validation) & set(splits.calibration)
    assert np.array_equal(np.bincount(labels[splits.train]), np.full(10, 4500))
    assert np.array_equal(np.bincount(labels[splits.validation]), np.full(10, 500))
    assert np.array_equal(np.bincount(labels[splits.calibration]), np.full(10, 1000))


def test_split_generation_is_reproducible() -> None:
    labels = balanced_labels()
    first = stratified_split_indices(labels, 45000, 5000, 10000, seed=2027)
    second = stratified_split_indices(labels, 45000, 5000, 10000, seed=2027)
    other_seed = stratified_split_indices(labels, 45000, 5000, 10000, seed=2028)

    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.calibration, second.calibration)
    assert not np.array_equal(first.train, other_seed.train)


def test_saved_split_indices_are_reused(config) -> None:
    labels = balanced_labels()
    first = get_or_create_splits(labels, config)
    modified_labels = labels[::-1].copy()
    reused = get_or_create_splits(modified_labels, config)
    loaded = load_split_indices(config.split_file)

    assert config.split_file.is_file()
    assert np.array_equal(first.train, reused.train)
    assert np.array_equal(first.train, loaded.train)
    assert loaded.seed == config.seeds.split


def test_fixed_seed_reproduces_torch_numpy_and_python() -> None:
    import random

    seed_everything(31)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(31)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_official_test_loader_is_separate_and_tensor_shapes_are_valid(config, monkeypatch) -> None:
    from PIL import Image
    from torch.utils.data import Dataset

    import clothsense.data as data_module

    class FakeFashionMNIST(Dataset):
        def __init__(self, root, train, download, transform=None):
            del root, download
            self.train = train
            self.transform = transform
            self.size = 60000 if train else 10000
            self.targets = torch.arange(self.size) % 10

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            image = Image.fromarray(np.full((28, 28), index % 256, dtype=np.uint8), mode="L")
            if self.transform:
                image = self.transform(image)
            return image, int(self.targets[index])

    monkeypatch.setattr(data_module.datasets, "FashionMNIST", FakeFashionMNIST)
    small_training = replace(config.training, batch_size=16)
    test_config = replace(config, training=small_training)
    loaders: DataLoaders = data_module.build_dataloaders(test_config)

    for loader in (
        loaders.train,
        loaders.validation,
        loaders.calibration,
        loaders.clean_test,
    ):
        images, labels = next(iter(loader))
        assert images.shape == (16, 1, 28, 28)
        assert labels.shape == (16,)
        assert torch.isfinite(images).all()
    assert not isinstance(loaders.clean_test.dataset, torch.utils.data.Subset)
    assert loaders.clean_test.dataset.train is False
    assert len(loaders.clean_test.dataset) == 10000
