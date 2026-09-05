from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from clothsense.data import DataLoaders, SplitIndices
from clothsense.evaluation import classification_metrics, collect_logits
from clothsense.model import FashionCNN
from clothsense.training import EarlyStopping, load_checkpoint, run_epoch, train_seed


def tiny_loader(samples: int = 16, batch_size: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(9)
    images = torch.randn(samples, 1, 28, 28, generator=generator)
    labels = torch.arange(samples) % 10
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


def test_cnn_returns_ten_logits(config) -> None:
    model = FashionCNN(config.model)
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_validation_does_not_update_parameters(config) -> None:
    model = FashionCNN(config.model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = run_epoch(model, tiny_loader(), nn.CrossEntropyLoss(), torch.device("cpu"))
    after = model.state_dict()
    assert metrics.loss > 0
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_early_stopping_counts_consecutive_non_improvements() -> None:
    stopping = EarlyStopping(patience=3)
    assert stopping.update(1.0)
    assert not stopping.update(1.1)
    assert not stopping.update(1.05)
    assert not stopping.should_stop
    assert not stopping.update(1.2)
    assert stopping.should_stop


def test_tiny_training_checkpoint_reload_and_reserved_data_are_untouched(config) -> None:
    training = replace(config.training, max_epochs=1, early_stopping_patience=1, batch_size=8)
    test_config = replace(config, training=training)

    class ForbiddenLoader:
        def __iter__(self):
            raise AssertionError("Reserved calibration/test data was consumed during training")

    loader = tiny_loader()
    loaders = DataLoaders(
        train=loader,
        validation=loader,
        calibration=ForbiddenLoader(),
        clean_test=ForbiddenLoader(),
        splits=SplitIndices(np.array([0]), np.array([1]), np.array([2]), 1, 3),
    )
    trained = train_seed(test_config, seed=123, loaders=loaders, device=torch.device("cpu"))
    reloaded, metadata = load_checkpoint(trained.artifacts.checkpoint)
    assert trained.artifacts.checkpoint.is_file()
    assert trained.artifacts.history.is_file()
    assert int(metadata["seed"]) == 123
    assert metadata["config_hash"] == trained.config_hash
    assert int(metadata["best_epoch"]) == trained.fit.best_epoch
    assert torch.equal(
        trained.model.state_dict()["classifier.4.weight"],
        reloaded.state_dict()["classifier.4.weight"],
    )


def test_clean_evaluation_metric_structure(config) -> None:
    model = FashionCNN(config.model)
    logits, labels, predictions = collect_logits(model, tiny_loader(20, 5), torch.device("cpu"))
    metrics = classification_metrics(labels, predictions)
    assert logits.shape == (20, 10)
    assert labels.shape == predictions.shape == (20,)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["per_class"]) == 10
    assert np.asarray(metrics["confusion_matrix"]).shape == (10, 10)

