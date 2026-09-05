from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import ModelConfig, ProjectConfig
from .data import DataLoaders
from .model import FashionCNN
from .reproducibility import seed_everything, select_device
from .storage import record_model_metadata


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class TrainingHistory:
    epoch: list[int]
    train_loss: list[float]
    validation_loss: list[float]
    train_accuracy: list[float]
    validation_accuracy: list[float]
    epoch_seconds: list[float]

    @classmethod
    def empty(cls) -> "TrainingHistory":
        return cls([], [], [], [], [], [])

    def as_dict(self) -> dict[str, list[int] | list[float]]:
        return asdict(self)


@dataclass(frozen=True)
class FitResult:
    history: TrainingHistory
    best_epoch: int
    validation_loss: float
    validation_accuracy: float
    stopped_early: bool


@dataclass(frozen=True)
class SeedArtifacts:
    directory: Path
    checkpoint: Path
    history: Path
    clean_outputs: Path
    calibration_outputs: Path
    uncertainty: Path
    clean_metrics: Path
    training_curves: Path
    confusion_matrix: Path


@dataclass(frozen=True)
class TrainedSeed:
    model: FashionCNN
    fit: FitResult
    artifacts: SeedArtifacts
    config_hash: str
    model_id: int


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("Early-stopping patience must be at least one")
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def update(self, validation_loss: float) -> bool:
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def seed_artifact_paths(config: ProjectConfig, seed: int) -> SeedArtifacts:
    directory = config.paths.models / f"seed_{seed}"
    plot_directory = config.paths.plots / f"seed_{seed}"
    return SeedArtifacts(
        directory=directory,
        checkpoint=directory / "best_model.pt",
        history=directory / "training_history.json",
        clean_outputs=directory / "clean_test_outputs.npz",
        calibration_outputs=directory / "calibration_outputs.npz",
        uncertainty=directory / "uncertainty.json",
        clean_metrics=config.paths.results / f"clean_baseline_seed_{seed}.json",
        training_curves=plot_directory / "training_curves.png",
        confusion_matrix=plot_directory / "clean_confusion_matrix.png",
    )


def relevant_config_snapshot(config: ProjectConfig) -> dict[str, object]:
    return {
        "model": asdict(config.model),
        "training": asdict(config.training),
        "data": {
            "train_size": config.data.train_size,
            "validation_size": config.data.validation_size,
            "calibration_size": config.data.calibration_size,
            "test_size": config.data.test_size,
            "normalization_mean": config.data.normalization_mean,
            "normalization_std": config.data.normalization_std,
        },
        "split_seed": config.seeds.split,
    }


def configuration_hash(config: ProjectConfig) -> str:
    serialized = json.dumps(relevant_config_snapshot(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Cannot run an epoch over an empty DataLoader")
    return EpochMetrics(total_loss / total_samples, total_correct / total_samples)


def _checkpoint_payload(
    model: nn.Module,
    config: ProjectConfig,
    seed: int,
    epoch: int,
    validation: EpochMetrics,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "seed": seed,
        "config_hash": configuration_hash(config),
        "model_config": asdict(config.model),
        "training_config": asdict(config.training),
        "split_seed": config.seeds.split,
        "best_epoch": epoch,
        "validation_loss": validation.loss,
        "validation_accuracy": validation.accuracy,
        "model_state_dict": model.state_dict(),
    }


def save_checkpoint(payload: dict[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[FashionCNN, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model_values = checkpoint["model_config"]
    if not isinstance(model_values, dict):
        raise ValueError("Checkpoint model configuration is invalid")
    model_config = ModelConfig(
        input_channels=int(model_values["input_channels"]),
        conv_channels=tuple(int(value) for value in model_values["conv_channels"]),
        kernel_size=int(model_values["kernel_size"]),
        padding=int(model_values["padding"]),
        pool_size=int(model_values["pool_size"]),
        hidden_units=int(model_values["hidden_units"]),
        dropout=float(model_values["dropout"]),
        num_classes=int(model_values["num_classes"]),
    )
    model = FashionCNN(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def fit_model(
    model: FashionCNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ProjectConfig,
    seed: int,
    checkpoint_path: str | Path,
    device: torch.device,
    on_epoch: Callable[[int, EpochMetrics, EpochMetrics, float], None] | None = None,
) -> FitResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    early_stopping = EarlyStopping(config.training.early_stopping_patience)
    history = TrainingHistory.empty()
    best_epoch = 0
    best_validation = EpochMetrics(float("inf"), 0.0)

    for epoch in range(1, config.training.max_epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        elapsed = time.perf_counter() - start
        history.epoch.append(epoch)
        history.train_loss.append(train_metrics.loss)
        history.validation_loss.append(validation_metrics.loss)
        history.train_accuracy.append(train_metrics.accuracy)
        history.validation_accuracy.append(validation_metrics.accuracy)
        history.epoch_seconds.append(elapsed)

        if early_stopping.update(validation_metrics.loss):
            best_epoch = epoch
            best_validation = validation_metrics
            save_checkpoint(
                _checkpoint_payload(model, config, seed, epoch, validation_metrics),
                checkpoint_path,
            )
        if on_epoch is not None:
            on_epoch(epoch, train_metrics, validation_metrics, elapsed)
        if early_stopping.should_stop:
            break

    selected_model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(selected_model.state_dict())
    if int(checkpoint["best_epoch"]) != best_epoch:
        raise RuntimeError("Best checkpoint metadata is inconsistent")
    return FitResult(
        history=history,
        best_epoch=best_epoch,
        validation_loss=best_validation.loss,
        validation_accuracy=best_validation.accuracy,
        stopped_early=len(history.epoch) < config.training.max_epochs,
    )


def train_seed(
    config: ProjectConfig,
    seed: int,
    loaders: DataLoaders,
    *,
    device: torch.device | None = None,
    on_epoch: Callable[[int, EpochMetrics, EpochMetrics, float], None] | None = None,
) -> TrainedSeed:
    config.ensure_directories()
    seed_everything(seed, config.device.deterministic_algorithms)
    selected_device = device or select_device(config.device)
    model = FashionCNN(config.model).to(selected_device)
    artifacts = seed_artifact_paths(config, seed)
    artifacts.directory.mkdir(parents=True, exist_ok=True)
    fit = fit_model(
        model,
        loaders.train,
        loaders.validation,
        config,
        seed,
        artifacts.checkpoint,
        selected_device,
        on_epoch,
    )
    run_hash = configuration_hash(config)
    history_payload = {
        "seed": seed,
        "config_hash": run_hash,
        "checkpoint_path": str(artifacts.checkpoint),
        "best_epoch": fit.best_epoch,
        "best_validation_loss": fit.validation_loss,
        "best_validation_accuracy": fit.validation_accuracy,
        "stopped_early": fit.stopped_early,
        "history": fit.history.as_dict(),
    }
    artifacts.history.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    model_id = record_model_metadata(
        config.paths.database,
        {
            "seed": seed,
            "config_hash": run_hash,
            "checkpoint_path": str(artifacts.checkpoint.relative_to(config.root)),
            "best_epoch": fit.best_epoch,
            "validation_loss": fit.validation_loss,
            "validation_accuracy": fit.validation_accuracy,
        },
    )
    return TrainedSeed(model, fit, artifacts, run_hash, model_id)
