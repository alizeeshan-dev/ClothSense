from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class PathConfig:
    data: Path
    split_indices: Path
    models: Path
    plots: Path
    processed_examples: Path
    personal_photos: Path
    results: Path
    database: Path

    def directories(self) -> tuple[Path, ...]:
        return (
            self.data,
            self.split_indices,
            self.models,
            self.plots,
            self.processed_examples,
            self.personal_photos,
            self.results,
        )


@dataclass(frozen=True)
class SeedConfig:
    split: int
    dataloader: int
    experiments: tuple[int, ...]


@dataclass(frozen=True)
class TrainingConfig:
    loss: str
    optimizer: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    model_selection_metric: str
    num_workers: int
    pin_memory: bool
    augmentation: str


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    conv_channels: tuple[int, int]
    kernel_size: int
    padding: int
    pool_size: int
    hidden_units: int
    dropout: float
    num_classes: int


@dataclass(frozen=True)
class DataConfig:
    train_size: int
    validation_size: int
    calibration_size: int
    test_size: int
    image_size: tuple[int, int]
    channels: int
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]


@dataclass(frozen=True)
class UncertaintyConfig:
    alpha_values: tuple[float, ...]
    calibration_bin_edges: tuple[float, ...]
    abstention_thresholds: tuple[float, ...]
    upload_demo_alpha: float
    upload_demo_abstention_policy: str | None
    upload_demo_abstention_threshold: float | None


@dataclass(frozen=True)
class DeviceConfig:
    prefer_cuda: bool
    cuda_index: int
    deterministic_algorithms: bool


@dataclass(frozen=True)
class InferenceConfig:
    model_seed: int
    max_upload_bytes: int
    default_invert: bool


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    source_path: Path
    paths: PathConfig
    seeds: SeedConfig
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    uncertainty: UncertaintyConfig
    shifts: Mapping[str, Any]
    device: DeviceConfig
    inference: InferenceConfig

    @property
    def split_file(self) -> Path:
        return self.paths.split_indices / f"fashion_mnist_seed_{self.seeds.split}.npz"

    def ensure_directories(self) -> None:
        for directory in self.paths.directories():
            directory.mkdir(parents=True, exist_ok=True)


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path | None = None) -> ProjectConfig:
    source_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not source_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source_path}")

    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    root_value = raw.get("project_root")
    root = _resolve_path(source_path.parent, root_value) if root_value else PROJECT_ROOT

    path_values = _required(raw, "paths")
    paths = PathConfig(
        **{key: _resolve_path(root, _required(path_values, key)) for key in PathConfig.__annotations__}
    )

    seed_values = _required(raw, "seeds")
    seeds = SeedConfig(
        split=int(_required(seed_values, "split")),
        dataloader=int(_required(seed_values, "dataloader")),
        experiments=tuple(int(seed) for seed in _required(seed_values, "experiments")),
    )

    training_values = _required(raw, "training")
    training = TrainingConfig(
        loss=str(_required(training_values, "loss")),
        optimizer=str(_required(training_values, "optimizer")),
        learning_rate=float(_required(training_values, "learning_rate")),
        batch_size=int(_required(training_values, "batch_size")),
        max_epochs=int(_required(training_values, "max_epochs")),
        early_stopping_patience=int(_required(training_values, "early_stopping_patience")),
        model_selection_metric=str(_required(training_values, "model_selection_metric")),
        num_workers=int(_required(training_values, "num_workers")),
        pin_memory=bool(_required(training_values, "pin_memory")),
        augmentation=str(_required(training_values, "augmentation")),
    )

    model_values = _required(raw, "model")
    model = ModelConfig(
        input_channels=int(_required(model_values, "input_channels")),
        conv_channels=tuple(int(value) for value in _required(model_values, "conv_channels")),
        kernel_size=int(_required(model_values, "kernel_size")),
        padding=int(_required(model_values, "padding")),
        pool_size=int(_required(model_values, "pool_size")),
        hidden_units=int(_required(model_values, "hidden_units")),
        dropout=float(_required(model_values, "dropout")),
        num_classes=int(_required(model_values, "num_classes")),
    )

    data_values = _required(raw, "data")
    normalization = _required(data_values, "normalization")
    data = DataConfig(
        train_size=int(_required(data_values, "train_size")),
        validation_size=int(_required(data_values, "validation_size")),
        calibration_size=int(_required(data_values, "calibration_size")),
        test_size=int(_required(data_values, "test_size")),
        image_size=tuple(int(value) for value in _required(data_values, "image_size")),
        channels=int(_required(data_values, "channels")),
        normalization_mean=tuple(float(value) for value in _required(normalization, "mean")),
        normalization_std=tuple(float(value) for value in _required(normalization, "std")),
    )

    uncertainty_values = _required(raw, "uncertainty")
    uncertainty = UncertaintyConfig(
        alpha_values=tuple(float(value) for value in _required(uncertainty_values, "alpha_values")),
        calibration_bin_edges=tuple(
            float(value) for value in _required(uncertainty_values, "calibration_bin_edges")
        ),
        abstention_thresholds=tuple(
            float(value) for value in _required(uncertainty_values, "abstention_thresholds")
        ),
        upload_demo_alpha=float(_required(uncertainty_values, "upload_demo_alpha")),
        upload_demo_abstention_policy=(
            None
            if _required(uncertainty_values, "upload_demo_abstention_policy") is None
            else str(_required(uncertainty_values, "upload_demo_abstention_policy"))
        ),
        upload_demo_abstention_threshold=(
            None
            if _required(uncertainty_values, "upload_demo_abstention_threshold") is None
            else float(_required(uncertainty_values, "upload_demo_abstention_threshold"))
        ),
    )

    device_values = _required(raw, "device")
    device = DeviceConfig(
        prefer_cuda=bool(_required(device_values, "prefer_cuda")),
        cuda_index=int(_required(device_values, "cuda_index")),
        deterministic_algorithms=bool(_required(device_values, "deterministic_algorithms")),
    )

    inference_values = _required(raw, "inference")
    inference = InferenceConfig(
        model_seed=int(_required(inference_values, "model_seed")),
        max_upload_bytes=int(_required(inference_values, "max_upload_bytes")),
        default_invert=bool(_required(inference_values, "default_invert")),
    )

    config = ProjectConfig(
        root=root,
        source_path=source_path,
        paths=paths,
        seeds=seeds,
        training=training,
        model=model,
        data=data,
        uncertainty=uncertainty,
        shifts=_required(raw, "shifts"),
        device=device,
        inference=inference,
    )
    _validate_config(config)
    return config


def _validate_config(config: ProjectConfig) -> None:
    if config.data.image_size != (28, 28) or config.data.channels != 1:
        raise ValueError("Fashion-MNIST must use one 28x28 channel")
    if min(config.data.normalization_std) <= 0:
        raise ValueError("Normalization standard deviations must be positive")
    if config.training.batch_size <= 0 or config.training.num_workers < 0:
        raise ValueError("Invalid DataLoader configuration")
    if config.training.loss != "cross_entropy" or config.training.optimizer != "adam":
        raise ValueError("The fixed training pipeline requires cross_entropy and Adam")
    if config.training.model_selection_metric != "validation_loss":
        raise ValueError("Checkpoint selection must use validation loss")
    if config.training.augmentation != "none":
        raise ValueError("Phase 1 supports only deterministic augmentation='none'")
    if config.model.input_channels != 1 or config.model.num_classes != 10:
        raise ValueError("The fixed Fashion-MNIST model requires one input channel and ten classes")
    if config.model.conv_channels != (32, 64) or not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("Invalid fixed CNN configuration")
    if tuple(sorted(config.uncertainty.calibration_bin_edges)) != config.uncertainty.calibration_bin_edges:
        raise ValueError("Calibration bin edges must be sorted")
    if config.uncertainty.calibration_bin_edges[0] != 0.0 or config.uncertainty.calibration_bin_edges[-1] != 1.0:
        raise ValueError("Calibration bins must span [0, 1]")
    if any(not 0.0 < alpha < 1.0 for alpha in config.uncertainty.alpha_values):
        raise ValueError("Conformal alpha values must lie strictly between zero and one")
    if any(not 0.0 <= value <= 1.0 for value in config.uncertainty.abstention_thresholds):
        raise ValueError("Abstention thresholds must lie in [0, 1]")
    if config.uncertainty.upload_demo_alpha not in config.uncertainty.alpha_values:
        raise ValueError("Upload-demo alpha must be one of the configured alpha values")
    policy = config.uncertainty.upload_demo_abstention_policy
    if policy not in {None, "confidence_threshold", "singleton_conformal", "hybrid"}:
        raise ValueError("Unknown upload-demo abstention policy")
    threshold = config.uncertainty.upload_demo_abstention_threshold
    if policy in {"confidence_threshold", "hybrid"} and threshold not in config.uncertainty.abstention_thresholds:
        raise ValueError("The selected upload-demo policy requires a configured threshold")
    if policy == "singleton_conformal" and threshold is not None:
        raise ValueError("Singleton-conformal abstention does not use a confidence threshold")
    if config.inference.model_seed not in config.seeds.experiments:
        raise ValueError("Inference model seed must be one of the configured experiment seeds")
    if config.inference.max_upload_bytes <= 0:
        raise ValueError("Upload file-size limit must be positive")
