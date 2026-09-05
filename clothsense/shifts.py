from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import ProjectConfig


@dataclass(frozen=True)
class ShiftScenario:
    shift_name: str
    severity_name: str
    parameters: Mapping[str, Any]
    seed: int
    sample_count: int


@dataclass(frozen=True)
class GeneratedScenario:
    scenario: ShiftScenario
    dataset: "ShiftedDataset"


RawTransform = Callable[[torch.Tensor, int], torch.Tensor]


def _raw_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        else:
            tensor = tensor.to(torch.float32)
    else:
        tensor = TF.to_tensor(image)
    if tensor.shape != (1, 28, 28):
        raise ValueError(f"Expected a raw 1x28x28 image, received {tuple(tensor.shape)}")
    if torch.any(tensor < 0) or torch.any(tensor > 1):
        raise ValueError("Raw images must lie in [0, 1]")
    return tensor


def _sample_seed(seed: int, original_index: int) -> int:
    return (int(seed) * 1_000_003 + int(original_index) * 97_409 + 17) % (2**63 - 1)


class ShiftedDataset(Dataset):
    def __init__(
        self,
        source: Dataset,
        indices: Sequence[int],
        raw_transform: RawTransform,
        normalization_mean: Sequence[float],
        normalization_std: Sequence[float],
    ) -> None:
        self.source = source
        self.indices = tuple(int(index) for index in indices)
        self.raw_transform = raw_transform
        self.normalization_mean = tuple(float(value) for value in normalization_mean)
        self.normalization_std = tuple(float(value) for value in normalization_std)

    def __len__(self) -> int:
        return len(self.indices)

    def raw_item(self, index: int) -> tuple[torch.Tensor, int]:
        original_index = self.indices[index]
        image, label = self.source[original_index]
        shifted = self.raw_transform(_raw_tensor(image), original_index).to(torch.float32)
        if shifted.shape != (1, 28, 28):
            raise ValueError("A shift changed the Fashion-MNIST image shape")
        shifted = shifted.clamp(0.0, 1.0)
        return shifted, int(label)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        shifted, label = self.raw_item(index)
        normalized = TF.normalize(shifted, self.normalization_mean, self.normalization_std)
        return normalized, label


def identity_shift(image: torch.Tensor, original_index: int) -> torch.Tensor:
    del original_index
    return image.clone()


def gaussian_noise_shift(std: float, seed: int) -> RawTransform:
    if std < 0:
        raise ValueError("Noise standard deviation cannot be negative")

    def transform(image: torch.Tensor, original_index: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_sample_seed(seed, original_index))
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        return (image + noise * std).clamp(0.0, 1.0)

    return transform


def rotation_shift(degrees: float) -> RawTransform:
    def transform(image: torch.Tensor, original_index: int) -> torch.Tensor:
        del original_index
        return TF.rotate(
            image,
            angle=degrees,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0,
        ).clamp(0.0, 1.0)

    return transform


def gaussian_blur_shift(sigma: float) -> RawTransform:
    if sigma <= 0:
        raise ValueError("Blur sigma must be positive")
    kernel_size = 2 * math.ceil(3 * sigma) + 1

    def transform(image: torch.Tensor, original_index: int) -> torch.Tensor:
        del original_index
        return TF.gaussian_blur(image, [kernel_size, kernel_size], [sigma, sigma]).clamp(0.0, 1.0)

    return transform


def brightness_contrast_shift(brightness: float, contrast: float) -> RawTransform:
    if brightness < 0 or contrast < 0:
        raise ValueError("Brightness and contrast factors cannot be negative")

    def transform(image: torch.Tensor, original_index: int) -> torch.Tensor:
        del original_index
        adjusted = TF.adjust_brightness(image, brightness)
        adjusted = TF.adjust_contrast(adjusted, contrast)
        return adjusted.clamp(0.0, 1.0)

    return transform


def class_imbalance_indices(
    labels: Sequence[int] | torch.Tensor,
    majority_classes: Sequence[int],
    minority_fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 <= minority_fraction <= 1.0:
        raise ValueError("Minority fraction must lie in [0, 1]")
    labels_array = np.asarray(labels, dtype=np.int64)
    majority = {int(class_id) for class_id in majority_classes}
    if not majority:
        raise ValueError("At least one majority class is required")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in sorted(np.unique(labels_array)):
        candidates = np.flatnonzero(labels_array == class_id)
        if int(class_id) in majority:
            chosen = candidates
        else:
            retained = int(round(len(candidates) * minority_fraction))
            chosen = rng.choice(candidates, size=retained, replace=False)
        selected.append(np.asarray(chosen, dtype=np.int64))
    return np.sort(np.concatenate(selected))


def _dataset_labels(dataset: Dataset) -> np.ndarray:
    if not hasattr(dataset, "targets"):
        raise TypeError("The clean test dataset must expose targets")
    return np.asarray(dataset.targets, dtype=np.int64)


def build_shift_scenarios(
    clean_test_dataset: Dataset,
    config: ProjectConfig,
    *,
    seed: int,
) -> list[GeneratedScenario]:
    all_indices = np.arange(len(clean_test_dataset), dtype=np.int64)
    normalization = (config.data.normalization_mean, config.data.normalization_std)
    generated: list[GeneratedScenario] = []

    def add(
        shift_name: str,
        severity_name: str,
        parameters: Mapping[str, Any],
        transform: RawTransform,
        indices: Sequence[int] = all_indices,
    ) -> None:
        scenario = ShiftScenario(
            shift_name=shift_name,
            severity_name=severity_name,
            parameters=dict(parameters),
            seed=seed,
            sample_count=len(indices),
        )
        dataset = ShiftedDataset(clean_test_dataset, indices, transform, *normalization)
        generated.append(GeneratedScenario(scenario, dataset))

    for severity, value in config.shifts["gaussian_noise"].items():
        std = float(value)
        add("gaussian_noise", severity, {"std": std}, gaussian_noise_shift(std, seed))
    for severity, value in config.shifts["rotation_degrees"].items():
        degrees = float(value)
        add("rotation", severity, {"degrees": degrees}, rotation_shift(degrees))
    for severity, value in config.shifts["gaussian_blur_sigma"].items():
        sigma = float(value)
        add("gaussian_blur", severity, {"sigma": sigma}, gaussian_blur_shift(sigma))
    for severity, values in config.shifts["brightness_contrast"].items():
        brightness = float(values["brightness"])
        contrast = float(values["contrast"])
        add(
            "brightness_contrast",
            severity,
            {"brightness": brightness, "contrast": contrast},
            brightness_contrast_shift(brightness, contrast),
        )

    imbalance = config.shifts["class_imbalance"]
    majority_classes = tuple(int(value) for value in imbalance["majority_classes"])
    labels = _dataset_labels(clean_test_dataset)
    for severity in ("moderate", "strong"):
        fraction = float(imbalance[f"{severity}_minority_fraction"])
        indices = class_imbalance_indices(labels, majority_classes, fraction, seed)
        add(
            "class_imbalance",
            severity,
            {"majority_classes": list(majority_classes), "minority_fraction": fraction},
            identity_shift,
            indices,
        )
    return generated


def find_scenario(
    scenarios: Sequence[GeneratedScenario],
    shift_name: str,
    severity_name: str,
) -> GeneratedScenario:
    matches = [
        item
        for item in scenarios
        if item.scenario.shift_name == shift_name
        and item.scenario.severity_name == severity_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one scenario for {shift_name}/{severity_name}")
    return matches[0]

