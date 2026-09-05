from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from clothsense.shifts import build_shift_scenarios, find_scenario


class TinyRawFashion(Dataset):
    def __init__(self, samples_per_class: int = 10) -> None:
        self.targets = torch.arange(10).repeat_interleave(samples_per_class)
        base = torch.linspace(0, 255, 28, dtype=torch.uint8).repeat(28, 1)
        self.data = torch.stack([torch.roll(base, int(index), dims=1) for index in range(len(self.targets))])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.data[index], int(self.targets[index])


def test_all_shift_scenarios_preserve_shape_labels_range_and_source(config) -> None:
    source = TinyRawFashion()
    original = source.data.clone()
    scenarios = build_shift_scenarios(source, config, seed=31)
    assert len(scenarios) == 14
    for generated in scenarios:
        image, label = generated.dataset.raw_item(0)
        original_index = generated.dataset.indices[0]
        assert image.shape == (1, 28, 28)
        assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
        assert label == int(source.targets[original_index])
        assert generated.scenario.sample_count == len(generated.dataset)
    assert torch.equal(source.data, original)


def test_gaussian_noise_is_deterministic_and_severity_is_configured(config) -> None:
    source = TinyRawFashion()
    first = build_shift_scenarios(source, config, seed=7)
    second = build_shift_scenarios(source, config, seed=7)
    mild_a = find_scenario(first, "gaussian_noise", "mild")
    mild_b = find_scenario(second, "gaussian_noise", "mild")
    strong = find_scenario(first, "gaussian_noise", "strong")
    clean = source.data[0].float().unsqueeze(0) / 255.0
    mild_image, _ = mild_a.dataset.raw_item(0)
    repeated_image, _ = mild_b.dataset.raw_item(0)
    strong_image, _ = strong.dataset.raw_item(0)
    assert torch.equal(mild_image, repeated_image)
    assert mild_a.scenario.parameters["std"] == config.shifts["gaussian_noise"]["mild"]
    assert strong.scenario.parameters["std"] == config.shifts["gaussian_noise"]["strong"]
    assert torch.mean(torch.abs(strong_image - clean)) > torch.mean(torch.abs(mild_image - clean))


def test_rotation_blur_and_brightness_severities_come_from_config(config) -> None:
    scenarios = build_shift_scenarios(TinyRawFashion(), config, seed=4)
    rotations = [
        find_scenario(scenarios, "rotation", severity).scenario.parameters["degrees"]
        for severity in ("mild", "medium", "strong")
    ]
    blurs = [
        find_scenario(scenarios, "gaussian_blur", severity).scenario.parameters["sigma"]
        for severity in ("mild", "medium", "strong")
    ]
    assert rotations == sorted(rotations) and len(set(rotations)) == 3
    assert blurs == sorted(blurs) and len(set(blurs)) == 3
    assert find_scenario(scenarios, "rotation", "strong").scenario.parameters["degrees"] == 30
    assert find_scenario(scenarios, "gaussian_blur", "strong").scenario.parameters["sigma"] == 1.5
    contrast = find_scenario(scenarios, "brightness_contrast", "increased_contrast")
    assert contrast.scenario.parameters == config.shifts["brightness_contrast"]["increased_contrast"]


def test_class_imbalance_counts_labels_and_seeded_selection(config) -> None:
    source = TinyRawFashion()
    first = build_shift_scenarios(source, config, seed=12)
    repeated = build_shift_scenarios(source, config, seed=12)
    other_seed = build_shift_scenarios(source, config, seed=13)
    moderate = find_scenario(first, "class_imbalance", "moderate")
    strong = find_scenario(first, "class_imbalance", "strong")
    repeated_moderate = find_scenario(repeated, "class_imbalance", "moderate")
    other_moderate = find_scenario(other_seed, "class_imbalance", "moderate")
    assert len(moderate.dataset) == 70
    assert len(strong.dataset) == 52
    assert moderate.dataset.indices == repeated_moderate.dataset.indices
    assert moderate.dataset.indices != other_moderate.dataset.indices
    labels = source.targets[list(moderate.dataset.indices)]
    counts = torch.bincount(labels, minlength=10)
    assert torch.equal(counts[[0, 1, 8, 9]], torch.full((4,), 10))
    assert torch.equal(counts[[2, 3, 4, 5, 6, 7]], torch.full((6,), 5))
