from __future__ import annotations

import torch

from clothsense.calibration import (
    fit_temperature,
    softmax_predictions,
    temperature_scaled_predictions,
)


def test_temperature_is_positive_and_probabilities_are_valid() -> None:
    logits = torch.tensor(
        [[4.0, 1.0, -1.0], [0.5, 2.0, -0.5], [-1.0, 0.0, 3.0], [2.0, 1.0, 0.0]]
    )
    labels = torch.tensor([0, 1, 2, 1])
    temperature = fit_temperature(logits, labels, max_iterations=30)
    raw = softmax_predictions(logits)
    calibrated = temperature_scaled_predictions(logits, temperature)

    assert temperature > 0
    assert calibrated.probabilities.shape == logits.shape
    assert torch.allclose(calibrated.probabilities.sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.equal(raw.predicted_classes, calibrated.predicted_classes)


def test_temperature_fit_depends_only_on_passed_calibration_inputs() -> None:
    calibration_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    calibration_labels = torch.tensor([0, 1, 0, 1])
    unrelated_test_logits = torch.randn(20, 2)
    first = fit_temperature(calibration_logits, calibration_labels, max_iterations=20)
    unrelated_test_logits.mul_(1000)
    second = fit_temperature(calibration_logits, calibration_labels, max_iterations=20)
    assert first == second

