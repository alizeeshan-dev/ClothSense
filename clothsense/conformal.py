from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch


def _validate_probabilities(probabilities: torch.Tensor) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Probabilities must have shape [samples, classes]")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("Probabilities must lie in [0, 1]")


def nonconformity_scores(
    calibrated_probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    _validate_probabilities(calibrated_probabilities)
    labels = labels.long()
    if labels.ndim != 1 or len(labels) != len(calibrated_probabilities):
        raise ValueError("Labels must align with probabilities")
    if labels.numel() == 0:
        raise ValueError("At least one calibration sample is required")
    if labels.min() < 0 or labels.max() >= calibrated_probabilities.shape[1]:
        raise ValueError("Labels contain invalid class IDs")
    true_probabilities = calibrated_probabilities.gather(1, labels[:, None]).squeeze(1)
    return 1.0 - true_probabilities


def corrected_conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("Scores must be a non-empty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Alpha must lie strictly between zero and one")
    if not torch.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    sample_count = scores.numel()
    rank = math.ceil((sample_count + 1) * (1.0 - alpha))
    safe_rank = min(max(rank, 1), sample_count)
    return float(torch.sort(scores).values[safe_rank - 1].item())


def fit_standard_thresholds(
    calibrated_probabilities: torch.Tensor,
    labels: torch.Tensor,
    alpha_values: Iterable[float],
) -> dict[float, float]:
    scores = nonconformity_scores(calibrated_probabilities, labels)
    return {
        float(alpha): corrected_conformal_quantile(scores, float(alpha))
        for alpha in alpha_values
    }


def prediction_sets(
    calibrated_probabilities: torch.Tensor,
    threshold: float,
) -> list[tuple[int, ...]]:
    _validate_probabilities(calibrated_probabilities)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Conformal threshold must lie in [0, 1]")
    included = (1.0 - calibrated_probabilities) <= threshold
    return [
        tuple(int(class_id) for class_id in torch.where(row)[0].tolist())
        for row in included
    ]


def fit_class_conditional_thresholds(
    calibrated_probabilities: torch.Tensor,
    labels: torch.Tensor,
    alpha_values: Iterable[float],
) -> dict[float, dict[int, float]]:
    scores = nonconformity_scores(calibrated_probabilities, labels)
    labels = labels.long()
    class_count = calibrated_probabilities.shape[1]
    result: dict[float, dict[int, float]] = {}
    for alpha_value in alpha_values:
        alpha = float(alpha_value)
        thresholds: dict[int, float] = {}
        for class_id in range(class_count):
            class_scores = scores[labels == class_id]
            if class_scores.numel() == 0:
                raise ValueError(f"Calibration data has no examples for class {class_id}")
            thresholds[class_id] = corrected_conformal_quantile(class_scores, alpha)
        result[alpha] = thresholds
    return result


def class_conditional_prediction_sets(
    calibrated_probabilities: torch.Tensor,
    thresholds: Mapping[int, float],
) -> list[tuple[int, ...]]:
    _validate_probabilities(calibrated_probabilities)
    class_count = calibrated_probabilities.shape[1]
    if set(thresholds) != set(range(class_count)):
        raise ValueError("Class-conditional thresholds must cover every class exactly once")
    threshold_vector = torch.tensor(
        [thresholds[class_id] for class_id in range(class_count)],
        dtype=calibrated_probabilities.dtype,
        device=calibrated_probabilities.device,
    )
    if torch.any(threshold_vector < 0) or torch.any(threshold_vector > 1):
        raise ValueError("Class-conditional thresholds must lie in [0, 1]")
    included = (1.0 - calibrated_probabilities) <= threshold_vector[None, :]
    return [
        tuple(int(class_id) for class_id in torch.where(row)[0].tolist())
        for row in included
    ]

