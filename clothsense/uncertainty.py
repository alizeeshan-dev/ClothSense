from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from .calibration import (
    ProbabilityPredictions,
    fit_temperature,
    softmax_predictions,
    temperature_scaled_predictions,
)
from .conformal import (
    class_conditional_prediction_sets,
    fit_class_conditional_thresholds,
    fit_standard_thresholds,
    prediction_sets,
)


@dataclass(frozen=True)
class UncertaintyOutputs:
    raw: ProbabilityPredictions
    calibrated: ProbabilityPredictions
    standard_sets: list[tuple[int, ...]]
    class_conditional_sets: list[tuple[int, ...]]


@dataclass(frozen=True)
class FittedUncertainty:
    temperature: float
    standard_thresholds: Mapping[float, float]
    class_conditional_thresholds: Mapping[float, Mapping[int, float]]

    def apply(self, logits: torch.Tensor, alpha: float) -> UncertaintyOutputs:
        alpha = float(alpha)
        if alpha not in self.standard_thresholds or alpha not in self.class_conditional_thresholds:
            raise ValueError(f"No fitted conformal thresholds for alpha={alpha}")
        raw = softmax_predictions(logits)
        calibrated = temperature_scaled_predictions(logits, self.temperature)
        return UncertaintyOutputs(
            raw=raw,
            calibrated=calibrated,
            standard_sets=prediction_sets(
                calibrated.probabilities,
                self.standard_thresholds[alpha],
            ),
            class_conditional_sets=class_conditional_prediction_sets(
                calibrated.probabilities,
                self.class_conditional_thresholds[alpha],
            ),
        )


def fit_uncertainty(
    calibration_logits: torch.Tensor,
    calibration_labels: torch.Tensor,
    alpha_values: Iterable[float],
) -> FittedUncertainty:
    alphas = tuple(float(alpha) for alpha in alpha_values)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    calibrated = temperature_scaled_predictions(calibration_logits, temperature)
    return FittedUncertainty(
        temperature=temperature,
        standard_thresholds=fit_standard_thresholds(
            calibrated.probabilities,
            calibration_labels,
            alphas,
        ),
        class_conditional_thresholds=fit_class_conditional_thresholds(
            calibrated.probabilities,
            calibration_labels,
            alphas,
        ),
    )

