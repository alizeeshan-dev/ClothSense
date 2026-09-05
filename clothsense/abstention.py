from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AbstentionDecision:
    accepted: bool
    predicted_class: int | None
    reason: str


def confidence_threshold_decision(
    predicted_class: int,
    calibrated_top_probability: float,
    threshold: float,
) -> AbstentionDecision:
    if not 0.0 <= calibrated_top_probability <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("Probabilities and thresholds must lie in [0, 1]")
    if calibrated_top_probability >= threshold:
        return AbstentionDecision(True, int(predicted_class), "confidence_threshold_met")
    return AbstentionDecision(False, None, "below_confidence_threshold")


def singleton_set_decision(conformal_set: Sequence[int]) -> AbstentionDecision:
    if len(conformal_set) == 1:
        return AbstentionDecision(True, int(conformal_set[0]), "singleton_set")
    reason = "empty_conformal_set" if len(conformal_set) == 0 else "multiclass_conformal_set"
    return AbstentionDecision(False, None, reason)


def hybrid_decision(
    predicted_class: int,
    calibrated_top_probability: float,
    conformal_set: Sequence[int],
    threshold: float,
) -> AbstentionDecision:
    singleton = singleton_set_decision(conformal_set)
    if not singleton.accepted:
        return singleton
    confidence = confidence_threshold_decision(
        predicted_class,
        calibrated_top_probability,
        threshold,
    )
    if not confidence.accepted:
        return confidence
    return AbstentionDecision(True, int(predicted_class), "hybrid_conditions_met")

