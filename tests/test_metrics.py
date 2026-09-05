from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from clothsense.abstention import AbstentionDecision
from clothsense.metrics import (
    classification_metrics,
    conformal_metrics,
    probability_metrics,
    risk_coverage_data,
    selective_prediction_metrics,
)


def ten_class_probabilities() -> tuple[np.ndarray, np.ndarray]:
    leading = np.asarray([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]])
    probabilities = np.pad(leading, ((0, 0), (0, 8)))
    return probabilities, np.asarray([0, 1, 1, 0])


def test_classification_metrics_match_sklearn_with_absent_classes() -> None:
    labels = np.asarray([0, 0, 1, 1, 2])
    predictions = np.asarray([0, 1, 1, 1, 0])
    metrics = classification_metrics(labels, predictions)
    assert metrics["accuracy"] == accuracy_score(labels, predictions)
    assert metrics["macro_f1"] == f1_score(
        labels, predictions, labels=np.arange(10), average="macro", zero_division=0
    )
    assert len(metrics["per_class"]) == 10
    assert metrics["per_class"][9]["support"] == 0
    assert np.asarray(metrics["confusion_matrix"]).shape == (10, 10)


def test_probability_metrics_and_reliability_bins_match_hand_calculation() -> None:
    probabilities, labels = ten_class_probabilities()
    result = probability_metrics(probabilities, labels, [0.0, 0.5, 0.75, 1.0])
    aggregate = result["aggregate"]
    expected_nll = -np.log([0.8, 0.6, 0.3, 0.1]).mean()
    one_hot = np.eye(10)[labels]
    expected_brier = np.square(probabilities - one_hot).sum(axis=1).mean()
    assert aggregate["negative_log_likelihood"] == pytest.approx(expected_nll)
    assert aggregate["brier_score"] == pytest.approx(expected_brier)
    assert aggregate["expected_calibration_error"] == pytest.approx(0.25)
    assert aggregate["maximum_calibration_error"] == pytest.approx(0.35)
    assert aggregate["mean_confidence"] == pytest.approx(0.75)
    assert aggregate["accuracy_confidence_gap"] == pytest.approx(-0.25)
    bins = result["reliability_bins"]
    assert bins[0]["sample_count"] == 0 and bins[0]["average_confidence"] is None
    assert bins[1]["sample_count"] == 2 and bins[1]["empirical_accuracy"] == 0.5
    assert bins[2]["sample_count"] == 2 and bins[2]["average_confidence"] == pytest.approx(0.85)


def test_conformal_metrics_match_manual_sets() -> None:
    labels = np.asarray([0, 1, 2, 1])
    prediction_sets = [(0,), (), (1, 2), (0, 1, 2)]
    result = conformal_metrics(prediction_sets, labels, target_coverage=0.9)
    aggregate = result["aggregate"]
    assert aggregate["empirical_coverage"] == 0.75
    assert aggregate["coverage_gap"] == pytest.approx(-0.15)
    assert aggregate["average_set_size"] == 1.5
    assert aggregate["median_set_size"] == 1.5
    assert aggregate["empty_set_rate"] == 0.25
    assert aggregate["singleton_set_rate"] == 0.25
    assert aggregate["multiclass_set_rate"] == 0.5
    assert aggregate["worst_class_coverage"] == 0.5
    assert result["per_class"][1]["coverage"] == 0.5
    assert result["per_class"][1]["average_set_size"] == 1.5
    assert result["per_class"][9]["coverage"] is None


def test_selective_metrics_and_zero_acceptance() -> None:
    labels = np.asarray([0, 1, 0])
    decisions = [
        AbstentionDecision(True, 0, "accepted"),
        AbstentionDecision(False, None, "abstained"),
        AbstentionDecision(True, 1, "accepted"),
    ]
    metrics = selective_prediction_metrics(labels, decisions)
    assert metrics["number_accepted"] == 2
    assert metrics["number_abstained"] == 1
    assert metrics["acceptance_rate"] == pytest.approx(2 / 3)
    assert metrics["abstention_rate"] == pytest.approx(1 / 3)
    assert metrics["selective_accuracy"] == 0.5
    assert metrics["selective_risk"] == 0.5

    none = [AbstentionDecision(False, None, "abstained") for _ in labels]
    zero = selective_prediction_metrics(labels, none)
    assert zero["number_accepted"] == 0
    assert zero["selective_accuracy"] is None
    assert zero["selective_risk"] is None
    points = risk_coverage_data(labels, [("example", None, none), ("example", 0.8, decisions)])
    assert points[0]["coverage"] == 0.0 and points[0]["selective_risk"] is None
    assert points[1]["coverage"] == pytest.approx(2 / 3)

