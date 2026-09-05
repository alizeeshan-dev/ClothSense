from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .abstention import AbstentionDecision
from .data import CLASS_NAMES


def _numpy(values: np.ndarray | torch.Tensor, dtype: Any | None = None) -> np.ndarray:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _validate_labels(labels: np.ndarray, sample_count: int) -> None:
    if labels.ndim != 1 or len(labels) != sample_count or sample_count == 0:
        raise ValueError("Labels must be a non-empty vector aligned with samples")
    if labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
        raise ValueError("Labels contain invalid Fashion-MNIST class IDs")


def classification_metrics(
    labels: np.ndarray | torch.Tensor,
    predictions: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    labels_array = _numpy(labels, np.int64)
    predictions_array = _numpy(predictions, np.int64)
    _validate_labels(labels_array, len(predictions_array))
    if predictions_array.ndim != 1 or np.any(predictions_array < 0) or np.any(
        predictions_array >= len(CLASS_NAMES)
    ):
        raise ValueError("Predictions contain invalid Fashion-MNIST class IDs")
    class_ids = np.arange(len(CLASS_NAMES))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=class_ids,
        zero_division=0,
    )
    matrix = confusion_matrix(labels_array, predictions_array, labels=class_ids)
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "macro_f1": float(
            f1_score(
                labels_array,
                predictions_array,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": [
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in class_ids
        ],
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def reliability_bins(
    probabilities: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    bin_edges: Sequence[float],
) -> list[dict[str, float | int | None]]:
    probability_array = _numpy(probabilities, np.float64)
    label_array = _numpy(labels, np.int64)
    _validate_probabilities(probability_array, label_array)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Calibration bin edges must be strictly increasing")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("Calibration bins must span [0, 1]")

    confidence = probability_array.max(axis=1)
    predictions = probability_array.argmax(axis=1)
    correct = predictions == label_array
    assignments = np.searchsorted(edges, confidence, side="right") - 1
    assignments = np.clip(assignments, 0, len(edges) - 2)
    rows: list[dict[str, float | int | None]] = []
    for index in range(len(edges) - 1):
        mask = assignments == index
        count = int(mask.sum())
        rows.append(
            {
                "bin_index": index,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "sample_count": count,
                "average_confidence": float(confidence[mask].mean()) if count else None,
                "empirical_accuracy": float(correct[mask].mean()) if count else None,
            }
        )
    return rows


def _validate_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("Probabilities must have shape [samples, 10]")
    _validate_labels(labels, len(probabilities))
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(
        probabilities > 1
    ):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")


def probability_metrics(
    probabilities: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    bin_edges: Sequence[float],
) -> dict[str, Any]:
    probability_array = _numpy(probabilities, np.float64)
    label_array = _numpy(labels, np.int64)
    _validate_probabilities(probability_array, label_array)
    bins = reliability_bins(probability_array, label_array, bin_edges)
    sample_count = len(label_array)
    true_probabilities = probability_array[np.arange(sample_count), label_array]
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[label_array]
    confidence = probability_array.max(axis=1)
    predictions = probability_array.argmax(axis=1)
    accuracy = float((predictions == label_array).mean())
    mean_confidence = float(confidence.mean())
    nonempty = [row for row in bins if row["sample_count"]]
    calibration_errors = [
        abs(float(row["empirical_accuracy"]) - float(row["average_confidence"]))
        for row in nonempty
    ]
    expected_error = sum(
        int(row["sample_count"]) / sample_count * error
        for row, error in zip(nonempty, calibration_errors)
    )
    return {
        "aggregate": {
            "negative_log_likelihood": float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()),
            "brier_score": float(np.square(probability_array - one_hot).sum(axis=1).mean()),
            "expected_calibration_error": float(expected_error),
            "maximum_calibration_error": float(max(calibration_errors, default=0.0)),
            "mean_confidence": mean_confidence,
            "accuracy_confidence_gap": accuracy - mean_confidence,
        },
        "reliability_bins": bins,
    }


def conformal_metrics(
    prediction_sets: Sequence[Sequence[int]],
    labels: np.ndarray | torch.Tensor,
    *,
    target_coverage: float,
) -> dict[str, Any]:
    label_array = _numpy(labels, np.int64)
    _validate_labels(label_array, len(prediction_sets))
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("Target coverage must lie in [0, 1]")
    normalized_sets: list[set[int]] = []
    for prediction_set in prediction_sets:
        values = [int(class_id) for class_id in prediction_set]
        if len(values) != len(set(values)):
            raise ValueError("Prediction sets cannot contain duplicate class IDs")
        if any(class_id < 0 or class_id >= len(CLASS_NAMES) for class_id in values):
            raise ValueError("Prediction sets contain invalid class IDs")
        normalized_sets.append(set(values))

    covered = np.asarray(
        [int(label) in prediction_set for label, prediction_set in zip(label_array, normalized_sets)],
        dtype=bool,
    )
    sizes = np.asarray([len(prediction_set) for prediction_set in normalized_sets], dtype=np.int64)
    empirical_coverage = float(covered.mean())
    per_class: list[dict[str, float | int | str | None]] = []
    present_coverages: list[float] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = label_array == class_id
        count = int(mask.sum())
        class_coverage = float(covered[mask].mean()) if count else None
        average_size = float(sizes[mask].mean()) if count else None
        if class_coverage is not None:
            present_coverages.append(class_coverage)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "sample_count": count,
                "coverage": class_coverage,
                "average_set_size": average_size,
            }
        )
    return {
        "aggregate": {
            "empirical_coverage": empirical_coverage,
            "coverage_gap": empirical_coverage - target_coverage,
            "average_set_size": float(sizes.mean()),
            "median_set_size": float(np.median(sizes)),
            "singleton_set_rate": float((sizes == 1).mean()),
            "empty_set_rate": float((sizes == 0).mean()),
            "multiclass_set_rate": float((sizes > 1).mean()),
            "worst_class_coverage": min(present_coverages),
        },
        "per_class": per_class,
    }


def selective_prediction_metrics(
    labels: np.ndarray | torch.Tensor,
    decisions: Sequence[AbstentionDecision],
) -> dict[str, float | int | None]:
    label_array = _numpy(labels, np.int64)
    _validate_labels(label_array, len(decisions))
    accepted_mask = np.asarray([decision.accepted for decision in decisions], dtype=bool)
    accepted_count = int(accepted_mask.sum())
    abstained_count = len(decisions) - accepted_count
    if accepted_count:
        accepted_predictions = np.asarray(
            [
                int(decision.predicted_class)
                for decision in decisions
                if decision.accepted and decision.predicted_class is not None
            ],
            dtype=np.int64,
        )
        if len(accepted_predictions) != accepted_count:
            raise ValueError("Every accepted decision must provide a predicted class")
        if np.any(accepted_predictions < 0) or np.any(accepted_predictions >= len(CLASS_NAMES)):
            raise ValueError("Accepted decisions contain invalid Fashion-MNIST class IDs")
        selective_accuracy: float | None = float(
            (accepted_predictions == label_array[accepted_mask]).mean()
        )
        selective_risk: float | None = 1.0 - selective_accuracy
    else:
        selective_accuracy = None
        selective_risk = None
    return {
        "acceptance_rate": accepted_count / len(decisions),
        "abstention_rate": abstained_count / len(decisions),
        "selective_accuracy": selective_accuracy,
        "selective_risk": selective_risk,
        "number_accepted": accepted_count,
        "number_abstained": abstained_count,
    }


def risk_coverage_point(
    labels: np.ndarray | torch.Tensor,
    decisions: Sequence[AbstentionDecision],
    *,
    policy_name: str,
    threshold: float | None,
) -> dict[str, float | int | str | None]:
    metrics = selective_prediction_metrics(labels, decisions)
    return {
        "policy_name": policy_name,
        "threshold": threshold,
        "coverage": metrics["acceptance_rate"],
        "selective_risk": metrics["selective_risk"],
        "number_accepted": metrics["number_accepted"],
    }


def risk_coverage_data(
    labels: np.ndarray | torch.Tensor,
    decision_groups: Iterable[tuple[str, float | None, Sequence[AbstentionDecision]]],
) -> list[dict[str, float | int | str | None]]:
    return [
        risk_coverage_point(
            labels,
            decisions,
            policy_name=policy_name,
            threshold=threshold,
        )
        for policy_name, threshold, decisions in decision_groups
    ]
