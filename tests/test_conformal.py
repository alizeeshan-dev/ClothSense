from __future__ import annotations

import pytest
import torch

from clothsense.conformal import (
    class_conditional_prediction_sets,
    corrected_conformal_quantile,
    fit_class_conditional_thresholds,
    fit_standard_thresholds,
    nonconformity_scores,
    prediction_sets,
)


def test_standard_scores_corrected_quantile_and_sets(config) -> None:
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]], dtype=torch.float64
    )
    labels = torch.tensor([0, 1, 1, 1])
    scores = nonconformity_scores(probabilities, labels)
    assert torch.allclose(scores, torch.tensor([0.2, 0.4, 0.7, 0.1], dtype=torch.float64))
    assert corrected_conformal_quantile(scores, 0.5) == 0.4
    thresholds = fit_standard_thresholds(probabilities, labels, config.uncertainty.alpha_values)
    assert set(thresholds) == set(config.uncertainty.alpha_values)

    sets = prediction_sets(torch.tensor([[0.65, 0.35]]), threshold=0.4)
    assert sets == [(0,)]
    assert all(0 <= class_id < 2 for value in sets for class_id in value)


def test_finite_sample_rank_is_clipped_to_available_scores() -> None:
    scores = torch.tensor([0.1, 0.4, 0.2, 0.3])
    assert corrected_conformal_quantile(scores, 0.05) == pytest.approx(0.4)


def test_class_conditional_thresholds_and_candidate_indexing() -> None:
    probabilities = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.6, 0.2],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    thresholds = fit_class_conditional_thresholds(probabilities, labels, [0.5])[0.5]
    assert thresholds[0] == pytest.approx(0.2)
    assert thresholds[1] == pytest.approx(0.4)
    assert thresholds[2] == pytest.approx(0.2)

    sets = class_conditional_prediction_sets(
        torch.tensor([[0.1, 0.4, 0.5]]),
        {0: 0.2, 1: 0.5, 2: 0.8},
    )
    assert sets == [(2,)]
