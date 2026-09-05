from clothsense.abstention import (
    confidence_threshold_decision,
    hybrid_decision,
    singleton_set_decision,
)


def test_confidence_threshold_includes_boundary() -> None:
    accepted = confidence_threshold_decision(3, 0.8, 0.8)
    rejected = confidence_threshold_decision(3, 0.799, 0.8)
    assert accepted.accepted and accepted.predicted_class == 3
    assert not rejected.accepted and rejected.predicted_class is None


def test_singleton_empty_and_multiclass_behavior() -> None:
    assert singleton_set_decision([4]).accepted
    assert not singleton_set_decision([]).accepted
    assert singleton_set_decision([]).reason == "empty_conformal_set"
    assert not singleton_set_decision([2, 4]).accepted
    assert singleton_set_decision([2, 4]).reason == "multiclass_conformal_set"


def test_hybrid_requires_both_conditions_and_counts_are_derivable() -> None:
    decisions = [
        hybrid_decision(1, 0.9, [1], 0.8),
        hybrid_decision(1, 0.7, [1], 0.8),
        hybrid_decision(1, 0.9, [1, 2], 0.8),
    ]
    assert [decision.accepted for decision in decisions] == [True, False, False]
    assert sum(decision.accepted for decision in decisions) == 1

