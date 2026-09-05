from __future__ import annotations

import math
import sqlite3

import pandas as pd

from clothsense.config import load_config
from clothsense.experiments import build_run_specs
from clothsense.reporting import aggregate_run_metrics, export_results, student_t_summary
from clothsense.storage import (
    complete_run,
    fail_run,
    foreign_key_violations,
    initialize_database,
    start_run,
    upsert_scenario,
)


def _model_id(database) -> int:
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(
            """
            INSERT INTO models (
                seed, config_hash, checkpoint_path, best_epoch,
                validation_loss, validation_accuracy, created_at
            ) VALUES (17, 'hash', 'model.pt', 2, 0.2, 0.9, 'now')
            """
        )
        return int(cursor.lastrowid)


def test_experiment_matrix_has_only_meaningful_combinations():
    config = load_config()
    specs = build_run_specs(config)
    assert len(specs) == 35
    assert len(set(specs)) == len(specs)
    raw = [spec for spec in specs if spec.method_name == "raw_softmax"]
    assert raw == [raw[0]]
    assert raw[0].alpha is None and raw[0].abstention_policy is None
    confidence = [spec for spec in specs if spec.abstention_policy == "confidence_threshold"]
    assert len(confidence) == len(config.uncertainty.abstention_thresholds)
    assert all(spec.alpha is None for spec in confidence)


def test_storage_tracks_state_and_skips_duplicate_completed_runs(tmp_path):
    database = initialize_database(tmp_path / "experiment.sqlite")
    model_id = _model_id(database)
    scenario_id = upsert_scenario(
        database,
        shift_name="clean",
        severity_name="clean",
        parameters={},
        seed=17,
        sample_count=10,
    )
    run_id, execute = start_run(
        database,
        model_id=model_id,
        scenario_id=scenario_id,
        method_name="raw_softmax",
    )
    assert execute
    complete_run(database, run_id, [("accuracy", None, 0.9)])
    duplicate_id, execute = start_run(
        database,
        model_id=model_id,
        scenario_id=scenario_id,
        method_name="raw_softmax",
    )
    assert duplicate_id == run_id
    assert not execute
    assert foreign_key_violations(database) == []

    failed_id, execute = start_run(
        database,
        model_id=model_id,
        scenario_id=scenario_id,
        method_name="temperature_scaled",
    )
    assert execute
    fail_run(database, failed_id, "controlled failure")
    with sqlite3.connect(database) as connection:
        statuses = dict(connection.execute("SELECT method_name, status FROM runs"))
    assert statuses == {"raw_softmax": "completed", "temperature_scaled": "failed"}


def test_student_t_ci_and_incomplete_aggregation():
    summary = student_t_summary([1.0, 2.0, 3.0])
    expected_half_width = 4.302652729911275 / math.sqrt(3)
    assert summary["mean"] == 2.0
    assert summary["std"] == 1.0
    assert math.isclose(summary["ci_lower"], 2.0 - expected_half_width)
    assert math.isclose(summary["ci_upper"], 2.0 + expected_half_width)

    base = {
        "shift_name": "clean",
        "severity_name": "clean",
        "method_name": "raw_softmax",
        "alpha": None,
        "abstention_policy": None,
        "abstention_threshold": None,
        "metric_name": "accuracy",
    }
    frame = pd.DataFrame([{**base, "seed": seed, "metric_value": value} for seed, value in [(17, 0.8), (42, 0.9)]])
    aggregated = aggregate_run_metrics(frame, expected_seeds=[17, 42, 73])
    assert len(aggregated) == 1
    assert not bool(aggregated.iloc[0]["complete_group"])
    assert pd.isna(aggregated.iloc[0]["mean"])


def test_csv_export_uses_only_completed_runs(tmp_path):
    database = initialize_database(tmp_path / "export.sqlite")
    model_id = _model_id(database)
    scenario_id = upsert_scenario(
        database,
        shift_name="clean",
        severity_name="clean",
        parameters={},
        seed=17,
        sample_count=10,
    )
    completed_id, _ = start_run(
        database,
        model_id=model_id,
        scenario_id=scenario_id,
        method_name="raw_softmax",
    )
    complete_run(database, completed_id, [("accuracy", None, 0.9)])
    failed_id, _ = start_run(
        database,
        model_id=model_id,
        scenario_id=scenario_id,
        method_name="temperature_scaled",
    )
    fail_run(database, failed_id, "expected")
    paths = export_results(database, tmp_path / "csv", expected_seeds=[17])
    exported = pd.read_csv(paths.run_metrics)
    assert exported["method_name"].tolist() == ["raw_softmax"]
    aggregate = pd.read_csv(paths.aggregate_metrics)
    assert bool(aggregate.iloc[0]["complete_group"])
