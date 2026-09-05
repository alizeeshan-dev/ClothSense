from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from clothsense.config import load_config
from clothsense.experiments import build_run_specs
from clothsense.storage import foreign_key_violations


def main() -> None:
    config = load_config()
    database = config.paths.database
    expected_scenarios = 15 * len(config.seeds.experiments)
    expected_runs = expected_scenarios * len(build_run_specs(config))
    with sqlite3.connect(database) as connection:
        scenario_count = connection.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]
        completed_count = connection.execute(
            "SELECT COUNT(*) FROM runs WHERE status = 'completed'"
        ).fetchone()[0]
        failed_count = connection.execute(
            "SELECT COUNT(*) FROM runs WHERE status != 'completed'"
        ).fetchone()[0]
        duplicate_count = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT model_id, scenario_id, method_name,
                       COALESCE(alpha, -1), COALESCE(abstention_policy, ''),
                       COALESCE(abstention_threshold, -1), COUNT(*) AS copies
                FROM runs WHERE status = 'completed'
                GROUP BY 1, 2, 3, 4, 5, 6 HAVING copies > 1
            )
            """
        ).fetchone()[0]
        scenario_rows = connection.execute(
            "SELECT seed, shift_name, severity_name, sample_count FROM scenarios"
        ).fetchall()
        database_accuracy = connection.execute(
            """
            SELECT AVG(me.metric_value)
            FROM metrics me
            JOIN runs ru ON ru.run_id = me.run_id
            JOIN models mo ON mo.model_id = ru.model_id
            JOIN scenarios sc ON sc.scenario_id = ru.scenario_id
            WHERE ru.status = 'completed' AND sc.shift_name = 'clean'
              AND ru.method_name = 'raw_softmax' AND me.metric_name = 'accuracy'
            """
        ).fetchone()[0]

    assert scenario_count == expected_scenarios
    assert completed_count == expected_runs
    assert failed_count == 0
    assert duplicate_count == 0
    assert foreign_key_violations(database) == []
    scenario_frame = pd.DataFrame(
        scenario_rows, columns=["seed", "shift_name", "severity_name", "sample_count"]
    )
    assert (scenario_frame.groupby("seed").size() == 15).all()
    counts = scenario_frame.groupby(["shift_name", "severity_name"])["sample_count"].nunique()
    assert (counts == 1).all()

    required = {
        "run_metrics.csv": {"seed", "shift_name", "severity_name", "method_name", "alpha", "abstention_policy", "abstention_threshold", "metric_name", "metric_value"},
        "aggregate_metrics.csv": {"mean", "std", "ci_lower", "ci_upper", "seed_count", "complete_group"},
        "per_class_metrics.csv": {"seed", "class", "metric_name", "metric_value"},
        "reliability_bins.csv": {"bin_index", "sample_count", "average_confidence", "empirical_accuracy"},
        "risk_coverage.csv": {"seed", "abstention_policy", "coverage", "selective_risk"},
    }
    loaded: dict[str, pd.DataFrame] = {}
    for filename, columns in required.items():
        frame = pd.read_csv(config.paths.results / filename)
        assert columns <= set(frame.columns)
        loaded[filename] = frame

    run_metrics = loaded["run_metrics.csv"]
    csv_accuracy = run_metrics[
        (run_metrics["shift_name"] == "clean")
        & (run_metrics["method_name"] == "raw_softmax")
        & (run_metrics["metric_name"] == "accuracy")
    ]["metric_value"].mean()
    assert np.isclose(csv_accuracy, database_accuracy)
    aggregate = loaded["aggregate_metrics.csv"]
    row = aggregate[
        (aggregate["shift_name"] == "clean")
        & (aggregate["method_name"] == "raw_softmax")
        & (aggregate["metric_name"] == "accuracy")
    ].iloc[0]
    assert bool(row["complete_group"])
    assert int(row["seed_count"]) == len(config.seeds.experiments)
    assert np.isclose(row["mean"], csv_accuracy)
    print(
        f"verified scenarios={scenario_count} completed_runs={completed_count} "
        f"csv_accuracy={csv_accuracy:.6f}"
    )


if __name__ == "__main__":
    main()
