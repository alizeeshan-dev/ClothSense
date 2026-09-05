from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t


IDENTIFIERS = [
    "seed",
    "shift_name",
    "severity_name",
    "scenario_parameters",
    "method_name",
    "alpha",
    "abstention_policy",
    "abstention_threshold",
]


def _fill_optional_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["alpha"] = result["alpha"].fillna(-1.0)
    result["abstention_policy"] = result["abstention_policy"].fillna("")
    result["abstention_threshold"] = result["abstention_threshold"].fillna(-1.0)
    return result


def _restore_optional_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["alpha"] = result["alpha"].replace(-1.0, np.nan)
    result["abstention_policy"] = result["abstention_policy"].replace("", np.nan)
    result["abstention_threshold"] = result["abstention_threshold"].replace(-1.0, np.nan)
    return result


@dataclass(frozen=True)
class ExportedResults:
    run_metrics: Path
    aggregate_metrics: Path
    per_class_metrics: Path
    reliability_bins: Path
    risk_coverage: Path


def student_t_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = int(len(array))
    if count == 0:
        return {"count": 0, "mean": None, "std": None, "ci_lower": None, "ci_upper": None}
    mean = float(array.mean())
    if count == 1:
        return {"count": 1, "mean": mean, "std": None, "ci_lower": None, "ci_upper": None}
    standard_deviation = float(array.std(ddof=1))
    half_width = float(t.ppf(0.975, df=count - 1) * standard_deviation / np.sqrt(count))
    return {
        "count": count,
        "mean": mean,
        "std": standard_deviation,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }


def _completed_metrics(database_path: str | Path) -> pd.DataFrame:
    query = """
        SELECT
            mo.seed,
            sc.shift_name,
            sc.severity_name,
            sc.parameters AS scenario_parameters,
            sc.sample_count,
            ru.method_name,
            ru.alpha,
            ru.abstention_policy,
            ru.abstention_threshold,
            me.metric_name,
            me.class_name,
            me.metric_value
        FROM metrics AS me
        JOIN runs AS ru ON ru.run_id = me.run_id
        JOIN models AS mo ON mo.model_id = ru.model_id
        JOIN scenarios AS sc ON sc.scenario_id = ru.scenario_id
        WHERE ru.status = 'completed'
    """
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(query, connection)


def aggregate_run_metrics(
    run_metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    columns = [
        "shift_name",
        "severity_name",
        "method_name",
        "alpha",
        "abstention_policy",
        "abstention_threshold",
        "metric_name",
        "seed_count",
        "expected_seed_count",
        "complete_group",
        "mean",
        "std",
        "ci_lower",
        "ci_upper",
    ]
    if run_metrics.empty:
        return pd.DataFrame(columns=columns)
    eligible = run_metrics[
        ~run_metrics["metric_name"].str.startswith(("training_", "confusion_"))
    ].copy()
    group_columns = [
        "shift_name",
        "severity_name",
        "method_name",
        "alpha",
        "abstention_policy",
        "abstention_threshold",
        "metric_name",
    ]
    expected = len(tuple(expected_seeds))
    rows: list[dict] = []
    for keys, group in eligible.groupby(group_columns, dropna=False, sort=True):
        summary = student_t_summary(group["metric_value"].tolist())
        seed_count = int(group["seed"].nunique())
        complete = seed_count == expected and summary["count"] == expected
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "seed_count": seed_count,
                "expected_seed_count": expected,
                "complete_group": complete,
                "mean": summary["mean"] if complete else None,
                "std": summary["std"] if complete else None,
                "ci_lower": summary["ci_lower"] if complete else None,
                "ci_upper": summary["ci_upper"] if complete else None,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paired_clean_shift_differences(
    run_metrics: pd.DataFrame,
    *,
    metric_name: str,
    method_name: str,
) -> pd.DataFrame:
    selected = run_metrics[
        (run_metrics["metric_name"] == metric_name)
        & (run_metrics["method_name"] == method_name)
        & run_metrics["abstention_policy"].isna()
    ]
    clean = selected[
        (selected["shift_name"] == "clean") & (selected["severity_name"] == "clean")
    ][["seed", "metric_value"]].rename(columns={"metric_value": "clean_value"})
    shifted = selected[selected["shift_name"] != "clean"].copy()
    paired = shifted.merge(clean, on="seed", how="inner", validate="many_to_one")
    paired["difference_from_clean"] = paired["metric_value"] - paired["clean_value"]
    return paired


def export_results(
    database_path: str | Path,
    output_directory: str | Path,
    *,
    expected_seeds: Sequence[int],
) -> ExportedResults:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    all_metrics = _completed_metrics(database_path)

    run_metrics = all_metrics[all_metrics["class_name"].isna()].copy()
    run_metrics = run_metrics[IDENTIFIERS + ["metric_name", "metric_value"]]
    run_metrics = run_metrics.sort_values(IDENTIFIERS + ["metric_name"], na_position="last")

    per_class = all_metrics[
        all_metrics["class_name"].notna()
        & ~all_metrics["class_name"].str.startswith("bin:", na=False)
    ].copy()
    per_class = per_class[
        IDENTIFIERS + ["class_name", "metric_name", "metric_value"]
    ].rename(columns={"class_name": "class"})
    per_class = per_class.sort_values(IDENTIFIERS + ["class", "metric_name"], na_position="last")

    reliability_source = all_metrics[
        all_metrics["class_name"].str.startswith("bin:", na=False)
    ].copy()
    reliability_columns = IDENTIFIERS + [
        "bin_index",
        "lower_bound",
        "upper_bound",
        "sample_count",
        "average_confidence",
        "empirical_accuracy",
    ]
    if reliability_source.empty:
        reliability = pd.DataFrame(columns=reliability_columns)
    else:
        reliability_source["bin_index"] = reliability_source["class_name"].str.split(":").str[1].astype(int)
        reliability_source = _fill_optional_identifiers(reliability_source)
        reliability = (
            reliability_source.pivot_table(
                index=IDENTIFIERS + ["bin_index"],
                columns="metric_name",
                values="metric_value",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        reliability = _restore_optional_identifiers(reliability)
        reliability = reliability.reindex(columns=reliability_columns)
        reliability = reliability.dropna(subset=["seed"])

    risk_columns = IDENTIFIERS + [
        "coverage",
        "selective_risk",
        "number_accepted",
        "number_abstained",
    ]
    risk_source = run_metrics[run_metrics["abstention_policy"].notna()]
    if risk_source.empty:
        risk_coverage = pd.DataFrame(columns=risk_columns)
    else:
        risk_source = _fill_optional_identifiers(risk_source)
        risk_coverage = (
            risk_source[
                risk_source["metric_name"].isin(
                    ["acceptance_rate", "selective_risk", "number_accepted", "number_abstained"]
                )
            ]
            .pivot_table(
                index=IDENTIFIERS,
                columns="metric_name",
                values="metric_value",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(columns=None)
            .rename(columns={"acceptance_rate": "coverage"})
        )
        risk_coverage = _restore_optional_identifiers(risk_coverage)
        risk_coverage = risk_coverage.reindex(columns=risk_columns).dropna(subset=["seed"])

    aggregate = aggregate_run_metrics(run_metrics, expected_seeds=expected_seeds)
    paths = ExportedResults(
        run_metrics=destination / "run_metrics.csv",
        aggregate_metrics=destination / "aggregate_metrics.csv",
        per_class_metrics=destination / "per_class_metrics.csv",
        reliability_bins=destination / "reliability_bins.csv",
        risk_coverage=destination / "risk_coverage.csv",
    )
    run_metrics.to_csv(paths.run_metrics, index=False)
    aggregate.to_csv(paths.aggregate_metrics, index=False)
    per_class.to_csv(paths.per_class_metrics, index=False)
    reliability.to_csv(paths.reliability_bins, index=False)
    risk_coverage.to_csv(paths.risk_coverage, index=False)
    return paths
