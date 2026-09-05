from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig


STRONGEST_SCENARIOS = (
    ("gaussian_noise", "strong"),
    ("rotation", "strong"),
    ("gaussian_blur", "strong"),
    ("brightness_contrast", "strong_brightness"),
    ("class_imbalance", "strong"),
)


def _result_row(
    aggregate: pd.DataFrame,
    *,
    shift_name: str,
    severity_name: str,
    method_name: str,
    metric_name: str,
    alpha: float | None = None,
    policy: str | None = None,
    threshold: float | None = None,
) -> dict[str, float | None]:
    selected = aggregate[
        (aggregate["shift_name"] == shift_name)
        & (aggregate["severity_name"] == severity_name)
        & (aggregate["method_name"] == method_name)
        & (aggregate["metric_name"] == metric_name)
    ]
    selected = selected[
        selected["abstention_policy"].isna()
        if policy is None
        else selected["abstention_policy"].eq(policy)
    ]
    if alpha is None:
        selected = selected[selected["alpha"].isna()]
    else:
        selected = selected[np.isclose(selected["alpha"], alpha)]
    if threshold is None:
        selected = selected[selected["abstention_threshold"].isna()]
    else:
        selected = selected[np.isclose(selected["abstention_threshold"], threshold)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one aggregate result for {shift_name}/{severity_name}/"
            f"{method_name}/{metric_name}, found {len(selected)}"
        )
    row = selected.iloc[0]
    return {
        "mean": float(row["mean"]),
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
    }


def load_dashboard_summary(config: ProjectConfig) -> dict[str, Any]:
    path = config.paths.results / "aggregate_metrics.csv"
    aggregate = pd.read_csv(path)
    alpha = config.uncertainty.upload_demo_alpha
    policy = config.uncertainty.upload_demo_abstention_policy
    threshold = config.uncertainty.upload_demo_abstention_threshold
    clean = {
        "accuracy": _result_row(
            aggregate,
            shift_name="clean",
            severity_name="clean",
            method_name="raw_softmax",
            metric_name="accuracy",
        ),
        "raw_ece": _result_row(
            aggregate,
            shift_name="clean",
            severity_name="clean",
            method_name="raw_softmax",
            metric_name="expected_calibration_error",
        ),
        "calibrated_ece": _result_row(
            aggregate,
            shift_name="clean",
            severity_name="clean",
            method_name="temperature_scaled",
            metric_name="expected_calibration_error",
        ),
        "standard_conformal_coverage": _result_row(
            aggregate,
            shift_name="clean",
            severity_name="clean",
            method_name="standard_conformal",
            metric_name="empirical_coverage",
            alpha=alpha,
        ),
    }
    shifts = []
    for shift_name, severity_name in STRONGEST_SCENARIOS:
        shifts.append(
            {
                "shift_name": shift_name,
                "severity_name": severity_name,
                "accuracy": _result_row(
                    aggregate,
                    shift_name=shift_name,
                    severity_name=severity_name,
                    method_name="raw_softmax",
                    metric_name="accuracy",
                ),
                "calibrated_ece": _result_row(
                    aggregate,
                    shift_name=shift_name,
                    severity_name=severity_name,
                    method_name="temperature_scaled",
                    metric_name="expected_calibration_error",
                ),
                "standard_conformal_coverage": _result_row(
                    aggregate,
                    shift_name=shift_name,
                    severity_name=severity_name,
                    method_name="standard_conformal",
                    metric_name="empirical_coverage",
                    alpha=alpha,
                ),
            }
        )
    method_name = "temperature_scaled" if policy == "confidence_threshold" else "standard_conformal"
    demo = {
        "model_seed": config.inference.model_seed,
        "alpha": alpha,
        "abstention_policy": policy,
        "abstention_threshold": threshold,
        "clean_acceptance_rate": _result_row(
            aggregate,
            shift_name="clean",
            severity_name="clean",
            method_name=method_name,
            metric_name="acceptance_rate",
            alpha=None if policy == "confidence_threshold" else alpha,
            policy=policy,
            threshold=threshold,
        ),
        "clean_selective_risk": _result_row(
            aggregate,
            shift_name="clean",
            severity_name="clean",
            method_name=method_name,
            metric_name="selective_risk",
            alpha=None if policy == "confidence_threshold" else alpha,
            policy=policy,
            threshold=threshold,
        ),
    }
    return {
        "source": str(path),
        "seed_count": len(config.seeds.experiments),
        "clean": clean,
        "strongest_shifts": shifts,
        "demo_configuration": demo,
    }


def list_research_charts(config: ProjectConfig) -> list[dict[str, str]]:
    pattern = re.compile(r"^(\d{2})_(.+)\.png$")
    charts = []
    for path in sorted(config.paths.plots.glob("[0-9][0-9]_*.png")):
        match = pattern.match(path.name)
        if match is None:
            continue
        charts.append(
            {
                "name": path.name,
                "title": match.group(2).replace("_", " ").title(),
                "url": f"/charts/{path.name}",
            }
        )
    return charts
