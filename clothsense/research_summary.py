from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STRONGEST = {
    "gaussian_noise": "strong",
    "rotation": "strong",
    "gaussian_blur": "strong",
    "brightness_contrast": "strong_brightness",
    "class_imbalance": "strong",
}


def _mean_metric(
    frame: pd.DataFrame,
    *,
    shift_name: str,
    severity_name: str,
    method_name: str,
    metric_name: str,
    alpha: float | None = None,
) -> float:
    selected = frame[
        (frame["shift_name"] == shift_name)
        & (frame["severity_name"] == severity_name)
        & (frame["method_name"] == method_name)
        & (frame["metric_name"] == metric_name)
        & frame["abstention_policy"].isna()
    ]
    if alpha is not None:
        selected = selected[np.isclose(selected["alpha"], alpha)]
    if selected.empty:
        return float("nan")
    return float(selected["metric_value"].mean())


def generate_research_summary(
    results_directory: str | Path,
    destination: str | Path,
    *,
    demo_alpha: float,
    demo_policy: str,
    demo_threshold: float | None,
) -> Path:
    results = Path(results_directory)
    metrics = pd.read_csv(results / "run_metrics.csv")
    risk = pd.read_csv(results / "risk_coverage.csv")
    clean_accuracy = _mean_metric(
        metrics, shift_name="clean", severity_name="clean", method_name="raw_softmax", metric_name="accuracy"
    )
    clean_raw_ece = _mean_metric(
        metrics, shift_name="clean", severity_name="clean", method_name="raw_softmax", metric_name="expected_calibration_error"
    )
    clean_calibrated_ece = _mean_metric(
        metrics, shift_name="clean", severity_name="clean", method_name="temperature_scaled", metric_name="expected_calibration_error"
    )
    clean_raw_nll = _mean_metric(
        metrics, shift_name="clean", severity_name="clean", method_name="raw_softmax", metric_name="negative_log_likelihood"
    )
    clean_calibrated_nll = _mean_metric(
        metrics, shift_name="clean", severity_name="clean", method_name="temperature_scaled", metric_name="negative_log_likelihood"
    )

    lines = [
        "# ClothSense Phase 5 research summary",
        "",
        "## Clean baseline",
        "",
        f"- Accuracy: {clean_accuracy:.4f}",
        f"- ECE, raw → temperature-scaled: {clean_raw_ece:.4f} → {clean_calibrated_ece:.4f}",
        f"- NLL, raw → temperature-scaled: {clean_raw_nll:.4f} → {clean_calibrated_nll:.4f}",
        "",
        "## Strongest configured shifts",
        "",
        "| Shift | Accuracy | Δ accuracy | Calibrated ECE | Standard coverage | Average set size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for shift, severity in STRONGEST.items():
        accuracy = _mean_metric(metrics, shift_name=shift, severity_name=severity, method_name="raw_softmax", metric_name="accuracy")
        ece = _mean_metric(metrics, shift_name=shift, severity_name=severity, method_name="temperature_scaled", metric_name="expected_calibration_error")
        coverage = _mean_metric(metrics, shift_name=shift, severity_name=severity, method_name="standard_conformal", metric_name="empirical_coverage", alpha=demo_alpha)
        size = _mean_metric(metrics, shift_name=shift, severity_name=severity, method_name="standard_conformal", metric_name="average_set_size", alpha=demo_alpha)
        lines.append(f"| {shift}/{severity} | {accuracy:.4f} | {accuracy - clean_accuracy:+.4f} | {ece:.4f} | {coverage:.4f} | {size:.3f} |")

    clean_standard_worst = _mean_metric(metrics, shift_name="clean", severity_name="clean", method_name="standard_conformal", metric_name="worst_class_coverage", alpha=demo_alpha)
    clean_conditional_worst = _mean_metric(metrics, shift_name="clean", severity_name="clean", method_name="class_conditional_conformal", metric_name="worst_class_coverage", alpha=demo_alpha)
    lines.extend(
        [
            "",
            "## Conformal and abstention trade-offs",
            "",
            f"- At alpha={demo_alpha:g}, clean worst-class coverage is {clean_standard_worst:.4f} for standard conformal and {clean_conditional_worst:.4f} for class-conditional conformal.",
        ]
    )
    chosen = risk[
        (risk["abstention_policy"] == demo_policy)
        & ((risk["alpha"].isna()) if demo_policy == "confidence_threshold" else np.isclose(risk["alpha"], demo_alpha))
    ]
    if demo_threshold is None:
        chosen = chosen[chosen["abstention_threshold"].isna()]
    else:
        chosen = chosen[np.isclose(chosen["abstention_threshold"], demo_threshold)]
    clean_choice = chosen[chosen["shift_name"] == "clean"]
    lines.append(
        f"- Demo policy `{demo_policy}` has mean clean acceptance {clean_choice['coverage'].mean():.4f} and selective risk {clean_choice['selective_risk'].mean():.4f}."
    )
    lines.extend(
        [
            "",
            "## Selected upload-demo configuration",
            "",
            f"- Conformal alpha: {demo_alpha:g}",
            f"- Abstention policy: {demo_policy}",
            f"- Confidence threshold: {demo_threshold if demo_threshold is not None else 'not applicable'}",
            "- Selection balances coverage and selective risk while retaining a conformal singleton condition that is straightforward to explain.",
        ]
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
