from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SHIFT_ORDER = {
    "gaussian_noise": ["mild", "medium", "strong"],
    "rotation": ["mild", "medium", "strong"],
    "gaussian_blur": ["mild", "medium", "strong"],
    "brightness_contrast": ["mild_brightness", "strong_brightness", "increased_contrast"],
    "class_imbalance": ["moderate", "strong"],
}
STRONGEST = {
    "gaussian_noise": "strong",
    "rotation": "strong",
    "gaussian_blur": "strong",
    "brightness_contrast": "strong_brightness",
    "class_imbalance": "strong",
}


def _read(results: Path, name: str) -> pd.DataFrame:
    path = results / name
    if not path.is_file():
        raise FileNotFoundError(f"Required persisted result is missing: {path}")
    return pd.read_csv(path)


def _save(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _complete_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["complete_group"].dtype == bool:
        return frame[frame["complete_group"]].copy()
    return frame[frame["complete_group"].astype(str).str.lower() == "true"].copy()


def plot_training_loss(run_metrics: pd.DataFrame, destination: Path) -> None:
    selected = run_metrics[
        (run_metrics["shift_name"] == "clean")
        & (run_metrics["method_name"] == "raw_softmax")
        & run_metrics["metric_name"].str.match(r"training_(train|validation)_loss_epoch_\d+")
    ].copy()
    if selected.empty:
        raise ValueError("Training-history metrics are absent from run_metrics.csv")
    extracted = selected["metric_name"].str.extract(
        r"training_(train|validation)_loss_epoch_(\d+)"
    )
    selected["split"] = extracted[0]
    selected["epoch"] = extracted[1].astype(int)
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    for split, color in (("train", "#2563eb"), ("validation", "#dc2626")):
        values = selected[selected["split"] == split]
        grouped = values.groupby("epoch")["metric_value"].agg(["mean", "std", "count"])
        axis.plot(grouped.index, grouped["mean"], marker="o", markersize=3, label=split.title(), color=color)
        spread = grouped["std"].fillna(0.0)
        axis.fill_between(grouped.index, grouped["mean"] - spread, grouped["mean"] + spread, color=color, alpha=0.12)
    axis.set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, destination)


def plot_clean_confusion(run_metrics: pd.DataFrame, destination: Path) -> None:
    selected = run_metrics[
        (run_metrics["shift_name"] == "clean")
        & (run_metrics["method_name"] == "raw_softmax")
        & run_metrics["metric_name"].str.startswith("confusion_true_")
    ].copy()
    matrix = np.zeros((10, 10), dtype=float)
    for true_id in range(10):
        for predicted_id in range(10):
            name = f"confusion_true_{true_id}_pred_{predicted_id}"
            matrix[true_id, predicted_id] = selected.loc[selected["metric_name"] == name, "metric_value"].mean()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Mean sample count across seeds")
    axis.set(title="Clean-data confusion matrix", xlabel="Predicted class", ylabel="True class")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    _save(figure, destination)


def plot_reliability(reliability: pd.DataFrame, destination: Path) -> None:
    selected = reliability[
        (reliability["shift_name"] == "clean")
        & reliability["method_name"].isin(["raw_softmax", "temperature_scaled"])
        & (reliability["sample_count"] > 0)
    ]
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    labels = {"raw_softmax": "Raw softmax", "temperature_scaled": "Temperature scaled"}
    for method, values in selected.groupby("method_name"):
        grouped = values.groupby("bin_index").agg(
            confidence=("average_confidence", "mean"), accuracy=("empirical_accuracy", "mean")
        )
        axis.plot(grouped["confidence"], grouped["accuracy"], marker="o", label=labels[method])
    axis.set(xlim=(0, 1), ylim=(0, 1), title="Clean reliability before and after scaling", xlabel="Mean confidence", ylabel="Empirical accuracy")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, destination)


def _severity_panels(
    aggregate: pd.DataFrame,
    *,
    metric_name: str,
    methods: list[str],
    title: str,
    ylabel: str,
    destination: Path,
    alphas: list[float] | None = None,
    nominal: bool = False,
) -> None:
    selected = aggregate[
        (aggregate["metric_name"] == metric_name)
        & aggregate["method_name"].isin(methods)
        & aggregate["abstention_policy"].isna()
    ].copy()
    if alphas is not None:
        selected = selected[selected["alpha"].isin(alphas)]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharey=True)
    axes_flat = axes.flat
    for axis, (shift, severities) in zip(axes_flat, SHIFT_ORDER.items()):
        values = selected[selected["shift_name"] == shift]
        labels = []
        for method in methods:
            method_values = values[values["method_name"] == method]
            alpha_values = sorted(method_values["alpha"].dropna().unique()) if alphas else [None]
            for alpha in alpha_values:
                subset = method_values if alpha is None else method_values[np.isclose(method_values["alpha"], alpha)]
                subset = subset.set_index("severity_name").reindex(severities)
                x = np.arange(len(severities))
                y = subset["mean"].to_numpy(dtype=float)
                lower = subset["ci_lower"].to_numpy(dtype=float)
                upper = subset["ci_upper"].to_numpy(dtype=float)
                error = np.vstack([y - lower, upper - y])
                label = method.replace("_", " ") + (f" α={alpha:g}" if alpha is not None else "")
                axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=label)
                labels.append(label)
                if nominal and alpha is not None:
                    axis.axhline(1.0 - alpha, linestyle="--", alpha=0.35)
        axis.set_title(shift.replace("_", " ").title())
        axis.set_xticks(np.arange(len(severities)), [value.replace("_", "\n") for value in severities])
        axis.grid(alpha=0.25)
    axes_flat[-1].axis("off")
    axes[1, 0].set_ylabel(ylabel)
    axes[0, 0].set_ylabel(ylabel)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.98, 0.08))
    figure.suptitle(title)
    _save(figure, destination)


def plot_worst_class(aggregate: pd.DataFrame, destination: Path) -> None:
    selected = aggregate[
        (aggregate["metric_name"] == "worst_class_coverage")
        & aggregate["method_name"].isin(["standard_conformal", "class_conditional_conformal"])
        & aggregate["abstention_policy"].isna()
        & np.isclose(aggregate["alpha"], 0.10)
    ]
    scenarios = [("clean", "clean"), *STRONGEST.items()]
    labels = ["Clean", "Noise", "Rotation", "Blur", "Brightness", "Imbalance"]
    x = np.arange(len(scenarios))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for offset, method in ((-width / 2, "standard_conformal"), (width / 2, "class_conditional_conformal")):
        values = []
        errors = []
        for shift, severity in scenarios:
            row = selected[(selected["shift_name"] == shift) & (selected["severity_name"] == severity) & (selected["method_name"] == method)]
            values.append(float(row["mean"].iloc[0]))
            errors.append(float(row["ci_upper"].iloc[0] - row["mean"].iloc[0]))
        axis.bar(x + offset, values, width, yerr=errors, capsize=3, label=method.replace("_", " "))
    axis.axhline(0.90, color="black", linestyle="--", label="Nominal 90%")
    axis.set(title="Worst-class coverage: standard vs class-conditional", xlabel="Scenario", ylabel="Worst-class coverage", xticks=x, xticklabels=labels, ylim=(0, 1.05))
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(figure, destination)


def plot_risk_coverage(risk: pd.DataFrame, destination: Path) -> None:
    selected = risk[risk["shift_name"] == "clean"].copy()
    selected = selected[
        (selected["abstention_policy"] == "confidence_threshold")
        | ((selected["abstention_policy"] == "hybrid") & np.isclose(selected["alpha"], 0.10))
        | (selected["abstention_policy"] == "singleton_conformal")
    ]
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    for policy, values in selected.groupby("abstention_policy"):
        grouped = values.groupby(["alpha", "abstention_threshold"], dropna=False).agg(
            coverage=("coverage", "mean"), risk=("selective_risk", "mean")
        ).sort_values("coverage")
        axis.plot(grouped["coverage"], grouped["risk"], marker="o", label=policy.replace("_", " "))
    axis.set(title="Clean-data risk–coverage trade-off", xlabel="Coverage / acceptance rate", ylabel="Selective risk", xlim=(0, 1))
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, destination)


def plot_clean_strongest(aggregate: pd.DataFrame, destination: Path) -> None:
    scenarios = [("clean", "clean"), *STRONGEST.items()]
    labels = ["Clean", "Noise", "Rotation", "Blur", "Brightness", "Imbalance"]
    accuracy = aggregate[
        (aggregate["method_name"] == "raw_softmax")
        & (aggregate["metric_name"] == "accuracy")
        & aggregate["abstention_policy"].isna()
    ]
    ece = aggregate[
        (aggregate["method_name"] == "temperature_scaled")
        & (aggregate["metric_name"] == "expected_calibration_error")
        & aggregate["abstention_policy"].isna()
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, frame, ylabel, heading in (
        (axes[0], accuracy, "Accuracy", "Classification"),
        (axes[1], ece, "ECE", "Calibration"),
    ):
        values, errors = [], []
        for shift, severity in scenarios:
            row = frame[(frame["shift_name"] == shift) & (frame["severity_name"] == severity)]
            values.append(float(row["mean"].iloc[0]))
            errors.append(float(row["ci_upper"].iloc[0] - row["mean"].iloc[0]))
        axis.bar(np.arange(len(labels)), values, yerr=errors, capsize=3)
        axis.set(title=heading, ylabel=ylabel, xticks=np.arange(len(labels)), xticklabels=labels)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Clean vs strongest configured shifts")
    _save(figure, destination)


def generate_research_plots(results_directory: str | Path, plot_directory: str | Path) -> list[Path]:
    results = Path(results_directory)
    plots = Path(plot_directory)
    run_metrics = _read(results, "run_metrics.csv")
    aggregate = _complete_aggregate(_read(results, "aggregate_metrics.csv"))
    reliability = _read(results, "reliability_bins.csv")
    risk = _read(results, "risk_coverage.csv")
    if aggregate.empty:
        raise ValueError("No complete three-seed aggregate groups are available")

    paths = [
        plots / "01_training_validation_loss.png",
        plots / "02_clean_confusion_matrix.png",
        plots / "03_reliability_raw_vs_temperature.png",
        plots / "04_accuracy_vs_shift_severity.png",
        plots / "05_ece_vs_shift_severity.png",
        plots / "06_conformal_coverage_vs_shift_severity.png",
        plots / "07_conformal_set_size_vs_shift_severity.png",
        plots / "08_worst_class_coverage.png",
        plots / "09_risk_coverage.png",
        plots / "10_clean_vs_strongest_shift.png",
    ]
    plot_training_loss(run_metrics, paths[0])
    plot_clean_confusion(run_metrics, paths[1])
    plot_reliability(reliability, paths[2])
    _severity_panels(aggregate, metric_name="accuracy", methods=["raw_softmax"], title="Accuracy vs shift severity", ylabel="Accuracy", destination=paths[3])
    _severity_panels(aggregate, metric_name="expected_calibration_error", methods=["raw_softmax", "temperature_scaled"], title="ECE vs shift severity", ylabel="ECE", destination=paths[4])
    _severity_panels(aggregate, metric_name="empirical_coverage", methods=["standard_conformal"], title="Conformal coverage vs shift severity", ylabel="Empirical coverage", destination=paths[5], alphas=[0.05, 0.10, 0.20], nominal=True)
    _severity_panels(aggregate, metric_name="average_set_size", methods=["standard_conformal"], title="Average conformal-set size vs shift severity", ylabel="Average set size", destination=paths[6], alphas=[0.05, 0.10, 0.20])
    plot_worst_class(aggregate, paths[7])
    plot_risk_coverage(risk, paths[8])
    plot_clean_strongest(aggregate, paths[9])
    return paths
