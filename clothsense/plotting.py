from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from .data import CLASS_NAMES
from .shifts import GeneratedScenario
from .training import TrainingHistory


def plot_training_curves(history: TrainingHistory, destination: str | Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history.epoch, history.train_loss, label="Training")
    axes[0].plot(history.epoch, history.validation_loss, label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.train_accuracy, label="Training")
    axes[1].plot(history.epoch, history.validation_accuracy, label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.0))
    axes[1].legend()
    figure.suptitle("Fashion-MNIST training history")
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def plot_confusion_matrix(metrics: dict[str, Any], destination: str | Path) -> Path:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title="Clean-test confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    axis.tick_params(axis="x", rotation=45)
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def plot_shift_verification_grid(
    scenarios: list[GeneratedScenario],
    destination: str | Path,
) -> Path:
    family_order = (
        "gaussian_noise",
        "rotation",
        "gaussian_blur",
        "brightness_contrast",
        "class_imbalance",
    )
    figure, axes = plt.subplots(len(family_order), 3, figsize=(9, 14))
    for row, family in enumerate(family_order):
        family_scenarios = [item for item in scenarios if item.scenario.shift_name == family]
        for column in range(3):
            axis = axes[row, column]
            if column >= len(family_scenarios):
                axis.axis("off")
                continue
            generated = family_scenarios[column]
            image, label = generated.dataset.raw_item(0)
            axis.imshow(image.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(
                f"{family.replace('_', ' ').title()}\n"
                f"{generated.scenario.severity_name}\n"
                f"n={generated.scenario.sample_count}, label={CLASS_NAMES[label]}",
                fontsize=9,
            )
            axis.axis("off")
    figure.suptitle("Configured Fashion-MNIST test shifts", y=0.99)
    figure.subplots_adjust(top=0.93, bottom=0.03, hspace=0.42, wspace=0.08)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output
