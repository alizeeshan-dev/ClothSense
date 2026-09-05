from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ProbabilityPredictions:
    probabilities: torch.Tensor
    predicted_classes: torch.Tensor
    confidence: torch.Tensor


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Logits must have shape [samples, classes]")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite")


def softmax_predictions(logits: torch.Tensor) -> ProbabilityPredictions:
    _validate_logits(logits)
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_classes = probabilities.max(dim=1)
    return ProbabilityPredictions(probabilities, predicted_classes, confidence)


def temperature_scaled_predictions(
    logits: torch.Tensor,
    temperature: float,
) -> ProbabilityPredictions:
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
        raise ValueError("Temperature must be finite and strictly positive")
    return softmax_predictions(logits / temperature)


def fit_temperature(
    calibration_logits: torch.Tensor,
    calibration_labels: torch.Tensor,
    *,
    max_iterations: int = 100,
) -> float:
    _validate_logits(calibration_logits)
    if calibration_labels.ndim != 1 or len(calibration_labels) != len(calibration_logits):
        raise ValueError("Calibration labels must align with calibration logits")
    if calibration_labels.dtype != torch.long:
        calibration_labels = calibration_labels.long()
    if calibration_labels.numel() == 0:
        raise ValueError("Temperature fitting requires calibration samples")
    if calibration_labels.min() < 0 or calibration_labels.max() >= calibration_logits.shape[1]:
        raise ValueError("Calibration labels contain invalid class IDs")

    logits = calibration_logits.detach().to(dtype=torch.float64, device="cpu")
    labels = calibration_labels.detach().to(dtype=torch.long, device="cpu")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iterations,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-7.0, 7.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-7.0, 7.0).exp().item())
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
        raise RuntimeError("Temperature optimization did not produce a positive finite value")
    return temperature

