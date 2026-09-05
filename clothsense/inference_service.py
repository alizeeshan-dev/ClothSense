from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .abstention import (
    AbstentionDecision,
    confidence_threshold_decision,
    hybrid_decision,
    singleton_set_decision,
)
from .config import ProjectConfig
from .data import CLASS_NAMES
from .reproducibility import select_device
from .training import configuration_hash, load_checkpoint, seed_artifact_paths
from .uncertainty_artifacts import file_sha256, load_uncertainty_artifact
from .upload_preprocessing import PreprocessedImage, preprocess_upload


DOMAIN_LIMITATION = (
    "Uploaded photographs can differ substantially from the Fashion-MNIST training data. "
    "Results are experimental; abstention or large conformal sets indicate uncertainty, "
    "not mathematically proven out-of-distribution detection."
)


REASON_MESSAGES = {
    "confidence_threshold_met": "Accepted because calibrated confidence passed the configured threshold.",
    "singleton_set": "Accepted because the conformal prediction set contains one class.",
    "hybrid_conditions_met": "Accepted because confidence and conformal singleton conditions passed.",
    "below_confidence_threshold": "Uncertain because calibrated confidence is below the configured threshold.",
    "empty_conformal_set": "Uncertain because the conformal prediction set is empty.",
    "multiclass_conformal_set": "Uncertain because the conformal prediction set contains multiple classes.",
}


def _class_item(class_id: int) -> dict[str, int | str]:
    return {"id": int(class_id), "name": CLASS_NAMES[int(class_id)]}


def resolve_alpha(config: ProjectConfig, requested: float | None) -> float:
    value = config.uncertainty.upload_demo_alpha if requested is None else float(requested)
    matches = [
        configured
        for configured in config.uncertainty.alpha_values
        if np.isclose(value, configured, rtol=0.0, atol=1e-12)
    ]
    if len(matches) != 1:
        supported = ", ".join(f"{alpha:g}" for alpha in config.uncertainty.alpha_values)
        raise ValueError(f"Unsupported alpha. Choose one of: {supported}.")
    return float(matches[0])


@dataclass(frozen=True)
class InferenceResult:
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.payload


class InferenceService:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.seed = config.inference.model_seed
        self.device = select_device(config.device)
        artifacts = seed_artifact_paths(config, self.seed)
        self.model, checkpoint = load_checkpoint(artifacts.checkpoint, self.device)
        run_hash = configuration_hash(config)
        if int(checkpoint["seed"]) != self.seed or checkpoint["config_hash"] != run_hash:
            raise ValueError("The selected inference checkpoint is incompatible with configuration")
        self.fitted, self.uncertainty_metadata = load_uncertainty_artifact(
            artifacts.uncertainty,
            expected_seed=self.seed,
            expected_config_hash=run_hash,
            expected_checkpoint_sha256=file_sha256(artifacts.checkpoint),
        )
        self.model.eval()
        self.checkpoint_path = artifacts.checkpoint
        self.uncertainty_path = artifacts.uncertainty

    def _decision(
        self,
        predicted_class: int,
        confidence: float,
        standard_set: tuple[int, ...],
    ) -> AbstentionDecision:
        policy = self.config.uncertainty.upload_demo_abstention_policy
        threshold = self.config.uncertainty.upload_demo_abstention_threshold
        if policy == "confidence_threshold":
            return confidence_threshold_decision(predicted_class, confidence, float(threshold))
        if policy == "singleton_conformal":
            return singleton_set_decision(standard_set)
        if policy == "hybrid":
            return hybrid_decision(predicted_class, confidence, standard_set, float(threshold))
        raise ValueError("No upload-demo abstention policy is configured")

    def classify(self, image: PreprocessedImage, *, alpha: float | None = None) -> InferenceResult:
        selected_alpha = resolve_alpha(self.config, alpha)
        with torch.inference_mode():
            logits = self.model(image.model_tensor.to(self.device))
            outputs = self.fitted.apply(logits.detach().cpu(), selected_alpha)
        raw_confidence = float(outputs.raw.confidence[0])
        calibrated_confidence = float(outputs.calibrated.confidence[0])
        predicted_class = int(outputs.calibrated.predicted_classes[0])
        probabilities = outputs.calibrated.probabilities[0]
        top_probabilities, top_classes = torch.topk(probabilities, k=3)
        standard_set = outputs.standard_sets[0]
        conditional_set = outputs.class_conditional_sets[0]
        decision = self._decision(predicted_class, calibrated_confidence, standard_set)
        accepted_class = (
            _class_item(int(decision.predicted_class))
            if decision.accepted and decision.predicted_class is not None
            else None
        )
        payload = {
            "model_seed": self.seed,
            "predicted_class": _class_item(predicted_class),
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "calibrated_probabilities": [float(value) for value in probabilities],
            "top_classes": [
                {**_class_item(int(class_id)), "probability": float(probability)}
                for probability, class_id in zip(top_probabilities, top_classes)
            ],
            "standard_conformal_set": [_class_item(class_id) for class_id in standard_set],
            "class_conditional_conformal_set": [
                _class_item(class_id) for class_id in conditional_set
            ],
            "alpha": selected_alpha,
            "temperature": float(self.fitted.temperature),
            "abstention": {
                "accepted": decision.accepted,
                "predicted_class": accepted_class,
                "policy": self.config.uncertainty.upload_demo_abstention_policy,
                "threshold": self.config.uncertainty.upload_demo_abstention_threshold,
                "reason_code": decision.reason,
            },
            "decision_reason": REASON_MESSAGES[decision.reason],
            "processed_image": {
                "mime_type": "image/png",
                "width": 28,
                "height": 28,
                "data_url": image.processed_data_url,
                "inverted": image.inverted,
            },
            "domain_limitation": DOMAIN_LIMITATION,
        }
        return InferenceResult(payload)

    def classify_upload(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str,
        invert: bool | None = None,
        alpha: float | None = None,
    ) -> InferenceResult:
        image = preprocess_upload(
            content,
            filename=filename,
            content_type=content_type,
            config=self.config,
            invert=invert,
        )
        return self.classify(image, alpha=alpha)
