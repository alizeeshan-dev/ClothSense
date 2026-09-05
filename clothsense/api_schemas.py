from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ClassItem(BaseModel):
    id: int = Field(ge=0, le=9)
    name: str


class ProbabilityClass(ClassItem):
    probability: float = Field(ge=0.0, le=1.0)


class AbstentionResult(BaseModel):
    accepted: bool
    predicted_class: ClassItem | None
    policy: str
    threshold: float | None
    reason_code: str


class ProcessedImageResult(BaseModel):
    mime_type: str
    width: int
    height: int
    data_url: str
    inverted: bool


class ClassificationResponse(BaseModel):
    model_seed: int
    predicted_class: ClassItem
    raw_confidence: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    calibrated_probabilities: list[float]
    top_classes: list[ProbabilityClass]
    standard_conformal_set: list[ClassItem]
    class_conditional_conformal_set: list[ClassItem]
    alpha: float
    temperature: float = Field(gt=0.0)
    abstention: AbstentionResult
    decision_reason: str
    processed_image: ProcessedImageResult
    domain_limitation: str


class ChartItem(BaseModel):
    name: str
    title: str
    url: str


class ChartsResponse(BaseModel):
    charts: list[ChartItem]


class ResultsSummaryResponse(BaseModel):
    source: str
    seed_count: int
    clean: dict[str, Any]
    strongest_shifts: list[dict[str, Any]]
    demo_configuration: dict[str, Any]
