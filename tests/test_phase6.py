from __future__ import annotations

import io
import base64
from dataclasses import replace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from clothsense.api import app, get_inference_service
from clothsense.config import load_config
from clothsense.inference_service import InferenceService, REASON_MESSAGES
from clothsense.upload_preprocessing import UploadValidationError, preprocess_upload


def image_bytes(
    image_format: str = "PNG",
    *,
    size: tuple[int, int] = (36, 20),
    orientation: int | None = None,
) -> bytes:
    image = Image.new("RGB", size, color=(220, 80, 40))
    buffer = io.BytesIO()
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        kwargs["exif"] = exif
    image.save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("filename", "content_type", "image_format"),
    [
        ("sample.jpg", "image/jpeg", "JPEG"),
        ("sample.jpeg", "image/jpeg", "JPEG"),
        ("sample.png", "image/png", "PNG"),
    ],
)
def test_upload_accepts_supported_formats(config, filename, content_type, image_format):
    result = preprocess_upload(
        image_bytes(image_format),
        filename=filename,
        content_type=content_type,
        config=config,
    )
    assert result.model_tensor.shape == (1, 1, 28, 28)
    assert result.processed_data_url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(result.processed_data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(decoded)) as processed:
        assert processed.size == (28, 28)
        assert np.array_equal(np.asarray(processed), result.processed_pixels)


def test_upload_rejects_unsupported_malformed_and_oversized(config):
    with pytest.raises(UploadValidationError, match="Only JPG"):
        preprocess_upload(b"GIF89a", filename="sample.gif", content_type="image/gif", config=config)
    with pytest.raises(UploadValidationError, match="not a readable image"):
        preprocess_upload(b"not an image", filename="sample.png", content_type="image/png", config=config)
    small_limit = replace(config, inference=replace(config.inference, max_upload_bytes=10))
    with pytest.raises(UploadValidationError) as error:
        preprocess_upload(image_bytes(), filename="sample.png", content_type="image/png", config=small_limit)
    assert error.value.status_code == 413


def test_preprocessing_is_deterministic_normalized_and_invertible(config):
    content = image_bytes(size=(50, 18))
    first = preprocess_upload(content, filename="sample.png", content_type="image/png", config=config, invert=False)
    second = preprocess_upload(content, filename="sample.png", content_type="image/png", config=config, invert=False)
    inverted = preprocess_upload(content, filename="sample.png", content_type="image/png", config=config, invert=True)
    assert np.array_equal(first.processed_pixels, second.processed_pixels)
    assert torch.equal(first.model_tensor, second.model_tensor)
    assert first.processed_pixels.shape == (28, 28)
    assert np.array_equal(inverted.processed_pixels, 255 - first.processed_pixels)
    expected = (
        torch.from_numpy(first.processed_pixels).float().div(255.0)
        - config.data.normalization_mean[0]
    ) / config.data.normalization_std[0]
    assert torch.allclose(first.model_tensor[0, 0], expected)
    assert first.processed_pixels[0].mean() < first.processed_pixels[14].mean()


def test_exif_orientation_is_applied(config):
    content = image_bytes("JPEG", size=(10, 20), orientation=6)
    result = preprocess_upload(
        content,
        filename="oriented.jpg",
        content_type="image/jpeg",
        config=config,
    )
    assert (result.original_width, result.original_height) == (20, 10)
    assert result.model_tensor.shape == (1, 1, 28, 28)


@pytest.fixture(scope="module")
def inference_service() -> InferenceService:
    return InferenceService(load_config())


@pytest.fixture(scope="module")
def api_client(inference_service):
    app.dependency_overrides[get_inference_service] = lambda: inference_service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_inference_service_schema_and_no_refitting(inference_service):
    config = inference_service.config
    checkpoint_time = inference_service.checkpoint_path.stat().st_mtime_ns
    uncertainty_time = inference_service.uncertainty_path.stat().st_mtime_ns
    result = inference_service.classify_upload(
        image_bytes(),
        filename="sample.png",
        content_type="image/png",
    ).as_dict()
    assert result["model_seed"] == config.inference.model_seed
    assert result["alpha"] == config.uncertainty.upload_demo_alpha
    assert result["abstention"]["policy"] == config.uncertainty.upload_demo_abstention_policy
    assert 0.0 <= result["raw_confidence"] <= 1.0
    assert 0.0 <= result["calibrated_confidence"] <= 1.0
    assert len(result["calibrated_probabilities"]) == 10
    assert sum(result["calibrated_probabilities"]) == pytest.approx(1.0)
    top = [item["probability"] for item in result["top_classes"]]
    assert len(top) == 3 and top == sorted(top, reverse=True)
    assert all(0 <= item["id"] < 10 for item in result["standard_conformal_set"])
    assert all(0 <= item["id"] < 10 for item in result["class_conditional_conformal_set"])
    assert result["decision_reason"] == REASON_MESSAGES[result["abstention"]["reason_code"]]
    assert inference_service.checkpoint_path.stat().st_mtime_ns == checkpoint_time
    assert inference_service.uncertainty_path.stat().st_mtime_ns == uncertainty_time


def test_api_classes_summary_and_charts(api_client):
    classes = api_client.get("/api/classes")
    assert classes.status_code == 200
    assert len(classes.json()) == 10
    assert classes.json()[0] == {"id": 0, "name": "T-shirt/top"}
    summary = api_client.get("/api/results/summary")
    assert summary.status_code == 200
    assert {"clean", "strongest_shifts", "demo_configuration"} <= summary.json().keys()
    charts = api_client.get("/api/results/charts")
    assert charts.status_code == 200
    assert len(charts.json()["charts"]) == 10
    assert all(item["url"].startswith("/charts/") for item in charts.json()["charts"])
    for item in charts.json()["charts"]:
        chart = api_client.get(item["url"])
        assert chart.status_code == 200
        assert chart.headers["content-type"] == "image/png"


def test_api_classify_matches_direct_inference_and_retains_no_upload(api_client, inference_service):
    content = image_bytes()
    before = {path for path in inference_service.config.paths.processed_examples.rglob("*") if path.is_file()}
    direct = inference_service.classify_upload(
        content,
        filename="sample.png",
        content_type="image/png",
        invert=True,
        alpha=0.10,
    ).as_dict()
    response = api_client.post(
        "/api/classify",
        files=[("file", ("sample.png", content, "image/png"))],
        data={"invert": "true", "alpha": "0.10"},
    )
    assert response.status_code == 200, response.text
    api_result = response.json()
    for key in (
        "predicted_class",
        "raw_confidence",
        "calibrated_confidence",
        "standard_conformal_set",
        "class_conditional_conformal_set",
        "abstention",
        "decision_reason",
    ):
        assert api_result[key] == direct[key]
    after = {path for path in inference_service.config.paths.processed_examples.rglob("*") if path.is_file()}
    assert after == before


def test_api_rejects_invalid_upload_count_file_and_alpha(api_client):
    content = image_bytes()
    unsupported_alpha = api_client.post(
        "/api/classify",
        files=[("file", ("sample.png", content, "image/png"))],
        data={"alpha": "0.15"},
    )
    assert unsupported_alpha.status_code == 422
    malformed = api_client.post(
        "/api/classify",
        files=[("file", ("sample.png", b"broken", "image/png"))],
    )
    assert malformed.status_code == 400
    multiple = api_client.post(
        "/api/classify",
        files=[
            ("file", ("one.png", content, "image/png")),
            ("file", ("two.png", content, "image/png")),
        ],
    )
    assert multiple.status_code == 400
