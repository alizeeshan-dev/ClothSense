from __future__ import annotations

import io

from fastapi.testclient import TestClient

from clothsense.api import app, get_inference_service
from clothsense.config import load_config
from clothsense.data import load_raw_clean_test
from clothsense.inference_service import InferenceService


def main() -> None:
    config = load_config()
    dataset = load_raw_clean_test(config, download=False)
    image, _ = dataset[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    service = InferenceService(config)
    before = {path for path in config.paths.processed_examples.rglob("*") if path.is_file()}
    direct = service.classify_upload(
        content,
        filename="fashion_sample.png",
        content_type="image/png",
        alpha=config.uncertainty.upload_demo_alpha,
    ).as_dict()
    app.dependency_overrides[get_inference_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/classify",
                files=[("file", ("fashion_sample.png", content, "image/png"))],
                data={"alpha": str(config.uncertainty.upload_demo_alpha)},
            )
            response.raise_for_status()
            api_result = response.json()
            summary = client.get("/api/results/summary")
            charts = client.get("/api/results/charts")
            summary.raise_for_status()
            charts.raise_for_status()
    finally:
        app.dependency_overrides.clear()

    comparison_fields = (
        "predicted_class",
        "raw_confidence",
        "calibrated_confidence",
        "standard_conformal_set",
        "class_conditional_conformal_set",
        "abstention",
    )
    assert all(api_result[field] == direct[field] for field in comparison_fields)
    after = {path for path in config.paths.processed_examples.rglob("*") if path.is_file()}
    assert after == before
    assert len(charts.json()["charts"]) == 10
    print(
        f"match=true class={direct['predicted_class']['name']} "
        f"accepted={direct['abstention']['accepted']} charts=10 retained_upload=false"
    )


if __name__ == "__main__":
    main()
