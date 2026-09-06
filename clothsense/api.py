from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api_schemas import (
    ChartItem,
    ChartsResponse,
    ClassificationResponse,
    ClassItem,
    ResultsSummaryResponse,
)
from .config import ProjectConfig, load_config
from .data import CLASS_NAMES
from .inference_service import InferenceService, resolve_alpha
from .result_access import list_research_charts, load_dashboard_summary
from .upload_preprocessing import UploadValidationError


@lru_cache(maxsize=1)
def get_config() -> ProjectConfig:
    return load_config()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService(get_config())


def create_app() -> FastAPI:
    config = get_config()
    application = FastAPI(title="ClothSense local API", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Max-Upload-Bytes", "X-Supported-Alphas", "X-Default-Invert"],
    )
    application.mount(
        "/charts",
        StaticFiles(directory=str(config.paths.plots), check_dir=True),
        name="charts",
    )

    @application.get("/api/classes", response_model=list[ClassItem])
    def classes(response: Response) -> list[dict[str, int | str]]:
        response.headers["X-Max-Upload-Bytes"] = str(config.inference.max_upload_bytes)
        response.headers["X-Supported-Alphas"] = ",".join(
            f"{alpha:g}" for alpha in config.uncertainty.alpha_values
        )
        response.headers["X-Default-Invert"] = str(config.inference.default_invert).lower()
        return [{"id": class_id, "name": name} for class_id, name in enumerate(CLASS_NAMES)]

    @application.post("/api/classify", response_model=ClassificationResponse)
    async def classify(
        file: list[UploadFile] = File(...),
        invert: bool | None = Form(default=None),
        alpha: float | None = Form(default=None),
        service: InferenceService = Depends(get_inference_service),
    ) -> dict:
        if len(file) != 1:
            raise HTTPException(status_code=400, detail="Exactly one image must be uploaded.")
        upload = file[0]
        try:
            selected_alpha = resolve_alpha(config, alpha)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        content = await upload.read(config.inference.max_upload_bytes + 1)
        try:
            return service.classify_upload(
                content,
                filename=upload.filename or "",
                content_type=upload.content_type or "",
                invert=invert,
                alpha=selected_alpha,
            ).as_dict()
        except UploadValidationError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from None
        except (FileNotFoundError, RuntimeError, ValueError):
            raise HTTPException(
                status_code=503,
                detail="Inference artifacts are unavailable or incompatible.",
            ) from None
        finally:
            await upload.close()

    @application.get("/api/results/summary", response_model=ResultsSummaryResponse)
    def results_summary() -> dict:
        try:
            return load_dashboard_summary(config)
        except (FileNotFoundError, KeyError, ValueError):
            raise HTTPException(status_code=503, detail="Saved research results are unavailable.") from None

    @application.get("/api/results/charts", response_model=ChartsResponse)
    def result_charts() -> dict[str, list[ChartItem]]:
        return {"charts": list_research_charts(config)}

    return application


app = create_app()
