from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ProjectConfig


SUPPORTED_EXTENSIONS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
}


class UploadValidationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PreprocessedImage:
    model_tensor: torch.Tensor
    processed_pixels: np.ndarray
    processed_data_url: str
    original_width: int
    original_height: int
    inverted: bool


def _square_pad(image: Image.Image) -> Image.Image:
    side = max(image.size)
    square = Image.new("RGB", (side, side), color=(0, 0, 0))
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    square.paste(image, (left, top))
    return square


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def preprocess_upload(
    content: bytes,
    *,
    filename: str,
    content_type: str,
    config: ProjectConfig,
    invert: bool | None = None,
) -> PreprocessedImage:
    extension = Path(filename or "").suffix.lower()
    expected_format = SUPPORTED_EXTENSIONS.get(extension)
    if expected_format is None:
        raise UploadValidationError("Only JPG, JPEG, and PNG images are supported.")
    mime_format = SUPPORTED_CONTENT_TYPES.get((content_type or "").lower())
    if mime_format is None:
        raise UploadValidationError("Unsupported image content type.")
    if mime_format != expected_format:
        raise UploadValidationError("The filename extension and image content type do not match.")
    if not content:
        raise UploadValidationError("The uploaded image is empty.")
    if len(content) > config.inference.max_upload_bytes:
        raise UploadValidationError(
            f"The uploaded image exceeds the {config.inference.max_upload_bytes}-byte limit.",
            status_code=413,
        )

    try:
        with Image.open(io.BytesIO(content)) as candidate:
            detected_format = candidate.format
            candidate.verify()
        if detected_format != expected_format:
            raise UploadValidationError("The decoded image format does not match the upload metadata.")
        with Image.open(io.BytesIO(content)) as candidate:
            candidate.load()
            oriented = ImageOps.exif_transpose(candidate)
            rgb = oriented.convert("RGB")
    except UploadValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise UploadValidationError("The uploaded file is not a readable image.") from error

    original_width, original_height = rgb.size
    square = _square_pad(rgb)
    grayscale = ImageOps.grayscale(square).resize((28, 28), Image.Resampling.BILINEAR)
    should_invert = config.inference.default_invert if invert is None else bool(invert)
    if should_invert:
        grayscale = ImageOps.invert(grayscale)

    pixels = np.asarray(grayscale, dtype=np.uint8).copy()
    raw_tensor = torch.from_numpy(pixels).to(torch.float32).div(255.0).unsqueeze(0).unsqueeze(0)
    mean = torch.tensor(config.data.normalization_mean, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(config.data.normalization_std, dtype=torch.float32).view(1, -1, 1, 1)
    model_tensor = (raw_tensor - mean) / std
    if model_tensor.shape != (1, 1, 28, 28):
        raise RuntimeError("Upload preprocessing produced an invalid model tensor shape")
    return PreprocessedImage(
        model_tensor=model_tensor,
        processed_pixels=pixels,
        processed_data_url=_png_data_url(grayscale),
        original_width=original_width,
        original_height=original_height,
        inverted=should_invert,
    )
