from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from clothsense.config import PathConfig, ProjectConfig, load_config


@pytest.fixture
def config(tmp_path: Path) -> ProjectConfig:
    base = load_config()
    paths = PathConfig(
        data=tmp_path / "data",
        split_indices=tmp_path / "data" / "splits",
        models=tmp_path / "artifacts" / "models",
        plots=tmp_path / "artifacts" / "plots",
        processed_examples=tmp_path / "artifacts" / "processed_examples",
        personal_photos=tmp_path / "data" / "personal_photos",
        results=tmp_path / "results",
        database=tmp_path / "results" / "clothsense.sqlite",
    )
    return replace(base, root=tmp_path, paths=paths)
