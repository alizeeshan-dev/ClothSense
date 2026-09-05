from pathlib import Path

import clothsense
from clothsense.config import load_config
from clothsense.reproducibility import select_device


def test_package_import_and_default_config_load() -> None:
    config = load_config()
    assert clothsense.__version__ == "0.1.0"
    assert config.training.batch_size == 128
    assert config.data.image_size == (28, 28)
    assert config.uncertainty.alpha_values == (0.05, 0.10, 0.20)
    assert config.paths.data == Path(__file__).resolve().parents[1] / "data"
    assert select_device(config.device).type in {"cpu", "cuda"}


def test_output_directories_are_created(config) -> None:
    config.ensure_directories()
    assert all(path.is_dir() for path in config.paths.directories())

