from __future__ import annotations

import os
import random
from collections.abc import Callable

import numpy as np
import torch

from .config import DeviceConfig


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    """Seed supported random generators for a repeatable local run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def select_device(config: DeviceConfig) -> torch.device:
    if config.prefer_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{config.cuda_index}")
    return torch.device("cpu")


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def worker_init_fn() -> Callable[[int], None]:
    return seed_worker

