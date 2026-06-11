"""Deterministic seeding helpers for reproducibility (Spec sec.21)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional at import time
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch may be missing in some envs
        pass


def make_rng(seed: int) -> np.random.Generator:
    """Return an isolated NumPy Generator for a given seed."""
    return np.random.default_rng(seed)
