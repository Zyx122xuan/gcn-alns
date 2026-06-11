"""Configuration loading and access helpers.

All experiment parameters live in ``config/default.yaml``. This module loads
that file (or a user-supplied override) into a plain nested dict and exposes a
small helper for dotted-key access so downstream code never hard-codes
parameters.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repository root = three levels up from this file (src/gcnalns/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def _coerce_inf(obj: Any) -> Any:
    """Recursively convert YAML ``.inf`` sentinels into ``math.inf`` floats."""
    if isinstance(obj, dict):
        return {k: _coerce_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_inf(v) for v in obj]
    if isinstance(obj, str) and obj.strip().lower() in {".inf", "inf", "+inf"}:
        return math.inf
    return obj


class Config:
    """Lightweight wrapper around the configuration dictionary."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Access nested values with a dotted path, e.g. ``"alns.max_iter"``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def data(self) -> dict[str, Any]:
        return self._data


@lru_cache(maxsize=8)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration from YAML (cached by path)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(_coerce_inf(raw))
