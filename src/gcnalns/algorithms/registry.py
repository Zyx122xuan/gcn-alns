"""Registry mapping algorithm names to Scheduler classes."""
from __future__ import annotations

from .base import Scheduler
from .baselines import (
    CriticalityHybrid,
    CriticalPathBackup,
    FullProactive,
    NoFT,
    ReactiveOnly,
    ReliabilityAware,
)
from .heft import HEFT, PEFT

# Baseline registry. ALNS / GCN-ALNS are registered lazily to avoid importing
# torch when only baselines are needed.
_BASE: dict[str, type[Scheduler]] = {
    "HEFT": HEFT,
    "PEFT": PEFT,
    "No-FT": NoFT,
    "Reactive-Only": ReactiveOnly,
    "Full-Proactive": FullProactive,
    "Critical-Path Backup": CriticalPathBackup,
    "Criticality-Based Hybrid": CriticalityHybrid,
    "Reliability-Aware": ReliabilityAware,
}


def get_scheduler(name: str):
    """Return the Scheduler class for ``name`` (incl. ALNS / GCN-ALNS)."""
    if name in _BASE:
        return _BASE[name]
    if name == "ALNS":
        from .alns.solver import ALNS
        return ALNS
    if name == "GCN-ALNS":
        from .gcn_alns import GCNALNS
        return GCNALNS
    raise KeyError(f"unknown algorithm: {name}")


def baseline_names() -> list[str]:
    return list(_BASE.keys())
