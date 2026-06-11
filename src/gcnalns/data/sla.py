"""SLA parameter generation (Spec sec.7).

SLA is anchored to a unified reference schedule (HEFT-NoFT) so all algorithms
face the same deadline regardless of their own output:

    T_ref = C_base(S_HEFT)
    D     = alpha * T_ref
    R_max = beta  * T_ref

with (alpha, beta) per the tight/medium/loose scenarios in config.
"""
from __future__ import annotations

from ..config import Config
from ..core.model import Instance
from ..core.schedule_eval import compute_schedule
from .links import get_factory


def reference_makespan(instance: Instance, cfg: Config) -> float:
    """T_ref = HEFT no-fault makespan on this instance."""
    from ..algorithms.heft import HEFT

    sol = HEFT(cfg).schedule(instance)
    factory = get_factory(instance, cfg)
    sched = compute_schedule(instance, sol, factory, cfg)
    return sched.cmax


def generate_sla(instance: Instance, cfg: Config, scenario: str = "medium") -> tuple[float, float, float]:
    """Return (T_ref, D, R_max) for the given SLA scenario."""
    t_ref = reference_makespan(instance, cfg)
    sc = cfg["sla"]["scenarios"][scenario]
    deadline = sc["alpha"] * t_ref
    recovery_window = sc["beta"] * t_ref
    return t_ref, deadline, recovery_window
