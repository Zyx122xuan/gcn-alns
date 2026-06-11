"""Cost model (Modeling doc sec.10).

C_total = C_compute + C_backup + C_comm + C_reserve + C_recovery

* C_compute  = sum over all deployed instances rho_k * p_ik
* C_backup   = sum over active-backup tasks, over their instances rho_k * p_ik
* C_comm     = sum over edges rho_comm * d_ij  (effective cross-node transfers)
* C_reserve  = rho_res * sum over passive tasks sigma_i  (reserved recompute capacity)
* C_recovery = recovery_cost_weight * E[extra recovery compute]  (secondary; optional)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..data.links import LinkFactory
from .model import Instance, Solution
from .schedule_eval import ScheduleResult, exec_time


@dataclass
class CostBreakdown:
    compute: float = 0.0
    backup: float = 0.0
    comm: float = 0.0
    reserve: float = 0.0
    recovery: float = 0.0

    @property
    def total(self) -> float:
        return self.compute + self.backup + self.comm + self.reserve + self.recovery

    def as_dict(self) -> dict:
        return {
            "compute": self.compute,
            "backup": self.backup,
            "comm": self.comm,
            "reserve": self.reserve,
            "recovery": self.recovery,
            "total": self.total,
        }


def compute_cost(instance: Instance, solution: Solution, sched: ScheduleResult,
                 factory: LinkFactory, cfg: Config,
                 recovery_cost: float = 0.0) -> CostBreakdown:
    wf = instance.workflow
    nodes = instance.continuum.nodes
    cc = cfg["cost"]

    bd = CostBreakdown()

    # compute + backup
    for i, repl in solution.placement.items():
        active = solution.is_active(i)
        for k in repl:
            c = nodes[k].cost_coef * exec_time(instance, i, k)
            bd.compute += c
            if active:
                bd.backup += c

    # communication over effective placement
    coef = cc["comm_cost_coef"]
    for (i, j) in wf.edges:
        ki = sched.eff_node.get(i)
        kj = sched.eff_node.get(j)
        if ki is None or kj is None or ki == kj:
            continue
        bd.comm += coef * wf.data.get((i, j), 0.0)

    # reserved recompute capacity for passive tasks
    rho_res = cc["reserve_cost_coef"]
    for i, repl in solution.placement.items():
        if not solution.is_active(i):
            bd.reserve += rho_res * wf.tasks[i].resource

    bd.recovery = cc["recovery_cost_weight"] * recovery_cost
    return bd
