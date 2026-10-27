"""Cost model (Modeling doc sec.10).

C_total = C_compute + C_comm + C_reserve + C_recovery

* C_compute  = sum over all deployed instances rho_k * p_ik
               (this already covers active-backup replicas, since every deployed
               instance is counted; C_backup is therefore NOT added to the total
               to avoid double-counting active-backup replica cost)
* C_backup   = sum over active-backup tasks, over their instances rho_k * p_ik
               (informational breakdown only; excluded from C_total)
* C_comm     = sum_{(i,j) in A} sum_{k in repl(i), l in repl(j)} x_ik x_jl rho_kl d_ij
               (summed over ALL replica pairs, consistent with the schedule model
               where every replica of a task receives its predecessors' data;
               intra-node pairs k == l incur no transfer cost)
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
        # NOTE: backup is intentionally excluded. ``compute`` already sums over
        # every deployed instance (active-backup replicas included), so adding
        # ``backup`` here would double-count active-backup replica cost.
        return self.compute + self.comm + self.reserve + self.recovery

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

    # communication over ALL replica pairs (Modeling doc sec.10):
    #   C_comm = sum_{(i,j)} sum_{k in repl(i), l in repl(j)} rho_kl * d_ij
    # This matches compute/schedule accounting, where every replica of task j
    # receives input data from the predecessor i. Intra-node pairs (k == l) are
    # local transfers and incur no cost.
    coef = cc["comm_cost_coef"]
    for (i, j) in wf.edges:
        d = wf.data.get((i, j), 0.0)
        if d == 0.0:
            continue
        for ki in solution.replicas(i):
            for kj in solution.replicas(j):
                if ki == kj:
                    continue
                bd.comm += coef * d

    # reserved recompute capacity for passive tasks
    rho_res = cc["reserve_cost_coef"]
    for i, repl in solution.placement.items():
        if not solution.is_active(i):
            bd.reserve += rho_res * wf.tasks[i].resource

    bd.recovery = cc["recovery_cost_weight"] * recovery_cost
    return bd
