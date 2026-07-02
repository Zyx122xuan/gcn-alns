"""Per-run evaluation metrics (Spec sec.18).

Every algorithm's solution is scored through the *same* recovery verifier so the
comparison is apples-to-apples. ``evaluate`` returns a flat dict (one experiment
record) covering all reported metrics:

* feasibility / SLA: feasible flag, sla_violation, safety margin
* recovery: worst (r_max) and average recovery delay, effective budget
* makespan: no-fault (c_base) and worst-case post-fault (c_worst)
* cost: total + 5-way breakdown (compute/backup/comm/reserve/recovery)
* fault-tolerance: active-backup ratio, fault-set size |Omega|
* efficiency: scheduling runtime, recovery-verification time, GCN inference
  time, convergence iteration, candidate prune ratio
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import Config
from ..core.cost import compute_cost
from ..core.model import Instance, Solution
from ..core.recovery import verify
from ..core.schedule_eval import compute_schedule
from ..data.links import get_factory

METRIC_FIELDS = (
    "iid", "algorithm", "workflow_type", "scale", "load", "sla_scenario", "rep",
    "num_tasks", "num_nodes", "deadline", "sla_recovery_window",
    "feasible", "sla_violation", "found_feasible",
    "c_base", "c_worst", "r_max", "r_eff", "avg_recovery", "safety_margin",
    "n_faults", "active_backup_ratio",
    "cost_total", "cost_compute", "cost_backup", "cost_comm",
    "cost_reserve", "cost_recovery",
    "runtime", "verify_time", "gcn_infer_time", "convergence_iter", "prune_ratio",
)


def active_backup_ratio(instance: Instance, solution: Solution) -> float:
    tasks = instance.workflow.tasks
    if not tasks:
        return 0.0
    n_active = sum(1 for i in tasks if solution.is_active(i))
    return n_active / len(tasks)


def evaluate(instance: Instance, solution: Solution, cfg: Config,
             algorithm: str, factory=None, scheduler=None, rep: int = 0,
             ) -> dict:
    """Compute the full metric record for one (instance, solution) run."""
    factory = factory or get_factory(instance, cfg)
    sched = compute_schedule(instance, solution, factory, cfg)

    # Use the scheduler's ablation (if any) so the final verification matches
    # the dynamic-reschedule policy the search assumed.
    ablation = getattr(scheduler, "ablation", None) if scheduler else None
    t0 = time.perf_counter()
    res = verify(instance, solution, factory, cfg, sched, ablation=ablation)
    verify_time = time.perf_counter() - t0

    bd = compute_cost(instance, solution, sched, factory, cfg,
                      recovery_cost=res.recovery_cost)

    meta = instance.meta
    runtime = getattr(scheduler, "runtime", 0.0) if scheduler else 0.0
    conv = getattr(scheduler, "convergence_iter", 0) if scheduler else 0
    prune = getattr(scheduler, "prune_ratio", 0.0) if scheduler else 0.0
    gcn_t = getattr(scheduler, "gcn_infer_time", 0.0) if scheduler else 0.0

    return {
        "iid": instance.iid,
        "algorithm": algorithm,
        "workflow_type": meta.get("workflow_type", instance.workflow.name),
        "scale": meta.get("scale", ""),
        "load": meta.get("load", ""),
        "sla_scenario": meta.get("sla_scenario", ""),
        "rep": rep,
        "num_tasks": instance.workflow.num_tasks,
        "num_nodes": instance.continuum.num_nodes,
        "deadline": instance.deadline,
        "sla_recovery_window": instance.recovery_window,
        "feasible": bool(res.feasible),
        "sla_violation": 0.0 if res.feasible else 1.0,
        "found_feasible": bool(solution.meta.get("found_feasible", res.feasible)),
        "c_base": res.c_base,
        "c_worst": res.c_worst,
        "r_max": res.r_max,
        "r_eff": res.r_eff,
        "avg_recovery": res.avg_recovery,
        "safety_margin": res.margin,
        "n_faults": res.n_faults,
        "active_backup_ratio": active_backup_ratio(instance, solution),
        "cost_total": bd.total,
        "cost_compute": bd.compute,
        "cost_backup": bd.backup,
        "cost_comm": bd.comm,
        "cost_reserve": bd.reserve,
        "cost_recovery": bd.recovery,
        "runtime": runtime,
        "verify_time": verify_time,
        "gcn_infer_time": gcn_t,
        "convergence_iter": conv,
        "prune_ratio": prune,
    }
