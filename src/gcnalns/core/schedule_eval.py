"""No-fault schedule evaluation (Modeling doc sec.5, Spec sec.11.1-2).

Given an Instance and a Solution (placement + mode), compute a deterministic
no-fault schedule by topological list-scheduling with per-node serialization:

* execution time      p_ik = w_i / c_k
* per-replica start    s_ik = max(node_free[k], max_pred data_arrival)
* per-replica finish   f_ik = s_ik + p_ik
* effective finish     F_i  = min_{replicas} f_ik   (active backup => earliest)
* precedence           data_arrival(p -> k) = min_q (f_{p,q} + comm(node_q, k))

The execution order O on each node is the topological order restricted to that
node, which is always a valid linear extension of the DAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..data.links import LinkFactory
from .dag import topological_order
from .model import Instance, Solution


@dataclass
class ScheduleResult:
    start: dict[tuple[int, int], float] = field(default_factory=dict)   # (i,k) -> start
    finish: dict[tuple[int, int], float] = field(default_factory=dict)  # (i,k) -> finish
    eff_finish: dict[int, float] = field(default_factory=dict)          # i -> F_i
    eff_node: dict[int, int] = field(default_factory=dict)              # i -> earliest replica node
    node_free: dict[int, float] = field(default_factory=dict)           # k -> free time
    cmax: float = 0.0
    feasible: bool = True
    # task -> latest finish time given THIS schedule's makespan (for slack)
    sched_lft: dict[int, float] = field(default_factory=dict)
    sched_slack: dict[int, float] = field(default_factory=dict)

    def exec_time(self, i: int, k: int, instance: Instance) -> float:
        return instance.workflow.tasks[i].workload / instance.continuum.nodes[k].compute


def exec_time(instance: Instance, i: int, k: int) -> float:
    return instance.workflow.tasks[i].workload / instance.continuum.nodes[k].compute


def compute_schedule(instance: Instance, solution: Solution, factory: LinkFactory,
                     cfg: Config | None = None) -> ScheduleResult:
    """Compute the no-fault schedule for ``solution``."""
    wf = instance.workflow
    order = topological_order(wf)
    pred = {t: wf.predecessors(t) for t in wf.tasks}

    res = ScheduleResult()
    node_free: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}

    for i in order:
        replicas = solution.replicas(i)
        if not replicas:
            res.feasible = False
            continue
        # earliest data-ready time at each candidate replica node
        for k in replicas:
            ready = node_free[k]
            for p in pred[i]:
                # data volume on edge (p, i)
                best = float("inf")
                d = wf.data.get((p, i), 0.0)
                for q in solution.replicas(p):
                    arrive = res.finish[(p, q)] + factory.comm_delay(q, k, d)
                    best = min(best, arrive)
                ready = max(ready, best)
            p_ik = exec_time(instance, i, k)
            res.start[(i, k)] = ready
            res.finish[(i, k)] = ready + p_ik
            node_free[k] = ready + p_ik
        # effective finish = earliest finishing replica
        eff_k = min(replicas, key=lambda k: res.finish[(i, k)])
        res.eff_finish[i] = res.finish[(i, eff_k)]
        res.eff_node[i] = eff_k

    res.node_free = node_free
    res.cmax = max(res.eff_finish.values()) if res.eff_finish else 0.0

    _compute_sched_slack(instance, solution, res, factory)
    return res


def _compute_sched_slack(instance: Instance, solution: Solution,
                         res: ScheduleResult, factory: LinkFactory) -> None:
    """Latest finish time / slack for each task w.r.t. this schedule's makespan.

    slack_i = LFT_i - F_i where LFT is computed backward from C_max using the
    effective placement comm delays.
    """
    wf = instance.workflow
    order = topological_order(wf)
    succ = {t: wf.successors(t) for t in wf.tasks}
    lft: dict[int, float] = {}
    for i in reversed(order):
        if not succ[i]:
            lft[i] = res.cmax
        else:
            vals = []
            ki = res.eff_node.get(i)
            for j in succ[i]:
                kj = res.eff_node.get(j)
                comm = factory.comm_delay(ki, kj, wf.data.get((i, j), 0.0)) if ki is not None and kj is not None else 0.0
                p_j = exec_time(instance, j, kj) if kj is not None else 0.0
                vals.append(lft[j] - p_j - comm)
            lft[i] = min(vals) if vals else res.cmax
    for i in wf.tasks:
        res.sched_lft[i] = lft.get(i, res.cmax)
        res.sched_slack[i] = max(lft.get(i, res.cmax) - res.eff_finish.get(i, 0.0), 0.0)


def check_resource_feasibility(instance: Instance, solution: Solution) -> bool:
    """Verify per-node CPU/mem capacity is not exceeded (sum sigma_i <= R_k^cap)."""
    used_cpu: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}
    used_mem: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}
    for i, nodes in solution.placement.items():
        task = instance.workflow.tasks[i]
        for k in nodes:
            used_cpu[k] += task.resource
            used_mem[k] += task.mem
    for k, node in instance.continuum.nodes.items():
        if used_cpu[k] > node.res_cpu + 1e-9 or used_mem[k] > node.res_mem + 1e-9:
            return False
    return True


def check_fault_domain(instance: Instance, solution: Solution) -> bool:
    """Active-backup replicas of a task must lie in distinct fault domains."""
    for i, nodes in solution.placement.items():
        if not solution.is_active(i):
            continue
        domains = [instance.continuum.nodes[k].fault_domain for k in nodes]
        if len(set(domains)) != len(domains):
            return False
    return True
