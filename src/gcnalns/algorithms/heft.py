"""HEFT and PEFT list-scheduling heuristics.

HEFT: rank tasks by upward rank (descending) and greedily assign each to the
node that yields the earliest finish time, respecting node capacity.

PEFT: uses an Optimistic Cost Table (OCT); tasks ranked by mean OCT, assigned
to minimize (EFT + OCT(task, node)).

Both produce single-instance (passive) placements; they serve as the SLA
reference (HEFT-NoFT) and as baselines.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..core.dag import compute_structural, topological_order
from ..core.model import Instance, Solution
from ..data.links import LinkFactory, get_factory
from .base import Scheduler, topo_node_order


def _avg_compute(instance: Instance) -> float:
    return float(np.mean([n.compute for n in instance.continuum.nodes.values()]))


def _topo_priority_order(wf, priority_order: list[int]) -> list[int]:
    """Return a topological order that follows ``priority_order`` among ready tasks.

    Kahn's algorithm where, at each step, the ready task with the smallest rank
    in ``priority_order`` (i.e. highest caller priority) is emitted next. This
    yields a valid linear DAG extension while honouring the heuristic priority,
    so a task is never scheduled before any of its predecessors.
    """
    rank = {t: r for r, t in enumerate(priority_order)}
    indeg = {t: 0 for t in wf.tasks}
    for (_, j) in wf.edges:
        indeg[j] += 1
    succ = {t: wf.successors(t) for t in wf.tasks}
    ready = [t for t in wf.tasks if indeg[t] == 0]
    out: list[int] = []
    while ready:
        ready.sort(key=lambda t: rank.get(t, len(rank)))
        u = ready.pop(0)
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return out


def _greedy_eft_place(instance: Instance, factory: LinkFactory,
                      task_order: list[int],
                      oct_table: dict[tuple[int, int], float] | None = None,
                      extra_metric=None) -> Solution:
    """Shared greedy earliest-finish-time placement used by HEFT/PEFT/baselines.

    ``extra_metric(i, k, eft)`` (optional) adds a term to the selection metric,
    used by reliability-aware baselines to bias toward low-failure nodes.
    """
    wf = instance.workflow
    nodes = instance.continuum.nodes
    pred = {t: wf.predecessors(t) for t in wf.tasks}

    # Respect the DAG: turn the priority order into a valid topological order,
    # breaking ties by the caller's priority (so a task is never placed before
    # one of its predecessors). HEFT's ranku order is already topological; PEFT's
    # mean-OCT order is not guaranteed to be, so this guard is required.
    task_order = _topo_priority_order(wf, task_order)

    sol = Solution()
    node_free = {k: 0.0 for k in nodes}
    used_cpu = {k: 0.0 for k in nodes}
    used_mem = {k: 0.0 for k in nodes}
    finish: dict[int, float] = {}
    where: dict[int, int] = {}

    for i in task_order:
        task = wf.tasks[i]
        best_k, best_eft, best_start = None, float("inf"), 0.0
        best_metric = float("inf")
        for k, node in nodes.items():
            if used_cpu[k] + task.resource > node.res_cpu + 1e-9:
                continue
            if used_mem[k] + task.mem > node.res_mem + 1e-9:
                continue
            est = node_free[k]
            for p in pred[i]:
                d = wf.data.get((p, i), 0.0)
                est = max(est, finish[p] + factory.comm_delay(where[p], k, d))
            p_ik = task.workload / node.compute
            eft = est + p_ik
            metric = eft + (oct_table.get((i, k), 0.0) if oct_table else 0.0)
            if extra_metric is not None:
                metric += extra_metric(i, k, eft)
            if metric < best_metric:
                best_metric, best_eft, best_start, best_k = metric, eft, est, k
        if best_k is None:  # capacity exhausted: fall back to earliest-free node
            best_k = min(nodes, key=lambda k: node_free[k])
            node = nodes[best_k]
            est = node_free[best_k]
            for p in pred[i]:
                d = wf.data.get((p, i), 0.0)
                est = max(est, finish[p] + factory.comm_delay(where[p], best_k, d))
            best_eft = est + task.workload / node.compute
            sol.meta["capacity_overflow"] = True
        sol.placement[i] = [best_k]
        sol.mode[i] = False
        node_free[best_k] = best_eft
        used_cpu[best_k] += task.resource
        used_mem[best_k] += task.mem
        finish[i] = best_eft
        where[i] = best_k

    sol.order = topo_node_order(instance, sol)
    return sol


class HEFT(Scheduler):
    name = "HEFT"

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)
        compute_structural(instance.workflow, instance.continuum, self.cfg, factory)
        order = sorted(instance.workflow.tasks,
                       key=lambda t: instance.workflow.tasks[t].ranku, reverse=True)
        sol = _greedy_eft_place(instance, factory, order)
        self.runtime = time.perf_counter() - t0
        return sol


def _compute_oct(instance: Instance, factory: LinkFactory) -> dict[tuple[int, int], float]:
    """Optimistic Cost Table for PEFT (averaged over downstream paths)."""
    wf = instance.workflow
    nodes = list(instance.continuum.nodes.keys())
    succ = {t: wf.successors(t) for t in wf.tasks}
    order = topological_order(wf)
    avg_lat, avg_bw = 0.0, 0.0
    sample = []
    for k in nodes[: min(len(nodes), 20)]:
        for l in nodes[: min(len(nodes), 20)]:
            if k != l:
                link = factory.get(k, l)
                if link:
                    sample.append((link.latency, link.bandwidth))
    if sample:
        avg_lat = float(np.mean([s[0] for s in sample]))
        avg_bw = float(np.mean([s[1] for s in sample]))

    oct_t: dict[tuple[int, int], float] = {}
    for i in reversed(order):
        for k in nodes:
            if not succ[i]:
                oct_t[(i, k)] = 0.0
                continue
            val = 0.0
            for j in succ[i]:
                best = float("inf")
                d = wf.data.get((i, j), 0.0)
                for w in nodes:
                    p_jw = wf.tasks[j].workload / instance.continuum.nodes[w].compute
                    comm = 0.0 if w == k else avg_lat + d / max(avg_bw, 1e-9)
                    best = min(best, oct_t[(j, w)] + p_jw + comm)
                val = max(val, best)
            oct_t[(i, k)] = val
    return oct_t


class PEFT(Scheduler):
    name = "PEFT"

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)
        compute_structural(instance.workflow, instance.continuum, self.cfg, factory)
        oct_t = _compute_oct(instance, factory)
        nodes = list(instance.continuum.nodes.keys())
        rank_oct = {i: float(np.mean([oct_t[(i, k)] for k in nodes])) for i in instance.workflow.tasks}
        order = sorted(instance.workflow.tasks, key=lambda t: rank_oct[t], reverse=True)
        sol = _greedy_eft_place(instance, factory, order, oct_table=oct_t)
        self.runtime = time.perf_counter() - t0
        return sol
