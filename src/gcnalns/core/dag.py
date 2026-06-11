"""DAG structural analysis (Modeling doc sec.11.1, Spec sec.2).

Computes placement-independent structural attributes for each task:
topological order/level, in/out degree, upward rank (ranku), average-timing
EST/EFT/LFT, slack, and the structural criticality score h_i. These depend
only on DAG structure, task workload and (average) candidate compute/comm, not
on a specific deployment.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from .model import Continuum, Workflow


def topological_order(wf: Workflow) -> list[int]:
    """Kahn's algorithm; raises on cycles."""
    indeg = {t: 0 for t in wf.tasks}
    for (_, j) in wf.edges:
        indeg[j] += 1
    queue = [t for t in wf.tasks if indeg[t] == 0]
    order: list[int] = []
    succ = {t: wf.successors(t) for t in wf.tasks}
    while queue:
        u = queue.pop()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != len(wf.tasks):
        raise ValueError("workflow graph contains a cycle")
    return order


def _average_compute(continuum: Continuum) -> float:
    return float(np.mean([n.compute for n in continuum.nodes.values()]))


def _average_link(continuum: Continuum, factory=None) -> tuple[float, float]:
    """Return (avg_latency, avg_bandwidth) across layer pairs (sampled)."""
    nodes = list(continuum.nodes.keys())
    if len(nodes) < 2:
        return 0.0, 1.0
    lats, bws = [], []
    # Sample up to 200 pairs for a stable average without O(N^2) cost.
    rng = np.random.default_rng(0)
    pairs = min(200, len(nodes) * (len(nodes) - 1) // 2)
    for _ in range(pairs):
        a, b = rng.choice(nodes, size=2, replace=False)
        if factory is not None:
            link = factory.get(int(a), int(b))
        else:
            link = continuum.link(int(a), int(b))
        if link is not None:
            lats.append(link.latency)
            bws.append(link.bandwidth)
    if not bws:
        return 0.0, 1.0
    return float(np.mean(lats)), float(np.mean(bws))


def compute_structural(wf: Workflow, continuum: Continuum, cfg: Config,
                       factory=None) -> None:
    """Fill structural attributes on every task in-place."""
    order = topological_order(wf)
    avg_c = _average_compute(continuum)
    avg_lat, avg_bw = _average_link(continuum, factory)

    succ = {t: wf.successors(t) for t in wf.tasks}
    pred = {t: wf.predecessors(t) for t in wf.tasks}
    avg_p = {t: wf.tasks[t].workload / avg_c for t in wf.tasks}

    def avg_comm(i: int, j: int) -> float:
        return avg_lat + wf.data.get((i, j), 0.0) / max(avg_bw, 1e-9)

    # in/out degree
    for t in wf.tasks:
        wf.tasks[t].indeg = len(pred[t])
        wf.tasks[t].outdeg = len(succ[t])

    # levels (longest path in edges)
    level = {t: 0 for t in wf.tasks}
    for u in order:
        for v in succ[u]:
            level[v] = max(level[v], level[u] + 1)

    # upward rank (reverse topological)
    ranku: dict[int, float] = {}
    for u in reversed(order):
        if not succ[u]:
            ranku[u] = avg_p[u]
        else:
            ranku[u] = avg_p[u] + max(avg_comm(u, v) + ranku[v] for v in succ[u])

    # forward pass: EST/EFT
    est: dict[int, float] = {}
    eft: dict[int, float] = {}
    for u in order:
        if not pred[u]:
            est[u] = 0.0
        else:
            est[u] = max(eft[p] + avg_comm(p, u) for p in pred[u])
        eft[u] = est[u] + avg_p[u]
    cmax = max(eft.values()) if eft else 0.0

    # backward pass: LFT
    lft: dict[int, float] = {}
    for u in reversed(order):
        if not succ[u]:
            lft[u] = cmax
        else:
            lft[u] = min(lft[v] - avg_p[v] - avg_comm(u, v) for v in succ[u])

    crit = cfg["criticality"]
    eps = crit["eps"]
    ranku_arr = np.array([ranku[t] for t in wf.tasks], dtype=float)
    inv_slack = np.array([1.0 / (max(lft[t] - eft[t], 0.0) + eps) for t in wf.tasks])
    outdeg_arr = np.array([wf.tasks[t].outdeg for t in wf.tasks], dtype=float)

    def _norm(a: np.ndarray) -> np.ndarray:
        lo, hi = float(a.min()), float(a.max())
        return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

    h = (crit["theta1"] * _norm(ranku_arr)
         + crit["theta2"] * _norm(inv_slack)
         + crit["theta3"] * _norm(outdeg_arr))

    for idx, t in enumerate(wf.tasks):
        task = wf.tasks[t]
        task.ranku = float(ranku[t])
        task.est = float(est[t])
        task.eft = float(eft[t])
        task.lft = float(lft[t])
        task.slack = float(max(lft[t] - eft[t], 0.0))
        task.level = int(level[t])
        task.criticality = float(h[idx])
