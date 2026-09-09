"""Shared helpers for the paper-reproduction baselines.

All reproductions align every paper to the same static continuum problem and the
same ``core.recovery.verify`` feasibility gate. The helpers here provide the two
constructs several papers share: per-task **sub-deadlines** (constraint
distribution, RECR-D Alg.1 / HFWPF preprocessing) and a cheap **passive
recovery-risk** estimate used to steer active-backup promotion (RECR-D TRM).
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..core.model import Instance, Solution
from ..core.schedule_eval import ScheduleResult, exec_time
from ..data.links import LinkFactory


def avg_link(instance: Instance, factory: LinkFactory, samples: int = 200
             ) -> tuple[float, float]:
    """Return (avg_latency, avg_bandwidth) sampled across node pairs."""
    nodes = list(instance.continuum.nodes.keys())
    if len(nodes) < 2:
        return 0.0, 1.0
    rng = np.random.default_rng(0)
    lats, bws = [], []
    pairs = min(samples, len(nodes) * (len(nodes) - 1) // 2)
    for _ in range(pairs):
        a, b = rng.choice(nodes, size=2, replace=False)
        link = factory.get(int(a), int(b))
        if link is not None:
            lats.append(link.latency)
            bws.append(link.bandwidth)
    if not bws:
        return 0.0, 1.0
    return float(np.mean(lats)), float(np.mean(bws))


def sub_deadlines(instance: Instance, db: float) -> tuple[dict[int, float], dict[int, float]]:
    """Distribute the global deadline D to per-task (start, finish) sub-deadlines.

    Uses the structural EST/EFT/LFT (must be filled by ``compute_structural``
    beforehand) and a dynamic-balance factor ``db`` in [0, 1]:

        dl_i = (D / cmax_struct) * (db * EFT_i + (1 - db) * LFT_i)
        st_i = (D / cmax_struct) *  EST_i

    ``db -> 1`` hugs the earliest schedule (tight sub-deadlines); ``db -> 0``
    hugs the latest-feasible schedule (loose). This is the RECR-D Alg.1 idea
    (convex EFT/LFT combination) reused by HFWPF preprocessing.
    """
    tasks = instance.workflow.tasks
    cmax = max((t.eft for t in tasks.values()), default=0.0)
    D = instance.deadline
    scale = (D / cmax) if cmax > 1e-12 else 1.0
    st: dict[int, float] = {}
    dl: dict[int, float] = {}
    for i, t in tasks.items():
        dl[i] = scale * (db * t.eft + (1.0 - db) * t.lft)
        st[i] = scale * t.est
    return st, dl


def passive_recovery_risk(instance: Instance, sol: Solution,
                          sched: ScheduleResult, factory: LinkFactory,
                          cfg: Config) -> dict[int, float]:
    """Per-passive-task local recovery risk vs. schedule slack.

    Mirrors the passive branch of :func:`core.recovery.estimate_rmax`: the local
    reactive-recovery delay (detect + replan + queue + input comm + recompute)
    minus the task's slack. Positive values are the tasks that push R_max over
    budget -- the natural candidates for TRM active-backup promotion.
    """
    wf = instance.workflow
    rcv = cfg["recovery"]
    det, replan = rcv["delta_detect"], rcv["delta_replan"]
    risks: dict[int, float] = {}
    for i, repl in sol.placement.items():
        if sol.is_active(i) or not repl:
            continue
        k = repl[0]
        node = instance.continuum.nodes[k]
        slack = sched.sched_slack.get(i, 0.0)
        comm_in = 0.0
        for p in wf.predecessors(i):
            d = wf.data.get((p, i), 0.0)
            kp = sched.eff_node.get(p)
            if kp is not None:
                comm_in += factory.comm_delay(kp, k, d)
        p_ik = exec_time(instance, i, k)
        local = det + replan + node.queue_wait + comm_in + p_ik
        risks[i] = max(0.0, local - slack)
    return risks
