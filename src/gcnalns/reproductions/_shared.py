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

    Faithful to RECR-D Algorithm 1 (Taghinezhad-Niar & Taheri, IEEE TSC 2025,
    lines 1-17), reusing the structural EST/EFT/LFT filled by
    ``compute_structural`` (LFT there is anchored at the structural makespan
    ``cmax``; we re-anchor it additively to the deadline ``D``):

        T_min = max EFT (structural makespan)            [Alg.1 line 1]
        rho   = clip((D - T_min) / D, 0, 1)              [Alg.1 line 6, slack ratio]
        DB    = a + (1 - a) * rho                        [Alg.1 line 6, balance factor]
        LFT_i^D = LFT_i + (D - T_min)                    [re-anchor exit LFT: cmax -> D]
        dl_i  = EFT_i + DB * (LFT_i^D - EFT_i)           [Alg.1 lines 16-17, convex combo]
        st_i  = EST_i

    The tunable constant ``a`` in [0, 1] is passed via ``db`` (config
    ``reproductions.*.db``). Larger slack (rho -> 1) => DB -> 1 => sub-deadlines
    hug the latest-feasible finish; tighter (rho -> 0) => DB -> a. Shared by
    HFWPF preprocessing.
    """
    a = db  # config `db` plays the role of the paper's tunable constant a in [0,1]
    tasks = instance.workflow.tasks
    cmax = max((t.eft for t in tasks.values()), default=0.0)  # T_min
    D = instance.deadline
    if D <= 1e-12 or cmax <= 1e-12:
        return ({i: t.est for i, t in tasks.items()},
                {i: t.eft for i, t in tasks.items()})
    rho = min(1.0, max(0.0, (D - cmax) / D))
    DB = a + (1.0 - a) * rho
    shift = D - cmax  # re-anchor structural LFT (exit=cmax) to the deadline (exit=D)
    st: dict[int, float] = {}
    dl: dict[int, float] = {}
    for i, t in tasks.items():
        lft_d = t.lft + shift
        dl[i] = t.eft + DB * (lft_d - t.eft)
        st[i] = t.est
    return st, dl


class DynamicSubDeadlines:
    """Alg.1 constraint distribution with the balance factor re-evaluated online.

    ``sub_deadlines`` freezes ``rho`` at the structural makespan, so every task
    is independently granted the *whole* remaining slack ``LFT_i - EFT_i``. On a
    150-task medium instance that hands out ~38 time units per task while the
    global slack is only ~11, which makes the per-task test vacuous: a list
    scheduler can satisfy all sub-deadlines and still overshoot ``D`` by 2x,
    because slack consumed on a machine timeline is never debited.

    Alg.1 line 6 calls DB a *dynamic* balance factor, so we recompute it from the
    slack that is still unspent: with ``L`` the largest lateness observed so far
    against the structural EFT,

        slack_left = max(0, (D - T_min) - L)
        rho        = clip(slack_left / D, 0, 1)
        DB         = a + (1 - a) * rho
        dl_i       = EFT_i + DB * ((LFT_i + slack_left) - EFT_i)

    As the schedule falls behind, ``slack_left -> 0`` and sub-deadlines tighten
    onto ``EFT_i + a*(LFT_i - EFT_i)``, which is what pushes Alg.2 out of
    cost-minimising mode into earliest-finish catch-up.
    """

    def __init__(self, instance: Instance, db: float) -> None:
        self._a = db
        self._tasks = instance.workflow.tasks
        self._D = instance.deadline
        self._cmax = max((t.eft for t in self._tasks.values()), default=0.0)
        self._slack0 = max(0.0, self._D - self._cmax)
        self._lateness = 0.0
        self._degenerate = self._D <= 1e-12 or self._cmax <= 1e-12

    def observe(self, i: int, finish: float) -> None:
        """Debit the slack spent by placing task ``i`` with finish time ``finish``."""
        t = self._tasks.get(i)
        if t is not None:
            self._lateness = max(self._lateness, finish - t.eft)

    def of(self, i: int) -> float:
        t = self._tasks.get(i)
        if t is None:
            return 0.0
        if self._degenerate:
            return t.eft
        slack_left = max(0.0, self._slack0 - self._lateness)
        rho = min(1.0, slack_left / self._D)
        DB = self._a + (1.0 - self._a) * rho
        return t.eft + DB * ((t.lft + slack_left) - t.eft)


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
