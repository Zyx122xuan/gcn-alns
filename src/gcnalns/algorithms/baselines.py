"""Baseline schedulers (Spec sec.5.3.1, sec.14).

All produce a :class:`Solution`; SLA feasibility is judged later by the shared
recovery verifier. Differences lie only in placement and fault-tolerance mode.

* NoFT                : HEFT placement, all passive, recovery ignored at schedule time.
* ReactiveOnly        : all passive, placement mildly reliability-aware.
* FullProactive       : all tasks active backup (m=2), fault-domain isolated.
* CriticalPathBackup  : active backup only for near-critical-path tasks.
* CriticalityHybrid   : active backup for high-criticality / low-slack tasks.
* ReliabilityAware    : passive, high-criticality tasks biased to reliable nodes.
"""
from __future__ import annotations

import time

import numpy as np

from ..core.dag import compute_structural
from ..core.model import Instance, Solution
from ..data.links import get_factory
from .base import Scheduler, topo_node_order
from .heft import _greedy_eft_place
from .replicas import add_active_replicas


def _failure_norm(instance: Instance) -> dict[int, float]:
    fr = np.array([n.failure_rate for n in instance.continuum.nodes.values()], dtype=float)
    lo, hi = float(fr.min()), float(fr.max())
    keys = list(instance.continuum.nodes.keys())
    if hi - lo < 1e-18:
        return {k: 0.0 for k in keys}
    return {k: (instance.continuum.nodes[k].failure_rate - lo) / (hi - lo) for k in keys}


def _ranku_order(instance: Instance) -> list[int]:
    return sorted(instance.workflow.tasks,
                  key=lambda t: instance.workflow.tasks[t].ranku, reverse=True)


class _PassiveBase(Scheduler):
    rel_weight: float = 0.0
    crit_weight: float = 0.0

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)
        compute_structural(instance.workflow, instance.continuum, self.cfg, factory)
        fnorm = _failure_norm(instance)

        def extra(i, k, eft):
            h = instance.workflow.tasks[i].criticality
            return (self.rel_weight + self.crit_weight * h) * eft * fnorm[k]

        sol = _greedy_eft_place(instance, factory, _ranku_order(instance),
                                extra_metric=extra if (self.rel_weight or self.crit_weight) else None)
        for i in instance.workflow.tasks:
            sol.mode[i] = False
        self.runtime = time.perf_counter() - t0
        return sol


class NoFT(_PassiveBase):
    name = "No-FT"
    rel_weight = 0.0


class ReactiveOnly(_PassiveBase):
    name = "Reactive-Only"
    rel_weight = 0.3


class ReliabilityAware(_PassiveBase):
    name = "Reliability-Aware"
    rel_weight = 0.2
    crit_weight = 1.5


class FullProactive(Scheduler):
    name = "Full-Proactive"

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)
        compute_structural(instance.workflow, instance.continuum, self.cfg, factory)
        sol = _greedy_eft_place(instance, factory, _ranku_order(instance))
        m = self.cfg.get("alns.m_max", 2)
        add_active_replicas(instance, sol, set(instance.workflow.tasks), m, self.cfg)
        sol.order = topo_node_order(instance, sol)
        self.runtime = time.perf_counter() - t0
        return sol


class CriticalPathBackup(Scheduler):
    name = "Critical-Path Backup"

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)
        compute_structural(instance.workflow, instance.continuum, self.cfg, factory)
        sol = _greedy_eft_place(instance, factory, _ranku_order(instance))
        cmax = max(t.eft for t in instance.workflow.tasks.values())
        frac = self.cfg.get("baselines.critical_path_slack_frac", 0.05)
        thr = frac * cmax
        cp = {i for i, t in instance.workflow.tasks.items() if t.slack <= thr}
        m = self.cfg.get("alns.m_max", 2)
        add_active_replicas(instance, sol, cp, m, self.cfg)
        sol.order = topo_node_order(instance, sol)
        self.runtime = time.perf_counter() - t0
        return sol


class CriticalityHybrid(Scheduler):
    name = "Criticality-Based Hybrid"

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)
        compute_structural(instance.workflow, instance.continuum, self.cfg, factory)
        sol = _greedy_eft_place(instance, factory, _ranku_order(instance))
        h = np.array([t.criticality for t in instance.workflow.tasks.values()])
        q = self.cfg.get("baselines.hybrid_criticality_quantile", 0.70)
        thr = float(np.quantile(h, q)) if len(h) else 1.0
        backup = {i for i, t in instance.workflow.tasks.items() if t.criticality >= thr}
        m = self.cfg.get("alns.m_max", 2)
        add_active_replicas(instance, sol, backup, m, self.cfg)
        sol.order = topo_node_order(instance, sol)
        self.runtime = time.perf_counter() - t0
        return sol
