"""HFWPF / Hybrid-FT -- Hybrid Fault-Tolerant Workflow scheduling (static core).

Reference:
    Q. Ren, G. Yao, "A Hybrid Fault-Tolerant Workflow Scheduling With
    Performance Fluctuated Cloud Resources," IEEE Transactions on Services
    Computing, 19(1):44-57, 2026. DOI: 10.1109/TSC.2025.3635177 (CCF-B).

The paper has four stages (preprocessing / initial scheduling / online
scheduling / online adjustment) with performance-fluctuating, elastic VMs. Only
the **static core** is reproduced (see reproductions/README.md):

* Preprocessing -> per-task sub-deadlines (shared with RECR-D).
* Initial scheduling, FT-strategy choice (eq.8): a task with
  ``slack = dl - st >= 2*(Tboot + max_pred(TT) + psi)`` uses **resubmission
  (passive, single instance)**; otherwise **replication (active, 2 instances on
  distinct nodes)**. ``psi`` is the minimum deterministic execution time,
  ``max_pred(TT)`` the largest predecessor transfer time. Execution-time
  *uncertainty* -- the paper's core -- is removed, so the slack test uses
  deterministic times; ``Tboot`` defaults to 0 (no VM boot in this framework).
* VM selection: replication tasks bias toward small-capacity nodes (save cost),
  resubmission tasks toward large-capacity nodes (keep headroom), on top of the
  earliest-finish-time placement. Active replicas are fault-domain-isolated via
  ``add_active_replicas``.

The online stages (fault re-execution, slack back-fill, replication->
resubmission down-switch) map onto our one-shot recovery verifier and are not
re-implemented.
"""
from __future__ import annotations

import time

import numpy as np

from ..algorithms.base import Scheduler, topo_node_order
from ..algorithms.heft import _greedy_eft_place
from ..algorithms.replicas import add_active_replicas
from ..core.dag import compute_structural
from ..core.model import Instance, Solution
from ..data.links import get_factory
from ._shared import avg_link, sub_deadlines


class HFWPF(Scheduler):
    name = "HFWPF"

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        cfg = self.cfg
        p = cfg["reproductions"]["hfwpf"]
        factory = get_factory(instance, cfg)
        compute_structural(instance.workflow, instance.continuum, cfg, factory)

        st, dl = sub_deadlines(instance, float(p["db"]))
        t_boot = float(p["t_boot"])
        active = self._choose_modes(instance, factory, st, dl, t_boot)

        # Capacity-biased earliest-finish-time placement (eq. VM selection).
        nodes = instance.continuum.nodes
        cap = np.array([nodes[k].cpu_cap for k in nodes], dtype=float)
        lo, hi = float(cap.min()), float(cap.max())
        cap_norm = {k: ((nodes[k].cpu_cap - lo) / (hi - lo) if hi - lo > 1e-12 else 0.0)
                    for k in nodes}
        bias = float(p["cap_bias"])

        def extra(i, k, eft):
            # active (replication): prefer small capacity  -> +bias*cap
            # passive (resubmission): prefer large capacity -> -bias*cap
            sign = 1.0 if i in active else -1.0
            return sign * bias * eft * cap_norm[k]

        order = sorted(instance.workflow.tasks,
                       key=lambda t: instance.workflow.tasks[t].ranku, reverse=True)
        sol = _greedy_eft_place(instance, factory, order, extra_metric=extra)

        # Promote replication tasks to fault-domain-isolated active backup.
        m_max = int(cfg.get("alns.m_max", 2))
        if active:
            add_active_replicas(instance, sol, active, m_max, cfg)
        sol.order = topo_node_order(instance, sol)
        self.runtime = time.perf_counter() - t0
        return sol

    def _choose_modes(self, instance: Instance, factory, st: dict[int, float],
                      dl: dict[int, float], t_boot: float) -> set[int]:
        """Eq.8 slack test -> set of tasks that must use active replication."""
        wf = instance.workflow
        nodes = instance.continuum.nodes
        avg_lat, avg_bw = avg_link(instance, factory)
        min_compute = max(n.compute for n in nodes.values())  # fastest node

        active: set[int] = set()
        for i, task in wf.tasks.items():
            psi = task.workload / min_compute  # minimum deterministic exec time
            max_tt = 0.0
            for pr in wf.predecessors(i):
                d = wf.data.get((pr, i), 0.0)
                max_tt = max(max_tt, avg_lat + d / max(avg_bw, 1e-9))
            slack = dl.get(i, 0.0) - st.get(i, 0.0)
            threshold = 2.0 * (t_boot + max_tt + psi)
            if slack < threshold:  # insufficient slack for recompute -> replicate
                active.add(i)
        return active
