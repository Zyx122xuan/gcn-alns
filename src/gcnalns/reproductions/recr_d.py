"""RECR-D -- Reliability/Energy/Cost-aware Resource selection with Deadline.

Reference:
    A. Taghinezhad-Niar, J. Taheri, "Fault-Tolerant Cost-Efficient Scheduling
    for Energy and Deadline-Constrained IoT Workflows in Edge-Cloud Continuum,"
    IEEE Transactions on Services Computing, 18(5):2892-2903, 2025.
    DOI: 10.1109/TSC.2025.3599497 (CCF-B).

Static adaptation (see reproductions/README.md):
* Alg.1 constraint distribution -> per-task sub-deadlines via a dynamic-balance
  convex combination of EFT/LFT (energy allocation dropped, alpha_energy = 0).
* Alg.2 main body -> tasks ordered by EDF (earliest sub-deadline first); each
  task is placed on the resource minimising the normalised penalty
  ``PN = zeta*cost + beta*deadline + gamma*reliability`` (eq.43), energy omitted.
* Alg.3 TRM -> a task whose placement violates its sub-deadline / reliability
  threshold, or whose reactive recovery exceeds the effective budget under our
  ``estimate_rmax`` gate, is promoted to active backup (fault-domain-isolated
  replicas via ``add_active_replicas``). The CTMC success-probability judgement
  is replaced by our recovery verifier's budget test.
"""
from __future__ import annotations

import math
import time

from ..algorithms.base import Scheduler, topo_node_order
from ..algorithms.replicas import add_active_replicas
from ..core.dag import compute_structural, topological_order
from ..core.model import Instance, Solution
from ..core.recovery import effective_budget, estimate_rmax
from ..core.schedule_eval import compute_schedule, exec_time
from ..data.links import get_factory
from ._shared import DynamicSubDeadlines, passive_recovery_risk


class RECRD(Scheduler):
    name = "RECR-D"

    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        cfg = self.cfg
        p = cfg["reproductions"]["recr_d"]
        factory = get_factory(instance, cfg)
        compute_structural(instance.workflow, instance.continuum, cfg, factory)

        theta = float(p["theta_reliability"])
        sol, flagged = self._penalty_place(instance, factory, theta)

        # TRM stage 1: tasks flagged during placement (deadline / reliability).
        m_max = int(cfg.get("alns.m_max", 2))
        if flagged:
            add_active_replicas(instance, sol, flagged, m_max, cfg)

        # TRM stage 2: recovery-budget-driven promotion under estimate_rmax.
        self._trm_recovery(instance, sol, factory, m_max)

        sol.order = topo_node_order(instance, sol)
        self.runtime = time.perf_counter() - t0
        return sol

    # -- Alg.2 normalised-penalty placement ---------------------------------
    def _penalty_place(self, instance: Instance, factory, theta: float
                       ) -> tuple[Solution, set[int]]:
        p = self.cfg["reproductions"]["recr_d"]
        zeta = float(p["zeta_cost"])
        beta = float(p["beta_deadline"])
        gamma = float(p["gamma_reliability"])

        wf = instance.workflow
        nodes = instance.continuum.nodes
        pred = {t: wf.predecessors(t) for t in wf.tasks}
        # ``compute_schedule`` replays every node's queue in this canonical
        # topological order and a node timeline is append-only (no gap filling),
        # so planning in any other order (e.g. EDF) makes the EFT estimates
        # below untruthful: on a 150-task instance Alg.2 plans a makespan of 103
        # that re-serialises to 262. Since Alg.2 concentrates load on cheap
        # resources, the mismatch is far more damaging than for a spread-out
        # EFT baseline, so we plan in the evaluator's order. The sub-deadlines
        # still drive Alg.2 through the resource-selection gate below.
        order = topological_order(wf)
        # Alg.1 line 6: the balance factor is dynamic, so sub-deadlines are
        # re-derived per placement from the slack that is still unspent.
        dyn = DynamicSubDeadlines(instance, float(p["db"]))

        sol = Solution()
        node_free = {k: 0.0 for k in nodes}
        used_cpu = {k: 0.0 for k in nodes}
        used_mem = {k: 0.0 for k in nodes}
        finish: dict[int, float] = {}
        where: dict[int, int] = {}
        flagged: set[int] = set()

        for i in order:
            task = wf.tasks[i]
            cands: list[tuple[int, float, float, float, float]] = []  # k, eft, est, cost, R
            for k, node in nodes.items():
                if used_cpu[k] + task.resource > node.res_cpu + 1e-9:
                    continue
                if used_mem[k] + task.mem > node.res_mem + 1e-9:
                    continue
                est = node_free[k]
                for pr in pred[i]:
                    d = wf.data.get((pr, i), 0.0)
                    est = max(est, finish[pr] + factory.comm_delay(where[pr], k, d))
                p_ik = exec_time(instance, i, k)
                eft = est + p_ik
                cost = node.cost_coef * p_ik
                rel = math.exp(-node.failure_rate * p_ik)
                cands.append((k, eft, est, cost, rel))

            if not cands:  # capacity exhausted -> earliest-free fallback
                best_k = min(nodes, key=lambda k: node_free[k])
                node = nodes[best_k]
                est = node_free[best_k]
                for pr in pred[i]:
                    d = wf.data.get((pr, i), 0.0)
                    est = max(est, finish[pr] + factory.comm_delay(where[pr], best_k, d))
                best_eft = est + exec_time(instance, i, best_k)
                best_rel = math.exp(-node.failure_rate * exec_time(instance, i, best_k))
                sol.meta["capacity_overflow"] = True
            else:
                cost_max = max(c[3] for c in cands)
                dl_i = dyn.of(i)
                # RECR-D is deadline-CONSTRAINED: prefer the min-penalty resource
                # among those meeting the task sub-deadline; only if none can, fall
                # back to the earliest-finish node (best-effort catch-up) so cost
                # minimisation never silently breaks the global deadline.
                feasible = [c for c in cands
                            if dl_i <= 1e-12 or c[1] <= dl_i + 1e-9]
                pool = feasible if feasible else cands
                best_k, best_pn, best_eft, best_rel = None, float("inf"), 0.0, 1.0
                for (k, eft, est, cost, rel) in pool:
                    dl_norm = max(0.0, (eft - dl_i) / dl_i) if dl_i > 1e-12 else 0.0
                    cost_norm = cost / (cost_max + 1.0)
                    rel_norm = max(0.0, theta - rel)
                    pn = zeta * cost_norm + beta * dl_norm + gamma * rel_norm
                    if not feasible:  # catch-up mode: earliest finish dominates
                        pn = eft
                    if pn < best_pn:
                        best_pn, best_k, best_eft, best_rel = pn, k, eft, rel

            sol.placement[i] = [best_k]
            sol.mode[i] = False
            node_free[best_k] = best_eft
            used_cpu[best_k] += task.resource
            used_mem[best_k] += task.mem
            finish[i] = best_eft
            where[i] = best_k
            # Alg.3 trigger: sub-deadline missed or reliability below Theta.
            dl_i = dyn.of(i)
            dyn.observe(i, best_eft)
            if (dl_i > 1e-12 and best_eft > dl_i + 1e-9) or best_rel < theta:
                flagged.add(i)

        return sol, flagged

    # -- Alg.3 TRM: recovery-budget-driven active backup --------------------
    def _trm_recovery(self, instance: Instance, sol: Solution, factory,
                      m_max: int) -> None:
        p = self.cfg["reproductions"]["recr_d"]
        max_rounds = int(p["trm_max_rounds"])
        batch_frac = float(p["trm_batch_frac"])
        for _ in range(max_rounds):
            sched = compute_schedule(instance, sol, factory, self.cfg)
            r_eff = effective_budget(instance, sched.cmax)
            est_r = estimate_rmax(instance, sol, sched, factory, self.cfg)
            deadline_ok = sched.cmax <= instance.deadline + 1e-9
            if deadline_ok and est_r <= r_eff + 1e-9:
                return  # estimated-feasible; final judgement by verify()
            risks = passive_recovery_risk(instance, sol, sched, factory, self.cfg)
            ranked = [i for i, r in sorted(risks.items(), key=lambda kv: kv[1],
                                           reverse=True) if r > 0.0]
            if not ranked:
                return
            batch = max(1, int(math.ceil(batch_frac * len(ranked))))
            promote = set(ranked[:batch])
            before = sum(1 for i in instance.workflow.tasks if sol.is_active(i))
            add_active_replicas(instance, sol, promote, m_max, self.cfg)
            after = sum(1 for i in instance.workflow.tasks if sol.is_active(i))
            if after == before:  # nothing could be isolated -> stop
                return
