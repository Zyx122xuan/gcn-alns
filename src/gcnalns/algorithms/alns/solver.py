"""Adaptive Large Neighborhood Search solver (Spec sec.15).

Loop: destroy -> repair -> full recovery verification -> simulated-annealing
acceptance. Operators are selected by adaptive roulette weights. Only solutions
passing full verification update the feasible incumbent; the augmented
objective (cost + recovery/deadline penalties) guides the search toward the
SLA-feasible region. GCN guidance is injected via a ``gcn_score`` hook
(``None`` for plain ALNS).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ...config import Config
from ...core.cost import compute_cost
from ...core.model import Instance, Solution
from ...core.recovery import effective_budget, estimate_rmax, verify
from ...core.schedule_eval import compute_schedule
from ...data.links import get_factory
from ..base import Scheduler
from ..baselines import CriticalityHybrid
from .insertion import Repairer
from .operators import DESTROY_OPS, REPAIR_OPS


class ALNS(Scheduler):
    name = "ALNS"
    use_gcn = False

    def __init__(self, cfg: Config, seed: int = 0, ablation: dict | None = None):
        super().__init__(cfg, seed)
        a = cfg["alns"]
        self.max_iter = a["max_iter"]
        self.m_max = a["m_max"]
        self.T0 = a["initial_temperature"]
        self.cooling = a["cooling_rate"]
        self.seg = a["segment_length"]
        self.react = a["reaction_factor"]
        self.scores = a["scores"]
        self.no_improve_stop = a["no_improve_stop"]
        self.restarts = int(a.get("restarts", 0))
        self.w_pen = a["recovery_penalty_weight"]
        self.w_dead = a["deadline_penalty_weight"]
        self.ablation = ablation or {}
        # B2 restart diversification can be ablated off (falls back to plain
        # break-on-stagnation).
        if not self.ablation.get("restarts", True):
            self.restarts = 0
        self.rng = np.random.default_rng(cfg.get("seeds.alns", 2026) + seed)
        self._gcn_score = None  # set by GCN-ALNS subclass

    # -- objective ----------------------------------------------------------
    def _quick_eval(self, instance, sol, factory):
        """Fast objective for the search loop (no full Omega enumeration).

        Returns ``(obj, cost, sched, est_r, r_eff, est_feasible, viol)`` where
        ``obj`` is the cost + penalty objective and ``viol`` is the raw
        constraint-violation magnitude V (deadline + recovery) used by the
        feasibility-first search phase.
        """
        sched = compute_schedule(instance, sol, factory, self.cfg)
        r_eff = effective_budget(instance, sched.cmax)
        est_r = estimate_rmax(instance, sol, sched, factory, self.cfg)
        bd = compute_cost(instance, sol, sched, factory, self.cfg)
        dead_viol = max(0.0, sched.cmax - instance.deadline)
        rec_viol = max(0.0, est_r - r_eff)
        obj = bd.total + self.w_dead * dead_viol
        if self.ablation.get("recovery_penalty", True):
            obj += self.w_pen * rec_viol
        # V reflects the *definition* of feasibility (both deadline & recovery),
        # regardless of whether the recovery penalty is included in the cost obj.
        viol = dead_viol + rec_viol
        est_feasible = (dead_viol <= 1e-9) and (est_r <= r_eff + 1e-9)
        return obj, bd.total, sched, est_r, r_eff, est_feasible, viol

    def _search_objective(self, obj, viol, cost, have_feasible):
        """Feasibility-first (B1): before any feasible incumbent is found,
        minimise the violation V (cost only breaks ties); once feasible, switch
        to the cost + penalty objective. Ablating ``feasibility_first`` off
        always optimises the cost + penalty objective (plain-ALNS behaviour)."""
        if have_feasible or not self.ablation.get("feasibility_first", True):
            return obj
        return viol + 1e-4 * cost

    def _full_eval(self, instance, sol, factory):
        sched = compute_schedule(instance, sol, factory, self.cfg)
        res = verify(instance, sol, factory, self.cfg, sched, ablation=self.ablation)
        bd = compute_cost(instance, sol, sched, factory, self.cfg,
                          recovery_cost=res.recovery_cost)
        return res, bd.total

    def _gcn_hook(self, instance, factory):
        return None  # overridden by GCN-ALNS

    def _gcn_mode_hook(self, instance, factory):
        return None  # overridden by GCN-ALNS (learned FT-mode head)

    # -- initial / restart solutions ----------------------------------------
    def _candidate_inits(self, instance, factory, gcn_score, gcn_mode):
        """Candidate warm-start solutions. Base ALNS offers the criticality
        hybrid plus a makespan-minimising HEFT-fastest solution (B3), which
        helps deadline-bound (tight) scenarios."""
        inits = [CriticalityHybrid(self.cfg).schedule(instance)]
        if self.ablation.get("init_diversify", True):
            try:
                from ..heft import HEFT
                inits.append(HEFT(self.cfg).schedule(instance))
            except Exception:
                pass
        return inits

    def _initial_solution(self, instance, factory, gcn_score, gcn_mode):
        """Pick the best warm-start among the candidates: prefer estimated
        feasible, then lower violation V, then lower objective."""
        best_sol, best_key = None, None
        for sol in self._candidate_inits(instance, factory, gcn_score, gcn_mode):
            if sol is None:
                continue
            obj, cost, sched, est_r, r_eff, est_feas, viol = self._quick_eval(
                instance, sol, factory)
            key = (0 if est_feas else 1, viol, obj)
            if best_key is None or key < best_key:
                best_key, best_sol = key, sol
        return best_sol

    def _restart_solution(self, instance, factory, base, gcn_score, gcn_mode):
        """Diversify on stagnation (B2): a large-kick destroy of the current
        best, repaired with the GCN-guided insertion operator."""
        cand = base.clone()
        n = instance.workflow.num_tasks
        q = max(1, int(round(0.5 * n)))
        tasks = list(cand.placement.keys())
        if tasks:
            removed = [int(x) for x in self.rng.choice(
                tasks, size=min(q, len(tasks)), replace=False)]
        else:
            removed = list(instance.workflow.tasks.keys())
        for i in removed:
            cand.placement.pop(i, None)
            cand.mode.pop(i, None)
        sched = compute_schedule(instance, base, factory, self.cfg)
        budget = max(effective_budget(instance, min(sched.cmax, instance.deadline)), 0.0)
        repairer = Repairer(instance, factory, self.cfg, budget, gcn_score,
                            ablation=self.ablation, gcn_mode=gcn_mode)
        from .operators import repair_insert
        repair_insert(instance, cand, removed, repairer, self.m_max)
        return cand

    # -- main loop ----------------------------------------------------------
    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)

        # GCN hooks first so they can inform the warm-start (A1) and destroy (A3).
        gcn_score = self._gcn_hook(instance, factory)
        gcn_mode = self._gcn_mode_hook(instance, factory)

        cur = self._initial_solution(instance, factory, gcn_score, gcn_mode)
        obj, cost, sched, est_r, r_eff, est_feas, viol = self._quick_eval(
            instance, cur, factory)
        cur_pen, cur_cost, cur_viol, cur_cbase = obj, cost, viol, sched.cmax
        best = cur.clone()
        best_pen, best_cost, best_viol = obj, cost, viol

        best_feasible = None
        best_feasible_cost = math.inf
        first_feas_iter = -1  # iteration at which feasibility was first reached
        # confirm the initial solution if it looks feasible
        if est_feas:
            res0, cost0 = self._full_eval(instance, cur, factory)
            if res0.feasible:
                best_feasible, best_feasible_cost = cur.clone(), cost0
                first_feas_iter = 0

        # GCN-guided destroy is only offered when a score hook + switch are on
        # (keeps plain ALNS behaviour identical for backward compatibility).
        d_names = [n for n in DESTROY_OPS if n != "low_gcn"]
        if gcn_score is not None and self.ablation.get("gcn_destroy", True):
            d_names.append("low_gcn")
        r_names = list(REPAIR_OPS)
        dw = {n: 1.0 for n in d_names}
        rw = {n: 1.0 for n in r_names}
        dscore = {n: 0.0 for n in d_names}
        rscore = {n: 0.0 for n in r_names}
        dcount = {n: 0 for n in d_names}
        rcount = {n: 0 for n in r_names}

        T = self.T0
        no_improve = 0
        restarts_left = self.restarts
        total_eval = 0
        total_prune = 0
        conv_iter = 0

        for it in range(self.max_iter):
            have_feasible = best_feasible is not None
            cur_val = self._search_objective(cur_pen, cur_viol, cur_cost, have_feasible)
            best_val = self._search_objective(best_pen, best_viol, best_cost, have_feasible)

            d_name = self._roulette(d_names, dw)
            r_name = self._roulette(r_names, rw)

            budget = max(effective_budget(instance, min(cur_cbase, instance.deadline)), 0.0)
            repairer = Repairer(instance, factory, self.cfg, budget, gcn_score,
                                ablation=self.ablation, gcn_mode=gcn_mode)

            cand = cur.clone()
            removed = DESTROY_OPS[d_name](instance, cand, self.cfg, self.rng, gcn_score)
            REPAIR_OPS[r_name](instance, cand, removed, repairer, self.m_max)
            total_eval += repairer.n_eval
            total_prune += repairer.n_pruned

            cobj, ccost, csched, cest_r, creff, cest_feas, cviol = self._quick_eval(
                instance, cand, factory)
            cand_val = self._search_objective(cobj, cviol, ccost, have_feasible)

            dcount[d_name] += 1
            rcount[r_name] += 1
            outcome = 0.0
            accept = False
            if cand_val < cur_val - 1e-9:
                accept = True
                outcome = self.scores["better"]
            elif math.exp(-(cand_val - cur_val) / max(T, 1e-9)) > self.rng.random():
                accept = True
                outcome = self.scores["accepted"]

            if accept:
                cur = cand
                cur_pen, cur_cost, cur_viol, cur_cbase = cobj, ccost, cviol, csched.cmax

            if cand_val < best_val - 1e-9:
                best = cand.clone()
                best_pen, best_cost, best_viol = cobj, ccost, cviol
                outcome = self.scores["best"]
                no_improve = 0
            else:
                no_improve += 1

            # exact-verify only promising candidates (cheap estimate feasible &
            # cheaper than incumbent) -> confirm feasible incumbent
            if cest_feas and ccost < best_feasible_cost - 1e-9:
                cres, cost_exact = self._full_eval(instance, cand, factory)
                if cres.feasible:
                    best_feasible = cand.clone()
                    best_feasible_cost = cost_exact
                    conv_iter = it
                    if first_feas_iter < 0:
                        first_feas_iter = it

            dscore[d_name] += outcome
            rscore[r_name] += outcome

            if (it + 1) % self.seg == 0:
                self._update_weights(dw, dscore, dcount)
                self._update_weights(rw, rscore, rcount)
                dscore = {n: 0.0 for n in d_names}
                rscore = {n: 0.0 for n in r_names}
                dcount = {n: 0 for n in d_names}
                rcount = {n: 0 for n in r_names}

            T *= self.cooling
            if no_improve >= self.no_improve_stop:
                if restarts_left > 0:
                    # B2: diversifying restart instead of terminating.
                    restarts_left -= 1
                    base_kick = best_feasible if best_feasible is not None else best
                    cur = self._restart_solution(instance, factory, base_kick,
                                                 gcn_score, gcn_mode)
                    o, c, s, er, rf, ef, v = self._quick_eval(instance, cur, factory)
                    cur_pen, cur_cost, cur_viol, cur_cbase = o, c, v, s.cmax
                    T = self.T0
                    no_improve = 0
                    if ef and c < best_feasible_cost - 1e-9:
                        r2, ce = self._full_eval(instance, cur, factory)
                        if r2.feasible:
                            best_feasible = cur.clone()
                            best_feasible_cost = ce
                            conv_iter = it
                            if first_feas_iter < 0:
                                first_feas_iter = it
                else:
                    break

        out = best_feasible if best_feasible is not None else best
        self.runtime = time.perf_counter() - t0
        self.convergence_iter = conv_iter
        self.first_feasible_iter = first_feas_iter
        self.prune_ratio = total_prune / total_eval if total_eval else 0.0
        out.meta["found_feasible"] = best_feasible is not None
        return out

    # -- adaptive helpers ---------------------------------------------------
    def _roulette(self, names, weights):
        w = np.array([weights[n] for n in names], dtype=float)
        w = w / w.sum()
        return names[int(self.rng.choice(len(names), p=w))]

    def _update_weights(self, weights, score, count):
        for n in weights:
            if count[n] > 0:
                weights[n] = (1 - self.react) * weights[n] + self.react * (score[n] / count[n])
            weights[n] = max(weights[n], 1e-3)
