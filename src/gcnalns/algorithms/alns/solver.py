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
        self.w_pen = a["recovery_penalty_weight"]
        self.w_dead = a["deadline_penalty_weight"]
        self.ablation = ablation or {}
        self.rng = np.random.default_rng(cfg.get("seeds.alns", 2026) + seed)
        self._gcn_score = None  # set by GCN-ALNS subclass

    # -- objective ----------------------------------------------------------
    def _quick_eval(self, instance, sol, factory):
        """Fast objective for the search loop (no full Omega enumeration)."""
        sched = compute_schedule(instance, sol, factory, self.cfg)
        r_eff = effective_budget(instance, sched.cmax)
        est_r = estimate_rmax(instance, sol, sched, factory, self.cfg)
        bd = compute_cost(instance, sol, sched, factory, self.cfg)
        obj = bd.total + self.w_dead * max(0.0, sched.cmax - instance.deadline)
        if self.ablation.get("recovery_penalty", True):
            obj += self.w_pen * max(0.0, est_r - r_eff)
        est_feasible = (sched.cmax <= instance.deadline + 1e-9) and (est_r <= r_eff + 1e-9)
        return obj, bd.total, sched, est_r, r_eff, est_feasible

    def _full_eval(self, instance, sol, factory):
        sched = compute_schedule(instance, sol, factory, self.cfg)
        res = verify(instance, sol, factory, self.cfg, sched, ablation=self.ablation)
        bd = compute_cost(instance, sol, sched, factory, self.cfg,
                          recovery_cost=res.recovery_cost)
        return res, bd.total

    def _gcn_hook(self, instance, factory):
        return None  # overridden by GCN-ALNS

    # -- main loop ----------------------------------------------------------
    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        factory = get_factory(instance, self.cfg)

        # initial solution: Criticality-Based Hybrid
        cur = CriticalityHybrid(self.cfg).schedule(instance)
        gcn_score = self._gcn_hook(instance, factory)

        obj, cost, sched, est_r, r_eff, est_feas = self._quick_eval(instance, cur, factory)
        cur_obj = obj
        cur_cbase = sched.cmax
        best = cur.clone()
        best_obj = obj
        best_feasible = None
        best_feasible_cost = math.inf
        # confirm the initial solution if it looks feasible
        if est_feas:
            res0, cost0 = self._full_eval(instance, cur, factory)
            if res0.feasible:
                best_feasible, best_feasible_cost = cur.clone(), cost0

        d_names = list(DESTROY_OPS)
        r_names = list(REPAIR_OPS)
        dw = {n: 1.0 for n in d_names}
        rw = {n: 1.0 for n in r_names}
        dscore = {n: 0.0 for n in d_names}
        rscore = {n: 0.0 for n in r_names}
        dcount = {n: 0 for n in d_names}
        rcount = {n: 0 for n in r_names}

        T = self.T0
        no_improve = 0
        total_eval = 0
        total_prune = 0
        conv_iter = 0

        for it in range(self.max_iter):
            d_name = self._roulette(d_names, dw)
            r_name = self._roulette(r_names, rw)

            budget = max(effective_budget(instance, min(cur_cbase, instance.deadline)), 0.0)
            repairer = Repairer(instance, factory, self.cfg, budget, gcn_score,
                                ablation=self.ablation)

            cand = cur.clone()
            removed = DESTROY_OPS[d_name](instance, cand, self.cfg, self.rng)
            REPAIR_OPS[r_name](instance, cand, removed, repairer, self.m_max)
            total_eval += repairer.n_eval
            total_prune += repairer.n_pruned

            cobj, ccost, csched, cest_r, creff, cest_feas = self._quick_eval(instance, cand, factory)

            dcount[d_name] += 1
            rcount[r_name] += 1
            outcome = 0.0
            accept = False
            if cobj < cur_obj - 1e-9:
                accept = True
                outcome = self.scores["better"]
            elif math.exp(-(cobj - cur_obj) / max(T, 1e-9)) > self.rng.random():
                accept = True
                outcome = self.scores["accepted"]

            if accept:
                cur, cur_obj, cur_cbase = cand, cobj, csched.cmax

            if cobj < best_obj - 1e-9:
                best, best_obj = cand.clone(), cobj
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
                break

        out = best_feasible if best_feasible is not None else best
        self.runtime = time.perf_counter() - t0
        self.convergence_iter = conv_iter
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
