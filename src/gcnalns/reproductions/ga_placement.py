"""GA-Placement -- a standard genetic algorithm placement baseline.

Stand-in for:
    M. Xu, Y. Mei, S. Zhu, B. Zhang, T. Xiang, F. Zhang, M. Zhang, "Genetic
    Programming for Dynamic Workflow Scheduling in Fog Computing," IEEE
    Transactions on Services Computing, 16(4):2657-2671, 2023.
    DOI: 10.1109/TSC.2023.3249160 (CCF-B). Code: github.com/MengXu95/Fog-Computing

MTGP's original problem (dynamic arrivals, makespan minimisation, no
fault-tolerance / SLA feasibility) is not directly comparable to our static
fault-tolerant placement. Following the handoff, we instead provide a standard
GA as an *evolutionary-metaheuristic reference* against ALNS / GCN-ALNS:

* Individual: per task a (node choice, active/passive bit).
* Decode: primary node from the gene; active tasks get fault-domain-isolated
  replicas via ``add_active_replicas``.
* Fitness: our ``compute_cost`` total with SLA feasibility as a hard constraint,
  using the fast ``estimate_rmax`` recovery gate + deadline/recovery penalties
  (same quick objective ALNS uses inside its loop). The returned incumbent is
  finally judged by the exact ``core.recovery.verify`` gate, like every method.
* Operators: tournament selection, uniform crossover, per-gene mutation,
  elitism. Optionally seeded with a greedy-EFT individual for a fair warm start.
"""
from __future__ import annotations

import time

import numpy as np

from ..algorithms.base import Scheduler, topo_node_order
from ..algorithms.heft import _greedy_eft_place
from ..algorithms.replicas import add_active_replicas
from ..core.cost import compute_cost
from ..core.dag import compute_structural
from ..core.model import Instance, Solution
from ..core.recovery import effective_budget, estimate_rmax
from ..core.schedule_eval import compute_schedule
from ..data.links import get_factory


class GAPlacement(Scheduler):
    name = "GA-Placement"

    def __init__(self, cfg, seed: int = 0):
        super().__init__(cfg, seed)
        g = cfg["reproductions"]["ga_placement"]
        self.pop_size = int(g["pop_size"])
        self.generations = int(g["generations"])
        self.cx_rate = float(g["crossover_rate"])
        self.mut_rate = float(g["mutation_rate"])
        self.tour_k = int(g["tournament_k"])
        self.elitism = int(g["elitism"])
        self.seed_greedy = bool(g["seed_greedy"])
        self.max_candidates = int(g["max_candidates"])
        self.m_max = int(cfg.get("alns.m_max", 2))
        self.w_dead = float(cfg.get("alns.deadline_penalty_weight", 200.0))
        self.w_pen = float(cfg.get("alns.recovery_penalty_weight", 100.0))
        self.rng = np.random.default_rng(int(cfg.get("seeds.alns", 2026)) + seed)

    # -- decode + evaluate --------------------------------------------------
    def _decode(self, instance: Instance, node_gene: np.ndarray,
                mode_gene: np.ndarray) -> Solution:
        sol = Solution()
        active: set[int] = set()
        for idx, i in enumerate(self.tasks):
            cand = self.candidates[i]
            k = cand[int(node_gene[idx]) % len(cand)]
            sol.placement[i] = [k]
            sol.mode[i] = False
            if mode_gene[idx]:
                active.add(i)
        if active:
            add_active_replicas(instance, sol, active, self.m_max, self.cfg)
        return sol

    def _fitness(self, instance: Instance, factory, sol: Solution
                 ) -> tuple[float, bool]:
        sched = compute_schedule(instance, sol, factory, self.cfg)
        r_eff = effective_budget(instance, sched.cmax)
        est_r = estimate_rmax(instance, sol, sched, factory, self.cfg)
        bd = compute_cost(instance, sol, sched, factory, self.cfg)
        dead_viol = max(0.0, sched.cmax - instance.deadline)
        rec_viol = max(0.0, est_r - r_eff)
        obj = bd.total + self.w_dead * dead_viol + self.w_pen * rec_viol
        feasible = dead_viol <= 1e-9 and rec_viol <= 1e-9
        return obj, feasible

    # -- GA driver ----------------------------------------------------------
    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        cfg = self.cfg
        factory = get_factory(instance, cfg)
        compute_structural(instance.workflow, instance.continuum, cfg, factory)

        self.tasks = list(instance.workflow.tasks.keys())
        n = len(self.tasks)
        all_nodes = list(instance.continuum.nodes.keys())

        # Per-task candidate node lists (cap size on large instances for speed).
        self.candidates: dict[int, list[int]] = {}
        if len(all_nodes) <= self.max_candidates:
            for i in self.tasks:
                self.candidates[i] = all_nodes
        else:
            for i in self.tasks:
                pick = self.rng.choice(len(all_nodes), size=self.max_candidates,
                                       replace=False)
                self.candidates[i] = [all_nodes[j] for j in pick]

        pop = self._init_population(instance, factory, n)
        fits = [self._fitness(instance, factory, self._decode(instance, ng, mg))
                for (ng, mg) in pop]
        objs = np.array([f[0] for f in fits])

        best_idx = int(np.argmin(objs))
        best = pop[best_idx]
        best_obj = float(objs[best_idx])
        best_feas = fits[best_idx][1]
        best_gen = 0

        for gen in range(1, self.generations + 1):
            new_pop = self._elites(pop, objs)
            while len(new_pop) < self.pop_size:
                pa = self._tournament(pop, objs)
                pb = self._tournament(pop, objs)
                ca, cb = self._crossover(pa, pb)
                self._mutate(ca)
                self._mutate(cb)
                new_pop.append(ca)
                if len(new_pop) < self.pop_size:
                    new_pop.append(cb)
            pop = new_pop
            fits = [self._fitness(instance, factory, self._decode(instance, ng, mg))
                    for (ng, mg) in pop]
            objs = np.array([f[0] for f in fits])
            gi = int(np.argmin(objs))
            # Prefer feasible incumbents; among same feasibility, lower obj.
            cand_feas = fits[gi][1]
            if (cand_feas and not best_feas) or \
               (cand_feas == best_feas and objs[gi] < best_obj - 1e-9):
                best, best_obj, best_feas, best_gen = pop[gi], float(objs[gi]), cand_feas, gen

        sol = self._decode(instance, best[0], best[1])
        sol.order = topo_node_order(instance, sol)
        sol.meta["found_feasible"] = best_feas
        self.convergence_iter = best_gen
        self.runtime = time.perf_counter() - t0
        return sol

    # -- population helpers -------------------------------------------------
    def _init_population(self, instance, factory, n
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
        pop: list[tuple[np.ndarray, np.ndarray]] = []
        if self.seed_greedy:
            order = sorted(instance.workflow.tasks,
                           key=lambda t: instance.workflow.tasks[t].ranku, reverse=True)
            greedy = _greedy_eft_place(instance, factory, order)
            ng = np.zeros(n, dtype=np.int64)
            for idx, i in enumerate(self.tasks):
                k = greedy.placement[i][0]
                cand = self.candidates[i]
                ng[idx] = cand.index(k) if k in cand else 0
            mg = np.zeros(n, dtype=bool)
            pop.append((ng, mg))
        while len(pop) < self.pop_size:
            ng = np.array([self.rng.integers(len(self.candidates[i]))
                           for i in self.tasks], dtype=np.int64)
            mg = self.rng.random(n) < 0.3
            pop.append((ng, mg))
        return pop

    def _elites(self, pop, objs) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.elitism <= 0:
            return []
        idx = np.argsort(objs)[: self.elitism]
        return [(pop[i][0].copy(), pop[i][1].copy()) for i in idx]

    def _tournament(self, pop, objs) -> tuple[np.ndarray, np.ndarray]:
        pick = self.rng.integers(0, len(pop), size=self.tour_k)
        best = pick[int(np.argmin(objs[pick]))]
        return pop[best]

    def _crossover(self, pa, pb):
        na, ma = pa[0].copy(), pa[1].copy()
        nb, mb = pb[0].copy(), pb[1].copy()
        if self.rng.random() < self.cx_rate:
            mask = self.rng.random(len(na)) < 0.5
            na[mask], nb[mask] = nb[mask], na[mask].copy()
            ma[mask], mb[mask] = mb[mask], ma[mask].copy()
        return (na, ma), (nb, mb)

    def _mutate(self, ind) -> None:
        ng, mg = ind
        for idx, i in enumerate(self.tasks):
            if self.rng.random() < self.mut_rate:
                ng[idx] = self.rng.integers(len(self.candidates[i]))
            if self.rng.random() < self.mut_rate:
                mg[idx] = not mg[idx]
