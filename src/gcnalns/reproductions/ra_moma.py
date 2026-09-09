"""RA-MOMA -- Reliability-Aware Multi-Objective Memetic Algorithm.

Reference:
    S. Qin, D. Pi, Z. Shao, Y. Xu, Y. Chen, "Reliability-Aware Multi-Objective
    Memetic Algorithm for Workflow Scheduling Problem in Multi-Cloud System,"
    IEEE Transactions on Parallel and Distributed Systems, 34(4):1343-1361, 2023.
    DOI: 10.1109/TPDS.2023.3245089 (CCF-A).

RA-MOMA solves MOWSP-MCS: multi-objective (makespan, cost, reliability) DAG
scheduling where each task may get **one backup copy** (primary + 1 replica) to
raise reliability. It is a memetic algorithm = NSGA-II global search + a hybrid
HEFT initialization + a problem-specific local search of four neighborhood
operators over the elite non-dominated archive.

Static adaptation to this framework (see reproductions/README.md):
* Encoding <alpha, beta, gamma, pi> -> only **alpha (primary node)** and
  **beta (backup node, none allowed)** are kept: our continuum is a *fixed* node
  set (gamma resource-pool/leasing dropped), and our evaluator derives the node
  execution order topologically (pi dropped -- it has no effect on our schedule).
* backup (primary + 1 copy on a distinct resource) -> **active backup, m=2**,
  requiring distinct fault domains (framework isolation rule); else the task
  stays passive.
* Objectives (makespan, cost, reliability) -> (c_base, compute_cost total,
  recovery-risk = estimate_rmax); minimising recovery-risk is our analogue of
  maximising reliability. NSGA-II evolves the (c_base, cost, risk) Pareto set;
  from the final archive we return the **min-cost solution that our estimate
  deems SLA-feasible** (the "user picks a schedule" step, here driven by the
  deadline + recovery budget). The incumbent is finally judged by the exact
  ``core.recovery.verify`` gate, like every method.
* Poisson/CTMC reliability, hourly/minute multi-cloud billing, and performance
  degradation are replaced by the framework's deterministic exec + cost + verify.
"""
from __future__ import annotations

import time

import numpy as np

from ..algorithms.base import Scheduler, topo_node_order
from ..algorithms.heft import _greedy_eft_place
from ..core.cost import compute_cost
from ..core.dag import compute_structural
from ..core.model import Instance, Solution
from ..core.recovery import effective_budget, estimate_rmax
from ..core.schedule_eval import compute_schedule, exec_time
from ..data.links import get_factory

NONE = -1  # beta gene value meaning "no backup"


class RAMOMA(Scheduler):
    name = "RA-MOMA"

    def __init__(self, cfg, seed: int = 0):
        super().__init__(cfg, seed)
        g = cfg["reproductions"]["ra_moma"]
        self.pop_size = int(g["pop_size"])
        self.generations = int(g["generations"])
        self.cr = float(g["crossover_rate"])
        self.mr = float(g["mutation_rate"])
        self.ls = int(g["intensify_size"])
        self.tour_k = int(g["tournament_k"])
        self.backup_init_prob = float(g["backup_init_prob"])
        self.max_candidates = int(g["max_candidates"])
        self.m_max = int(cfg.get("alns.m_max", 2))
        self.w_dead = float(cfg.get("alns.deadline_penalty_weight", 200.0))
        self.w_pen = float(cfg.get("alns.recovery_penalty_weight", 100.0))
        self.rng = np.random.default_rng(int(cfg.get("seeds.alns", 2026)) + seed)

    # ------------------------------------------------------------------ decode
    def _decode(self, prim: np.ndarray, back: np.ndarray) -> Solution:
        nodes = self.nodes
        sol = Solution()
        for idx, i in enumerate(self.tasks):
            kp = self.node_ids[int(prim[idx])]
            repl = [kp]
            active = False
            b = int(back[idx])
            if b != NONE:
                kb = self.node_ids[b]
                if kb != kp and nodes[kb].fault_domain != nodes[kp].fault_domain:
                    repl.append(kb)
                    active = True
            sol.placement[i] = repl
            sol.mode[i] = active
        return sol

    def _objectives(self, sol: Solution):
        """Return (obj_vector=(makespan,cost,risk), feasible, violation, cost)."""
        factory = self.factory
        sched = compute_schedule(self.instance, sol, factory, self.cfg)
        r_eff = effective_budget(self.instance, sched.cmax)
        est_r = estimate_rmax(self.instance, sol, sched, factory, self.cfg)
        bd = compute_cost(self.instance, sol, sched, factory, self.cfg)
        dead_viol = max(0.0, sched.cmax - self.instance.deadline)
        rec_viol = max(0.0, est_r - r_eff)
        feasible = dead_viol <= 1e-9 and rec_viol <= 1e-9
        obj = (sched.cmax, bd.total, est_r)  # makespan, cost, risk (all minimised)
        return obj, feasible, dead_viol + rec_viol, bd.total

    # -------------------------------------------------------------- GA driver
    def schedule(self, instance: Instance) -> Solution:
        t0 = time.perf_counter()
        cfg = self.cfg
        self.instance = instance
        self.factory = get_factory(instance, cfg)
        compute_structural(instance.workflow, instance.continuum, cfg, self.factory)

        self.nodes = instance.continuum.nodes
        self.tasks = list(instance.workflow.tasks.keys())
        self.node_ids = list(self.nodes.keys())
        self.node_index = {k: idx for idx, k in enumerate(self.node_ids)}
        n = len(self.tasks)

        # per-task candidate node indices (cap on large instances)
        if len(self.node_ids) <= self.max_candidates:
            base_cand = list(range(len(self.node_ids)))
            self.cand = {i: base_cand for i in self.tasks}
        else:
            self.cand = {}
            for i in self.tasks:
                pick = self.rng.choice(len(self.node_ids), size=self.max_candidates,
                                       replace=False)
                self.cand[i] = [int(x) for x in pick]

        pop = self._init_population(n)
        evals = [self._eval(ind) for ind in pop]      # (obj, feas, viol, cost)

        best, best_cost, best_feas, best_gen = self._pick_incumbent(pop, evals, 0)

        for gen in range(1, self.generations + 1):
            offspring: list = []
            n_pairs = max(1, self.pop_size // 2)
            for _ in range(n_pairs):
                pa = self._tournament(pop, evals)
                pb = self._tournament(pop, evals)
                ca, cb = self._crossover(pa, pb)
                self._mutate(ca)
                self._mutate(cb)
                offspring.append(ca)
                offspring.append(cb)
            # intensification on the current non-dominated archive
            archive = self._nondominated(pop, evals)
            elites = self._intensify(archive, self.ls)
            combined = pop + offspring + elites
            combined_ev = evals + [self._eval(x) for x in offspring] \
                + [self._eval(x) for x in elites]
            pop, evals = self._environmental_select(combined, combined_ev)
            best, best_cost, best_feas, best_gen = self._pick_incumbent(
                pop, evals, gen, best, best_cost, best_feas, best_gen)

        sol = self._decode(best[0], best[1])
        sol.order = topo_node_order(instance, sol)
        sol.meta["found_feasible"] = best_feas
        self.convergence_iter = best_gen
        self.runtime = time.perf_counter() - t0
        return sol

    def _eval(self, ind):
        obj, feas, viol, cost = self._objectives(self._decode(ind[0], ind[1]))
        return obj, feas, viol, cost

    def _pick_incumbent(self, pop, evals, gen, best=None, best_cost=float("inf"),
                        best_feas=False, best_gen=0):
        """Track the min-cost estimate-feasible individual (else min violation)."""
        for ind, (obj, feas, viol, cost) in zip(pop, evals):
            better = False
            if feas and not best_feas:
                better = True
            elif feas == best_feas:
                if best_feas:
                    better = cost < best_cost - 1e-9
                else:  # neither feasible -> prefer lower violation then cost
                    better = viol < best_cost - 1e-9
            if better:
                best = (ind[0].copy(), ind[1].copy())
                best_cost = cost if feas else viol
                best_feas = feas
                best_gen = gen
        return best, best_cost, best_feas, best_gen

    # ------------------------------------------------------ initialization
    def _init_population(self, n) -> list:
        pop: list = []
        # s1: HEFT (min makespan), no backup
        order = sorted(self.instance.workflow.tasks,
                       key=lambda t: self.instance.workflow.tasks[t].ranku, reverse=True)
        heft = _greedy_eft_place(self.instance, self.factory, order)
        prim_heft = self._sol_to_prim(heft)
        pop.append((prim_heft.copy(), np.full(n, NONE, dtype=np.int64)))
        # s2: HEFT + all backup
        pop.append((prim_heft.copy(), self._all_backup(prim_heft)))
        # s3: cheapest placement, no backup
        prim_cheap = self._cheapest_prim()
        pop.append((prim_cheap.copy(), np.full(n, NONE, dtype=np.int64)))
        # s4: cheapest + all backup
        pop.append((prim_cheap.copy(), self._all_backup(prim_cheap)))
        # remainder: random
        while len(pop) < self.pop_size:
            prim = np.array([self.rng.choice(self.cand[i]) for i in self.tasks],
                            dtype=np.int64)
            back = self._random_back(prim)
            pop.append((prim, back))
        return pop

    def _sol_to_prim(self, sol: Solution) -> np.ndarray:
        return np.array([self.node_index[sol.placement[i][0]] for i in self.tasks],
                        dtype=np.int64)

    def _cheapest_prim(self) -> np.ndarray:
        """Assign each task to the node with minimum cost_coef * exec_time."""
        nodes = self.nodes
        prim = np.empty(len(self.tasks), dtype=np.int64)
        for idx, i in enumerate(self.tasks):
            best_k, best_c = self.cand[i][0], float("inf")
            for ci in self.cand[i]:
                k = self.node_ids[ci]
                c = nodes[k].cost_coef * exec_time(self.instance, i, k)
                if c < best_c:
                    best_c, best_k = c, ci
            prim[idx] = best_k
        return prim

    def _all_backup(self, prim: np.ndarray) -> np.ndarray:
        """Backup every task on a distinct-fault-domain candidate node."""
        back = np.full(len(self.tasks), NONE, dtype=np.int64)
        for idx, i in enumerate(self.tasks):
            back[idx] = self._distinct_domain_node(i, int(prim[idx]))
        return back

    def _random_back(self, prim: np.ndarray) -> np.ndarray:
        back = np.full(len(self.tasks), NONE, dtype=np.int64)
        for idx, i in enumerate(self.tasks):
            if self.rng.random() < self.backup_init_prob:
                back[idx] = self._distinct_domain_node(i, int(prim[idx]))
        return back

    def _distinct_domain_node(self, i, prim_ci) -> int:
        prim_fd = self.nodes[self.node_ids[prim_ci]].fault_domain
        cands = [ci for ci in self.cand[i]
                 if self.nodes[self.node_ids[ci]].fault_domain != prim_fd]
        return int(self.rng.choice(cands)) if cands else NONE

    # -------------------------------------------------------- GA operators
    def _tournament(self, pop, evals):
        pick = self.rng.integers(0, len(pop), size=self.tour_k)
        best = pick[0]
        for p in pick[1:]:
            if self._rank_key(evals[p]) < self._rank_key(evals[best]):
                best = p
        return pop[best]

    @staticmethod
    def _rank_key(ev):
        # feasibility-first, then violation, then cost (scalar proxy for tournament)
        obj, feas, viol, cost = ev
        return (0 if feas else 1, viol, cost)

    def _crossover(self, pa, pb):
        na, ba = pa[0].copy(), pa[1].copy()
        nb, bb = pb[0].copy(), pb[1].copy()
        mask = self.rng.random(len(na)) < self.cr
        na[mask], nb[mask] = nb[mask], na[mask].copy()
        ba[mask], bb[mask] = bb[mask], ba[mask].copy()
        return (na, ba), (nb, bb)

    def _mutate(self, ind) -> None:
        prim, back = ind
        for idx, i in enumerate(self.tasks):
            if self.rng.random() < self.mr:
                prim[idx] = self.rng.choice(self.cand[i])
            if self.rng.random() < self.mr:
                if back[idx] == NONE:
                    back[idx] = self._distinct_domain_node(i, int(prim[idx]))
                else:
                    back[idx] = NONE

    # --------------------------------------------- intensification (local)
    def _intensify(self, archive: list, ls: int) -> list:
        if not archive:
            return []
        out: list = []
        for _ in range(ls):
            src = archive[int(self.rng.integers(0, len(archive)))]
            op = int(self.rng.integers(0, 4))
            child = (src[0].copy(), src[1].copy())
            if op == 0:
                self._cp_perturb(child)
            elif op == 1:
                self._cp_remove(child)
            elif op == 2:
                self._vm_release(child)
            else:
                self._vm_backup(child)
            out.append(child)
        return out

    def _critical_tasks(self, child) -> list[int]:
        sol = self._decode(child[0], child[1])
        sched = compute_schedule(self.instance, sol, self.factory, self.cfg)
        crit = [i for i in self.tasks if sched.sched_slack.get(i, 0.0) <= 1e-6]
        return crit or list(self.tasks)

    def _low_util_nodes(self, child) -> list[int]:
        used = {k: 0.0 for k in self.nodes}
        for idx, i in enumerate(self.tasks):
            used[self.node_ids[int(child[0][idx])]] += \
                self.instance.workflow.tasks[i].resource
        hosting = {k: used[k] / max(self.nodes[k].cpu_cap, 1e-9)
                   for k in self.nodes if used[k] > 0}
        if not hosting:
            return []
        mean_u = float(np.mean(list(hosting.values())))
        return [k for k, u in hosting.items() if u <= mean_u]

    def _cp_perturb(self, child) -> None:
        crit = self._critical_tasks(child)
        i = crit[int(self.rng.integers(0, len(crit)))]
        idx = self.tasks.index(i)
        cur_layer = self.nodes[self.node_ids[int(child[0][idx])]].layer
        alt = [ci for ci in self.cand[i]
               if self.nodes[self.node_ids[ci]].layer != cur_layer]
        child[0][idx] = int(self.rng.choice(alt)) if alt else \
            int(self.rng.choice(self.cand[i]))

    def _cp_remove(self, child) -> None:
        for i in self._critical_tasks(child):
            idx = self.tasks.index(i)
            child[1][idx] = NONE
            child[0][idx] = int(self.rng.choice(self.cand[i]))

    def _vm_release(self, child) -> None:
        low = self._low_util_nodes(child)
        if not low:
            return
        k = low[int(self.rng.integers(0, len(low)))]
        target = int(self.rng.integers(0, len(self.node_ids)))
        for idx in range(len(self.tasks)):
            if self.node_ids[int(child[0][idx])] == k:
                child[0][idx] = target

    def _vm_backup(self, child) -> None:
        low = self._low_util_nodes(child)
        no_backup = [idx for idx in range(len(self.tasks)) if child[1][idx] == NONE]
        if not no_backup:
            return
        idx = no_backup[int(self.rng.integers(0, len(no_backup)))]
        i = self.tasks[idx]
        prim_fd = self.nodes[self.node_ids[int(child[0][idx])]].fault_domain
        cands = [self.node_index[k] for k in low
                 if self.nodes[k].fault_domain != prim_fd] or \
                [ci for ci in self.cand[i]
                 if self.nodes[self.node_ids[ci]].fault_domain != prim_fd]
        if cands:
            child[1][idx] = int(self.rng.choice(cands))

    # ------------------------------------------------- NSGA-II selection
    @staticmethod
    def _dominates(a, b) -> bool:
        oa, ob = a[0], b[0]
        le = all(x <= y for x, y in zip(oa, ob))
        lt = any(x < y for x, y in zip(oa, ob))
        return le and lt

    def _nondominated(self, pop, evals) -> list:
        front = []
        for i, ev_i in enumerate(evals):
            if not any(self._dominates(evals[j], ev_i)
                       for j in range(len(evals)) if j != i):
                front.append((pop[i][0].copy(), pop[i][1].copy()))
        return front

    def _environmental_select(self, pop, evals):
        fronts = self._fast_nondominated_sort(evals)
        sel_idx: list[int] = []
        for front in fronts:
            if len(sel_idx) + len(front) <= self.pop_size:
                sel_idx.extend(front)
            else:
                cd = self._crowding(front, evals)
                order = sorted(front, key=lambda i: cd[i], reverse=True)
                sel_idx.extend(order[: self.pop_size - len(sel_idx)])
                break
        new_pop = [(pop[i][0].copy(), pop[i][1].copy()) for i in sel_idx]
        new_ev = [evals[i] for i in sel_idx]
        return new_pop, new_ev

    def _fast_nondominated_sort(self, evals) -> list[list[int]]:
        N = len(evals)
        S = [[] for _ in range(N)]
        n_dom = [0] * N
        fronts: list[list[int]] = [[]]
        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                if self._dominates(evals[p], evals[q]):
                    S[p].append(q)
                elif self._dominates(evals[q], evals[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt: list[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    @staticmethod
    def _crowding(front: list[int], evals) -> dict[int, float]:
        cd = {i: 0.0 for i in front}
        if len(front) <= 2:
            return {i: float("inf") for i in front}
        for m in range(3):  # three objectives
            order = sorted(front, key=lambda i: evals[i][0][m])
            cd[order[0]] = cd[order[-1]] = float("inf")
            lo, hi = evals[order[0]][0][m], evals[order[-1]][0][m]
            span = hi - lo
            if span < 1e-12:
                continue
            for r in range(1, len(order) - 1):
                cd[order[r]] += (evals[order[r + 1]][0][m]
                                 - evals[order[r - 1]][0][m]) / span
        return cd
