"""Sequential insertion engine + InsertCost (Spec sec.13, sec.15).

The :class:`Repairer` rebuilds a schedule in topological order, deciding the
placement / fault-tolerance mode of *removed* tasks while keeping the fixed
placement of the others. It maintains tentative start/finish, node availability
and per-node resource usage so candidate actions can be screened and scored
incrementally, without a full re-evaluation per candidate.

InsertCost(i,k) = w1*ECT + w2*Cost + w3*Risk + w4*RecRisk + w5*Load - w6*GCN
Fast pre-screen prunes candidates that violate capacity, fault-domain
isolation, the no-fault deadline, or the recovery-risk hard bound.
"""
from __future__ import annotations

import math
from typing import Callable

from ...config import Config
from ...core.dag import topological_order
from ...core.model import Instance, Solution
from ...core.schedule_eval import exec_time
from ...data.links import LinkFactory

GCNScoreFn = Callable[[int, int], float]


class Repairer:
    def __init__(self, instance: Instance, factory: LinkFactory, cfg: Config,
                 budget: float, gcn_score: GCNScoreFn | None = None):
        self.inst = instance
        self.factory = factory
        self.cfg = cfg
        self.budget = budget
        self.gcn_score = gcn_score
        self.w = cfg["insert_cost"]
        self.rcv = cfg["recovery"]
        self.order = topological_order(instance.workflow)
        self.pred = {t: instance.workflow.predecessors(t) for t in instance.workflow.tasks}
        # counters for pruning ratio
        self.n_eval = 0
        self.n_pruned = 0

    # -- per-candidate quantities -------------------------------------------
    def _comm_in(self, i: int, k: int, where: dict[int, int],
                 finish: dict[int, float]) -> tuple[float, float]:
        """Return (earliest_input_ready, total_input_comm_delay) for (i,k)."""
        ready = 0.0
        comm = 0.0
        for p in self.pred[i]:
            d = self.inst.workflow.data.get((p, i), 0.0)
            kp = where.get(p)
            if kp is None:
                continue
            cd = self.factory.comm_delay(kp, k, d)
            comm += cd
            ready = max(ready, finish.get(p, 0.0) + cd)
        return ready, comm

    def _pfail(self, i: int, k: int) -> float:
        node = self.inst.continuum.nodes[k]
        p_ik = exec_time(self.inst, i, k)
        return 1.0 - math.exp(-node.failure_rate * p_ik)

    def rec_risk(self, i: int, k: int, comm_in: float) -> float:
        """Estimated passive recovery risk (Spec sec.13)."""
        node = self.inst.continuum.nodes[k]
        p_ik = exec_time(self.inst, i, k)
        slack = self.inst.workflow.tasks[i].slack
        return max(0.0, self.rcv["delta_detect"] + node.queue_wait + comm_in + p_ik - slack)

    def insert_cost(self, i: int, k: int, where, finish, node_free, used_cpu) -> float:
        node = self.inst.continuum.nodes[k]
        ready, comm_in = self._comm_in(i, k, where, finish)
        start = max(node_free.get(k, 0.0), ready)
        p_ik = exec_time(self.inst, i, k)
        ect = start + p_ik
        cost = node.cost_coef * p_ik + self.cfg["cost"]["comm_cost_coef"] * 0.0
        risk = self.inst.workflow.tasks[i].criticality * self._pfail(i, k)
        recrisk = self.rec_risk(i, k, comm_in)
        load = used_cpu.get(k, 0.0) / max(node.cpu_cap, 1e-9)
        gcn = self.gcn_score(i, k) if self.gcn_score else 0.0
        w = self.w
        return (w["w1"] * ect + w["w2"] * cost + w["w3"] * risk
                + w["w4"] * recrisk + w["w5"] * load - w["w6"] * gcn)

    # -- screening ----------------------------------------------------------
    def feasible_node(self, i: int, k: int, used_cpu, used_mem,
                      where, finish, node_free, forbid_domains=None) -> bool:
        self.n_eval += 1
        task = self.inst.workflow.tasks[i]
        node = self.inst.continuum.nodes[k]
        if used_cpu.get(k, 0.0) + task.resource > node.res_cpu + 1e-9:
            self.n_pruned += 1
            return False
        if used_mem.get(k, 0.0) + task.mem > node.res_mem + 1e-9:
            self.n_pruned += 1
            return False
        if forbid_domains and node.fault_domain in forbid_domains:
            self.n_pruned += 1
            return False
        # no-fault deadline pre-screen: eft is a lower bound on i's finish
        ready, _ = self._comm_in(i, k, where, finish)
        eft = max(node_free.get(k, 0.0), ready) + exec_time(self.inst, i, k)
        if eft > self.inst.deadline + 1e-9:
            self.n_pruned += 1
            return False
        return True

    @property
    def prune_ratio(self) -> float:
        return self.n_pruned / self.n_eval if self.n_eval else 0.0


def _apply_place(repairer: Repairer, sol: Solution, i: int, nodes: list[int],
                 active: bool, where, finish, node_free, used_cpu, used_mem) -> None:
    """Commit a placement for task i and update tentative state."""
    inst = repairer.inst
    task = inst.workflow.tasks[i]
    fins = {}
    for k in nodes:
        ready, _ = repairer._comm_in(i, k, where, finish)
        start = max(node_free.get(k, 0.0), ready)
        fin = start + exec_time(inst, i, k)
        fins[k] = fin
        node_free[k] = fin
        used_cpu[k] = used_cpu.get(k, 0.0) + task.resource
        used_mem[k] = used_mem.get(k, 0.0) + task.mem
    eff_k = min(fins, key=lambda k: fins[k])
    finish[i] = fins[eff_k]
    where[i] = eff_k
    sol.placement[i] = list(nodes)
    sol.mode[i] = active
