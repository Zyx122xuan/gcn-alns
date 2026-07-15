"""ALNS destroy and repair operators (Spec sec.15).

Destroy: Random / Criticality / High-Risk / Congested-Node / Over-Backup removal.
Repair:  GCN-guided (InsertCost) / Min-Recovery-Delay / Min-Cost /
         Active-Backup-Enhancement / Passive-Downgrade.

Repair operators rebuild a complete solution from a partial one in topological
order, deciding placement + fault-tolerance mode of removed tasks via the
:class:`Repairer` engine, with fast pre-screening.
"""
from __future__ import annotations

import math

import numpy as np

from ...core.dag import topological_order
from ...core.model import Instance, Solution
from ...core.schedule_eval import exec_time
from .insertion import Repairer, _apply_place


# ---------------------------------------------------------------------------
# Destroy operators -> return list of removed task ids (and strip them from sol)
# ---------------------------------------------------------------------------
def _q(instance: Instance, cfg, rng) -> int:
    n = instance.workflow.num_tasks
    hi = max(1, int(math.ceil(cfg.get("alns.max_removal_frac", 0.2) * n)))
    return int(rng.integers(1, hi + 1))


def _strip(sol: Solution, removed: list[int]) -> None:
    for i in removed:
        sol.placement.pop(i, None)
        sol.mode.pop(i, None)


def random_removal(instance, sol, cfg, rng, gcn_score=None):
    q = _q(instance, cfg, rng)
    tasks = list(sol.placement.keys())
    removed = list(rng.choice(tasks, size=min(q, len(tasks)), replace=False))
    removed = [int(x) for x in removed]
    _strip(sol, removed)
    return removed


def criticality_removal(instance, sol, cfg, rng, gcn_score=None):
    q = _q(instance, cfg, rng)
    tasks = sorted(sol.placement.keys(),
                   key=lambda t: instance.workflow.tasks[t].criticality, reverse=True)
    removed = tasks[:q]
    _strip(sol, removed)
    return removed


def high_risk_removal(instance, sol, cfg, rng, gcn_score=None):
    q = _q(instance, cfg, rng)

    def risk(i):
        k = sol.placement[i][0]
        node = instance.continuum.nodes[k]
        p = exec_time(instance, i, k)
        pf = 1.0 - math.exp(-node.failure_rate * p)
        return instance.workflow.tasks[i].criticality * pf
    tasks = sorted(sol.placement.keys(), key=risk, reverse=True)
    removed = tasks[:q]
    _strip(sol, removed)
    return removed


def congested_node_removal(instance, sol, cfg, rng, gcn_score=None):
    q = _q(instance, cfg, rng)
    load = {k: 0 for k in instance.continuum.nodes}
    for i, repl in sol.placement.items():
        for k in repl:
            load[k] += 1
    hot = sorted(load, key=lambda k: load[k], reverse=True)
    removed: list[int] = []
    for k in hot:
        for i, repl in list(sol.placement.items()):
            if k in repl and i not in removed:
                removed.append(i)
            if len(removed) >= q:
                break
        if len(removed) >= q:
            break
    _strip(sol, removed)
    return removed


def over_backup_removal(instance, sol, cfg, rng, gcn_score=None):
    q = _q(instance, cfg, rng)
    actives = [i for i in sol.placement if sol.mode.get(i)]
    actives.sort(key=lambda t: instance.workflow.tasks[t].criticality)  # low crit first
    removed = actives[:q]
    _strip(sol, removed)
    return removed


def low_gcn_removal(instance, sol, cfg, rng, gcn_score=None):
    """GCN-guided destroy (A3): remove tasks whose *current* host node the model
    scores lowest, so repair can relocate them toward high-preference nodes.

    Falls back to random removal when no GCN hook is available (plain ALNS).
    """
    if gcn_score is None:
        return random_removal(instance, sol, cfg, rng)
    q = _q(instance, cfg, rng)

    def cur_score(i):
        repl = sol.placement[i]
        # a task's dislike = 1 - mean GCN score over its current hosting nodes
        return sum(gcn_score(i, k) for k in repl) / max(len(repl), 1)
    tasks = sorted(sol.placement.keys(), key=cur_score)  # lowest score first
    removed = tasks[:q]
    _strip(sol, removed)
    return removed


DESTROY_OPS = {
    "random": random_removal,
    "criticality": criticality_removal,
    "high_risk": high_risk_removal,
    "congested": congested_node_removal,
    "over_backup": over_backup_removal,
    "low_gcn": low_gcn_removal,
}


# ---------------------------------------------------------------------------
# Repair core
# ---------------------------------------------------------------------------
def _choose_active_replicas(repairer: Repairer, sol, i, primary_k, where, finish,
                            node_free, used_cpu, used_mem, m_max) -> list[int]:
    """Pick up to m_max isolated nodes (incl. primary) for an active task.

    With ``ablation["fault_domain"] is False`` the fault-domain isolation
    constraint is dropped, so replicas may co-locate in the same domain (and a
    node fault can wipe out several at once -> higher violation rate).
    """
    inst = repairer.inst
    nodes = inst.continuum.nodes
    isolate = repairer.ablation.get("fault_domain", True)
    # A2: bias replica placement toward learned high-preference nodes when the
    # gcn_mode switch is on and a GCN score hook is available.
    gscore = repairer.gcn_score if repairer.ablation.get("gcn_mode", True) else None
    chosen = [primary_k]
    domains = {nodes[primary_k].fault_domain}
    while len(chosen) < m_max:
        cands = [k for k in nodes
                 if k not in chosen
                 and (not isolate or nodes[k].fault_domain not in domains)
                 and repairer.feasible_node(i, k, used_cpu, used_mem, where, finish, node_free)]
        if not cands:
            break
        k = max(cands, key=lambda k: (gscore(i, k) if gscore else 0.0,
                                      nodes[k].compute / (1.0 + used_cpu.get(k, 0.0)),
                                      -nodes[k].failure_rate))
        chosen.append(k)
        domains.add(nodes[k].fault_domain)
    return chosen


def _repair_core(instance: Instance, partial: Solution, removed: list[int],
                 repairer: Repairer, score_fn, m_max: int) -> Solution:
    sol = partial  # mutate in place (already stripped)
    order = topological_order(instance.workflow)
    removed_set = set(removed)

    where: dict[int, int] = {}
    finish: dict[int, float] = {}
    node_free: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}
    used_cpu: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}
    used_mem: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}

    # seed resource usage from kept tasks so capacity screening is accurate
    for i, repl in sol.placement.items():
        if i in removed_set:
            continue
        for k in repl:
            used_cpu[k] += instance.workflow.tasks[i].resource
            used_mem[k] += instance.workflow.tasks[i].mem

    for i in order:
        if i not in removed_set:
            repl = sol.placement.get(i)
            if not repl:
                continue
            # de-count then re-apply to refresh tentative timing
            for k in repl:
                used_cpu[k] -= instance.workflow.tasks[i].resource
                used_mem[k] -= instance.workflow.tasks[i].mem
            _apply_place(repairer, sol, i, repl, sol.mode.get(i, False),
                         where, finish, node_free, used_cpu, used_mem)
            continue

        # choose placement for a removed task
        cands = [k for k in instance.continuum.nodes
                 if repairer.feasible_node(i, k, used_cpu, used_mem, where, finish, node_free)]
        if not cands:
            cands = [min(node_free, key=lambda k: node_free[k])]
        scores = score_fn(repairer, i, cands, where, finish, node_free, used_cpu)
        best_k = min(cands, key=lambda k: scores[k])

        # mode decision: passive unless estimated recovery risk exceeds budget.
        # ablation["hybrid"] forces a single fault-tolerance mode:
        #   False + "all-active"  -> always active backup
        #   False + "all-passive" -> always passive recompute (overrides budget)
        _, comm_in = repairer._comm_in(i, best_k, where, finish)
        recrisk = repairer.rec_risk(i, best_k, comm_in)
        hybrid = repairer.ablation.get("hybrid", True)
        if hybrid is False:
            active = repairer.ablation.get("hybrid_mode", "all-active") == "all-active"
        else:
            active = recrisk > repairer.budget
            # A2: let the learned FT-mode head request an active backup where it
            # predicts one is needed (only if a mode hook is present + switch on).
            gmode = repairer.gcn_mode if repairer.ablation.get("gcn_mode", True) else None
            if gmode is not None and gmode(i, best_k) >= 0.5:
                active = True
        nodes = [best_k]
        if active:
            nodes = _choose_active_replicas(repairer, sol, i, best_k, where, finish,
                                            node_free, used_cpu, used_mem, m_max)
            active = len(nodes) >= 2
        _apply_place(repairer, sol, i, nodes, active,
                     where, finish, node_free, used_cpu, used_mem)

    from ..base import topo_node_order
    sol.order = topo_node_order(instance, sol)
    return sol


# -- score functions (return {candidate_node: score}; lower = preferred) -----
def _score_insert(r, i, cands, where, finish, node_free, used_cpu):
    return r.insert_scores(i, cands, where, finish, node_free, used_cpu)


def _score_recdelay(r, i, cands, where, finish, node_free, used_cpu):
    out = {}
    for k in cands:
        _, comm_in = r._comm_in(i, k, where, finish)
        out[k] = r.rec_risk(i, k, comm_in)
    return out


def _score_cost(r, i, cands, where, finish, node_free, used_cpu):
    return {k: r.inst.continuum.nodes[k].cost_coef * exec_time(r.inst, i, k)
            for k in cands}


# -- repair operators ---------------------------------------------------------
def repair_insert(instance, sol, removed, repairer, m_max):
    return _repair_core(instance, sol, removed, repairer, _score_insert, m_max)


def repair_min_recdelay(instance, sol, removed, repairer, m_max):
    return _repair_core(instance, sol, removed, repairer, _score_recdelay, m_max)


def repair_min_cost(instance, sol, removed, repairer, m_max):
    return _repair_core(instance, sol, removed, repairer, _score_cost, m_max)


def repair_active_enhance(instance, sol, removed, repairer, m_max):
    sol = _repair_core(instance, sol, removed, repairer, _score_insert, m_max)
    _enhance_active(instance, sol, repairer, m_max)
    return sol


def repair_passive_downgrade(instance, sol, removed, repairer, m_max):
    sol = _repair_core(instance, sol, removed, repairer, _score_insert, m_max)
    _downgrade_passive(instance, sol, repairer)
    return sol


def _enhance_active(instance, sol, repairer, m_max):
    """Promote high recovery-risk passive tasks to active backup."""
    from ..base import topo_node_order
    from ..replicas import add_active_replicas
    promote = set()
    for i in list(sol.placement.keys()):
        if sol.mode.get(i):
            continue
        k = sol.placement[i][0]
        _, comm_in = repairer._comm_in(i, k, {p: sol.placement[p][0] for p in sol.placement}, {})
        if repairer.rec_risk(i, k, comm_in) > repairer.budget:
            promote.add(i)
    if promote:
        add_active_replicas(instance, sol, promote, m_max, repairer.cfg)
        sol.order = topo_node_order(instance, sol)


def _downgrade_passive(instance, sol, repairer):
    """Demote active tasks whose passive risk already fits the budget."""
    from ..base import topo_node_order
    changed = False
    for i in list(sol.placement.keys()):
        if not sol.mode.get(i):
            continue
        k = sol.placement[i][0]
        _, comm_in = repairer._comm_in(i, k, {p: sol.placement[p][0] for p in sol.placement}, {})
        if repairer.rec_risk(i, k, comm_in) <= repairer.budget:
            sol.placement[i] = [k]
            sol.mode[i] = False
            changed = True
    if changed:
        sol.order = topo_node_order(instance, sol)


REPAIR_OPS = {
    "insert": repair_insert,
    "min_recdelay": repair_min_recdelay,
    "min_cost": repair_min_cost,
    "active_enhance": repair_active_enhance,
    "passive_downgrade": repair_passive_downgrade,
}
