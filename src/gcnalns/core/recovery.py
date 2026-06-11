"""Recovery simulator / verifier (Modeling doc sec.8, Spec sec.8-13, 5.2).

THE correctness gate. Every algorithm calls this verifier so SLA feasibility is
judged identically. For a solution chi and each controllable single-point fault
omega in Omega(chi) it computes the post-fault makespan C_max^omega via a
fault-aware forward re-simulation, then:

    R_omega   = max(0, C_max^omega - C_max^0)
    R_max     = max_omega R_omega
    R_eff     = min(R^max, D - C_max^0)              (effective recovery budget)
    feasible  = C_max^0 <= D  and  R_max <= R_eff

Deterministic fault timing (Spec sec.9):
* task instance fault: t_omega = planned finish of that instance
* node fault:          t_omega = earliest start among that node's instances
* link fault:          t_omega = planned transfer completion time

Recovery semantics:
* active backup + healthy replica -> switch:
      C_rec = min_{healthy} max(f_r, t_omega + det + sw + sync)
* otherwise (passive / all replicas hit) -> dynamic reschedule + recompute:
      F'_i = max(input_ready, t_omega + det + replan) + queue + p_ik'
  recovery node chosen by RecCost (Spec sec.12); infeasible -> +inf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..data.links import LinkFactory
from .dag import topological_order
from .fault_set import Fault, build_fault_set
from .model import Instance, Solution
from .schedule_eval import ScheduleResult, compute_schedule, exec_time

INF = float("inf")


@dataclass
class RecoveryResult:
    c_base: float = 0.0
    c_worst: float = 0.0
    r_max: float = 0.0
    r_eff: float = 0.0
    margin: float = 0.0
    feasible: bool = False
    per_fault: dict[str, float] = field(default_factory=dict)     # label -> R_omega
    per_fault_cmax: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, list[float]] = field(default_factory=dict)  # kind -> [R_omega...]
    n_faults: int = 0
    avg_recovery: float = 0.0
    recovery_cost: float = 0.0


# ---------------------------------------------------------------------------
# Fault timing
# ---------------------------------------------------------------------------
def _fault_time(fault: Fault, sched: ScheduleResult, instance: Instance,
                factory: LinkFactory) -> float:
    if fault.kind == "task":
        return sched.finish[(fault.i, fault.k)]
    if fault.kind == "node":
        starts = [s for (i, k), s in sched.start.items() if k == fault.k]
        return min(starts) if starts else 0.0
    # link: producer finish on k + comm to l
    d = instance.workflow.data.get((fault.i, fault.j), 0.0)
    return sched.finish[(fault.i, fault.k)] + factory.comm_delay(fault.k, fault.l, d)


# ---------------------------------------------------------------------------
# Recovery-node selection (Spec sec.12)
# ---------------------------------------------------------------------------
def _rec_cost(instance: Instance, solution: Solution, i: int, k: int,
              f_omega: dict[int, float], node_omega: dict[int, int],
              factory: LinkFactory, cfg: Config) -> float:
    rc = cfg["rec_cost"]
    node = instance.continuum.nodes[k]
    wf = instance.workflow
    p_ik = exec_time(instance, i, k)
    comm_in = 0.0
    for p in wf.predecessors(i):
        d = wf.data.get((p, i), 0.0)
        kp = node_omega.get(p, k)
        comm_in += factory.comm_delay(kp, k, d)
    risk = wf.tasks[i].criticality * (1.0 - pow(2.718281828, -node.failure_rate * p_ik))
    cost = node.cost_coef * p_ik
    return (rc["theta1"] * node.queue_wait + rc["theta2"] * comm_in
            + rc["theta3"] * p_ik + rc["theta4"] * risk + rc["theta5"] * cost)


def _choose_recovery_node(instance: Instance, solution: Solution, i: int,
                          forbidden_nodes: set[int], forbidden_domains: set[str],
                          used_cpu: dict[int, float], used_mem: dict[int, float],
                          f_omega: dict[int, float], node_omega: dict[int, int],
                          factory: LinkFactory, cfg: Config) -> int | None:
    task = instance.workflow.tasks[i]
    best, best_cost = None, INF
    for k, node in instance.continuum.nodes.items():
        if k in forbidden_nodes:
            continue
        if node.fault_domain in forbidden_domains:
            continue
        if used_cpu.get(k, 0.0) + task.resource > node.res_cpu + 1e-9:
            continue
        if used_mem.get(k, 0.0) + task.mem > node.res_mem + 1e-9:
            continue
        c = _rec_cost(instance, solution, i, k, f_omega, node_omega, factory, cfg)
        if c < best_cost:
            best_cost, best = c, k
    return best


# ---------------------------------------------------------------------------
# Single-fault simulation
# ---------------------------------------------------------------------------
def simulate_fault(instance: Instance, solution: Solution, base: ScheduleResult,
                   fault: Fault, factory: LinkFactory, cfg: Config) -> float:
    """Return post-fault makespan C_max^omega (or +inf if unrecoverable)."""
    wf = instance.workflow
    rcv = cfg["recovery"]
    det, replan = rcv["delta_detect"], rcv["delta_replan"]
    sw, sync = rcv["delta_switch"], rcv["delta_sync"]
    link_rec = rcv["delta_link_rec"]

    t_omega = _fault_time(fault, base, instance, factory)
    failed_node = fault.k if fault.kind == "node" else -1

    order = topological_order(wf)
    f_omega: dict[int, float] = {}
    node_omega: dict[int, int] = {}
    node_free: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}
    # resource usage for recovery placement (start from solution footprint)
    used_cpu: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}
    used_mem: dict[int, float] = {k: 0.0 for k in instance.continuum.nodes}
    for ti, repl in solution.placement.items():
        for k in repl:
            used_cpu[k] += wf.tasks[ti].resource
            used_mem[k] += wf.tasks[ti].mem

    def input_ready(i: int, k: int, retransmit_edge: tuple[int, int] | None) -> float:
        ready = 0.0
        for p in wf.predecessors(i):
            d = wf.data.get((p, i), 0.0)
            kp = node_omega[p]
            arrive = f_omega[p] + factory.comm_delay(kp, k, d)
            if retransmit_edge == (p, i):
                arrive = max(f_omega[p], t_omega) + link_rec + factory.comm_delay(kp, k, d)
            ready = max(ready, arrive)
        return ready

    for i in order:
        replicas = solution.replicas(i)
        active = solution.is_active(i)
        # edge into i that suffered a link fault (consumer == i)
        link_edge = (fault.i, fault.j) if (fault.kind == "link" and fault.j == i) else None
        directly_hit = False

        # compute candidate finish for each replica, dropping failed ones
        survivor_finishes: dict[int, float] = {}
        for k in replicas:
            # task-instance fault: this exact instance fails
            if fault.kind == "task" and fault.i == i and fault.k == k:
                directly_hit = True
                continue
            start_k = node_free[k]
            start_k = max(start_k, input_ready(i, k, link_edge))
            finish_k = start_k + exec_time(instance, i, k)
            # node fault: replica on failed node survives only if it would have
            # finished by t_omega; else it is destroyed.
            if k == failed_node:
                if finish_k > t_omega + 1e-9:
                    directly_hit = True
                    continue  # destroyed
            survivor_finishes[k] = finish_k

        if link_edge is not None:
            directly_hit = True  # input transfer disrupted -> recovery accounted via retransmit

        if survivor_finishes:
            # apply active-switch floor if this task was directly hit
            if directly_hit and active:
                floor = t_omega + det + sw + sync
                for k in list(survivor_finishes):
                    survivor_finishes[k] = max(survivor_finishes[k], floor)
            eff_k = min(survivor_finishes, key=lambda k: survivor_finishes[k])
            for k, fin in survivor_finishes.items():
                node_free[k] = max(node_free[k], fin)
            f_omega[i] = survivor_finishes[eff_k]
            node_omega[i] = eff_k
        else:
            # dynamic reschedule + recompute on a fresh node
            forbidden_nodes = {failed_node} if failed_node >= 0 else set()
            forbidden_domains: set[str] = set()
            if active:
                for k in replicas:
                    forbidden_domains.add(instance.continuum.nodes[k].fault_domain)
            kp = _choose_recovery_node(instance, solution, i, forbidden_nodes,
                                       forbidden_domains, used_cpu, used_mem,
                                       f_omega, node_omega, factory, cfg)
            if kp is None:
                return INF
            node = instance.continuum.nodes[kp]
            ready = input_ready(i, kp, link_edge)
            start = max(ready, t_omega + det + replan)
            finish = start + node.queue_wait + exec_time(instance, i, kp)
            used_cpu[kp] += wf.tasks[i].resource
            used_mem[kp] += wf.tasks[i].mem
            node_free[kp] = max(node_free[kp], finish)
            f_omega[i] = finish
            node_omega[i] = kp

    return max(f_omega.values()) if f_omega else 0.0


# ---------------------------------------------------------------------------
# Full verification over Omega(S)
# ---------------------------------------------------------------------------
def effective_budget(instance: Instance, c_base: float) -> float:
    """Delta_bar(chi) = min(R^max, D - C_base)."""
    return min(instance.recovery_window, instance.deadline - c_base)


def estimate_rmax(instance: Instance, solution: Solution, sched: ScheduleResult,
                  factory: LinkFactory, cfg: Config) -> float:
    """Cheap upper-bound-style estimate of R_max for search guidance (Spec 5.2.5).

    Per-task local recovery risk vs. its schedule slack; the maximum across
    tasks approximates the worst-case recovery delay without enumerating
    Omega(S). Used only to steer ALNS; the final incumbent is always confirmed
    by the exact :func:`verify`.
    """
    wf = instance.workflow
    rcv = cfg["recovery"]
    det, replan = rcv["delta_detect"], rcv["delta_replan"]
    sw, sync = rcv["delta_switch"], rcv["delta_sync"]
    est = 0.0
    for i, repl in solution.placement.items():
        slack = sched.sched_slack.get(i, 0.0)
        comm_in = 0.0
        for p in wf.predecessors(i):
            d = wf.data.get((p, i), 0.0)
            kp = sched.eff_node.get(p)
            ki = sched.eff_node.get(i)
            if kp is not None and ki is not None:
                comm_in += factory.comm_delay(kp, ki, d)
        active = solution.is_active(i) and len(repl) >= 2
        if active:
            fins = sorted(sched.finish[(i, k)] for k in repl)
            f_i = fins[0]
            backup_gap = fins[1] - f_i if len(fins) > 1 else 0.0
            local = (det + sw + sync) + backup_gap
        else:
            k = repl[0]
            node = instance.continuum.nodes[k]
            p_ik = exec_time(instance, i, k)
            local = det + replan + node.queue_wait + comm_in + p_ik
        est = max(est, max(0.0, local - slack))
    return est


def verify(instance: Instance, solution: Solution, factory: LinkFactory,
           cfg: Config, base: ScheduleResult | None = None) -> RecoveryResult:
    """Run full recovery verification over the controllable fault set."""
    if base is None:
        base = compute_schedule(instance, solution, factory, cfg)

    res = RecoveryResult()
    res.c_base = base.cmax
    res.r_eff = effective_budget(instance, base.cmax)

    faults = build_fault_set(instance, solution, base)
    res.n_faults = len(faults)
    res.by_kind = {"task": [], "node": [], "link": []}

    worst_cmax = base.cmax
    r_max = 0.0
    sum_r = 0.0
    for fault in faults:
        c_omega = simulate_fault(instance, solution, base, fault, factory, cfg)
        r_omega = INF if c_omega == INF else max(0.0, c_omega - base.cmax)
        res.per_fault[fault.label()] = r_omega
        res.per_fault_cmax[fault.label()] = c_omega
        res.by_kind[fault.kind].append(r_omega)
        if c_omega > worst_cmax:
            worst_cmax = c_omega
        r_max = max(r_max, r_omega)
        sum_r += 0.0 if r_omega == INF else r_omega

    res.c_worst = worst_cmax
    res.r_max = r_max
    res.avg_recovery = (sum_r / len(faults)) if faults else 0.0
    res.margin = res.r_eff - r_max
    res.feasible = (base.cmax <= instance.deadline + 1e-9) and (r_max <= res.r_eff + 1e-9)
    # secondary recovery cost proxy: mean extra makespan across faults
    res.recovery_cost = res.avg_recovery
    return res
