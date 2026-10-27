"""Read-only diagnostic: does aligning Alg.2's planning order with the
evaluator's replay order fix RECR-D's makespan?

compute_schedule replays every node's queue in topological_order(wf) and
ignores solution.order, while Alg.2 plans in EDF order. Because a node's
timeline is append-only, the mismatch blocks heavily loaded nodes. Here we run
the placement loop under both orders and evaluate each with compute_schedule.
"""
import glob
import math

from gcnalns.algorithms.heft import _topo_priority_order
from gcnalns.config import load_config
from gcnalns.core.dag import compute_structural, topological_order
from gcnalns.core.model import Solution
from gcnalns.core.schedule_eval import compute_schedule, exec_time
from gcnalns.data.instance_builder import load_instance
from gcnalns.data.links import get_factory
from gcnalns.reproductions._shared import DynamicSubDeadlines, sub_deadlines

cfg = load_config(None)
p = cfg["reproductions"]["recr_d"]
theta = float(p["theta_reliability"])
zeta, beta, gamma = (float(p["zeta_cost"]), float(p["beta_deadline"]),
                     float(p["gamma_reliability"]))


def place(inst, factory, order, dyn):
    wf, nodes = inst.workflow, inst.continuum.nodes
    pred = {t: wf.predecessors(t) for t in wf.tasks}
    sol = Solution()
    node_free = {k: 0.0 for k in nodes}
    used_cpu = {k: 0.0 for k in nodes}
    used_mem = {k: 0.0 for k in nodes}
    finish, where = {}, {}
    for i in order:
        task = wf.tasks[i]
        cands = []
        for k, node in nodes.items():
            if used_cpu[k] + task.resource > node.res_cpu + 1e-9:
                continue
            if used_mem[k] + task.mem > node.res_mem + 1e-9:
                continue
            est = node_free[k]
            for pr in pred[i]:
                d = wf.data.get((pr, i), 0.0)
                est = max(est, finish[pr] + factory.comm_delay(where[pr], k, d))
            p_ik = exec_time(inst, i, k)
            cands.append((k, est + p_ik, node.cost_coef * p_ik,
                          math.exp(-node.failure_rate * p_ik)))
        if not cands:
            best_k = min(nodes, key=lambda k: node_free[k])
            best_eft = node_free[best_k] + exec_time(inst, i, best_k)
        else:
            cost_max = max(c[2] for c in cands)
            dl_i = dyn.of(i)
            feasible = [c for c in cands if dl_i <= 1e-12 or c[1] <= dl_i + 1e-9]
            pool = feasible if feasible else cands
            best = None
            for (k, eft, cost, rel) in pool:
                if feasible:
                    dl_norm = max(0.0, (eft - dl_i) / dl_i) if dl_i > 1e-12 else 0.0
                    pn = zeta * (cost / (cost_max + 1.0)) + beta * dl_norm \
                        + gamma * max(0.0, theta - rel)
                else:
                    pn = eft
                if best is None or pn < best[0]:
                    best = (pn, k, eft)
            _pn, best_k, best_eft = best
        sol.placement[i] = [best_k]
        sol.mode[i] = False
        node_free[best_k] = best_eft
        used_cpu[best_k] += task.resource
        used_mem[best_k] += task.mem
        finish[i] = best_eft
        where[i] = best_k
        dyn.observe(i, best_eft)
    return sol, max(finish.values())


print(f"{'instance':<40} {'D':>8} {'HEFT':>8} {'EDF_plan':>9} {'EDF_eval':>9} "
      f"{'TOPO_plan':>10} {'TOPO_eval':>10}")
from gcnalns.algorithms.registry import get_scheduler
for path in sorted(glob.glob("data/instances/medium/*_loose_*.json"))[:6]:
    inst = load_instance(path, cfg)
    factory = get_factory(inst, cfg)
    compute_structural(inst.workflow, inst.continuum, cfg, factory)
    wf = inst.workflow
    _st, subdl = sub_deadlines(inst, float(p["db"]))
    hs = compute_schedule(inst, get_scheduler("HEFT")(cfg, seed=0).schedule(inst),
                          factory, cfg)

    edf = _topo_priority_order(wf, sorted(
        wf.tasks, key=lambda i: (subdl.get(i, 0.0), -wf.tasks[i].ranku)))
    s1, plan1 = place(inst, factory, edf, DynamicSubDeadlines(inst, float(p["db"])))
    s1.order = edf
    e1 = compute_schedule(inst, s1, factory, cfg).cmax

    topo = topological_order(wf)
    s2, plan2 = place(inst, factory, topo, DynamicSubDeadlines(inst, float(p["db"])))
    s2.order = topo
    e2 = compute_schedule(inst, s2, factory, cfg).cmax

    name = path.split("\\")[-1].split("/")[-1][:-5]
    print(f"{name:<40} {inst.deadline:>8.1f} {hs.cmax:>8.1f} {plan1:>9.1f} "
          f"{e1:>9.1f} {plan2:>10.1f} {e2:>10.1f}")
