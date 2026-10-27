"""Read-only diagnostic: instrument RECR-D Alg.2 placement decisions.

For each placed task, record whether a sub-deadline-feasible pool existed
(cost-minimising mode) or not (earliest-finish catch-up mode), how many
candidates survived the capacity filter, and whether the winning node was the
same one a predecessor used (co-location, i.e. zero communication delay).
"""
import glob
import math
from collections import Counter

from gcnalns.algorithms.heft import _topo_priority_order
from gcnalns.config import load_config
from gcnalns.core.dag import compute_structural
from gcnalns.core.schedule_eval import exec_time
from gcnalns.data.instance_builder import load_instance
from gcnalns.data.links import get_factory
from gcnalns.reproductions._shared import DynamicSubDeadlines, sub_deadlines

cfg = load_config(None)
p = cfg["reproductions"]["recr_d"]
path = sorted(glob.glob("data/instances/medium/*_loose_*.json"))[0]

inst = load_instance(path, cfg)
factory = get_factory(inst, cfg)
compute_structural(inst.workflow, inst.continuum, cfg, factory)
wf, nodes = inst.workflow, inst.continuum.nodes
theta = float(p["theta_reliability"])
zeta, beta, gamma = (float(p["zeta_cost"]), float(p["beta_deadline"]),
                     float(p["gamma_reliability"]))

_st, subdl = sub_deadlines(inst, float(p["db"]))
edf = sorted(wf.tasks, key=lambda i: (subdl.get(i, 0.0), -wf.tasks[i].ranku))
order = _topo_priority_order(wf, edf)
dyn = DynamicSubDeadlines(inst, float(p["db"]))

pred = {t: wf.predecessors(t) for t in wf.tasks}
node_free = {k: 0.0 for k in nodes}
used_cpu = {k: 0.0 for k in nodes}
used_mem = {k: 0.0 for k in nodes}
finish, where = {}, {}
mode_count = Counter()
cand_sizes, pool_sizes = [], []
colocated = 0
eft_gap = []  # (chosen eft - min available eft) when in cost mode

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
        cands.append((k, est + p_ik, est, node.cost_coef * p_ik,
                      math.exp(-node.failure_rate * p_ik)))
    cand_sizes.append(len(cands))
    dl_i = dyn.of(i)
    feasible = [c for c in cands if dl_i <= 1e-12 or c[1] <= dl_i + 1e-9]
    pool = feasible if feasible else cands
    pool_sizes.append(len(pool))
    mode_count["cost" if feasible else "catchup"] += 1
    cost_max = max(c[3] for c in cands)
    best = None
    for (k, eft, est, cost, rel) in pool:
        if feasible:
            dl_norm = max(0.0, (eft - dl_i) / dl_i) if dl_i > 1e-12 else 0.0
            pn = zeta * (cost / (cost_max + 1.0)) + beta * dl_norm \
                + gamma * max(0.0, theta - rel)
        else:
            pn = eft
        if best is None or pn < best[0]:
            best = (pn, k, eft)
    _pn, best_k, best_eft = best
    if feasible:
        eft_gap.append(best_eft - min(c[1] for c in cands))
    if any(where.get(pr) == best_k for pr in pred[i]):
        colocated += 1
    node_free[best_k] = best_eft
    used_cpu[best_k] += task.resource
    used_mem[best_k] += task.mem
    finish[i] = best_eft
    where[i] = best_k
    dyn.observe(i, best_eft)

n = len(order)
print(f"instance={path}  tasks={n}  nodes={len(nodes)}  D={inst.deadline:.1f}")
print(f"placement mode: {dict(mode_count)}")
print(f"candidates surviving capacity filter: min={min(cand_sizes)} "
      f"mean={sum(cand_sizes) / n:.1f} max={max(cand_sizes)}")
print(f"sub-deadline-feasible pool size:      min={min(pool_sizes)} "
      f"mean={sum(pool_sizes) / n:.1f} max={max(pool_sizes)}")
if eft_gap:
    print(f"cost-mode EFT sacrifice (chosen - best possible): "
          f"mean={sum(eft_gap) / len(eft_gap):.2f} max={max(eft_gap):.2f}")
print(f"tasks co-located with a predecessor: {colocated}/{n}")
print(f"final makespan={max(finish.values()):.1f}  nodes_used={len(set(where.values()))}")
print("top-5 loaded nodes:", Counter(where.values()).most_common(5))
