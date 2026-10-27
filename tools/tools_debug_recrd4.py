"""Read-only diagnostic: where does RECR-D's makespan blow up after Alg.2?

Alg.2's own placement loop reaches makespan 103 <= D = 123, yet the returned
solution evaluates to ~237. Candidates: (a) compute_schedule replays each node's
queue in a different order than placement assumed, (b) the TRM stage's active
replicas serialise onto already-loaded nodes.
"""
import glob

from gcnalns.algorithms.base import topo_node_order
from gcnalns.algorithms.registry import get_scheduler
from gcnalns.config import load_config
from gcnalns.core.schedule_eval import compute_schedule
from gcnalns.data.instance_builder import load_instance
from gcnalns.data.links import get_factory

cfg = load_config(None)
path = sorted(glob.glob("data/instances/medium/*_loose_*.json"))[0]
inst = load_instance(path, cfg)
factory = get_factory(inst, cfg)
D = inst.deadline

sched_obj = get_scheduler("RECR-D")(cfg, seed=0)
sol = sched_obj.schedule(inst)

# Stage A: strip every active replica -> pure Alg.2 placement, same order.
import copy
bare = copy.deepcopy(sol)
n_active = 0
for i in list(bare.placement):
    if len(bare.placement[i]) > 1:
        n_active += 1
        bare.placement[i] = bare.placement[i][:1]
    bare.mode[i] = False
bare.order = topo_node_order(inst, bare)

rs_full = compute_schedule(inst, sol, factory, cfg)
rs_bare = compute_schedule(inst, bare, factory, cfg)
print(f"instance={path}  D={D:.1f}")
print(f"active-backup tasks in returned solution: {n_active}")
print(f"makespan with replicas   = {rs_full.cmax:.1f}")
print(f"makespan replicas removed= {rs_bare.cmax:.1f}   "
      f"(Alg.2 loop itself measured 103.0)")

# Stage B: does the queue order matter? Re-evaluate bare with no explicit order.
no_order = copy.deepcopy(bare)
no_order.order = None
try:
    rs_no = compute_schedule(inst, no_order, factory, cfg)
    print(f"makespan with order=None = {rs_no.cmax:.1f}")
except Exception as exc:  # noqa: BLE001
    print(f"order=None -> {type(exc).__name__}: {exc}")

# Stage C: how long is the busiest node's own timeline?
from collections import Counter
load = Counter()
for i, repl in sol.placement.items():
    for k in repl:
        load[k] += 1
print("top-5 loaded nodes (with replicas):", load.most_common(5))
busy = load.most_common(1)[0][0]
own = sorted((s, rs_full.finish_task.get((i, k), s))
             for (i, k), s in rs_full.start.items() if k == busy)
if own:
    print(f"busiest node {busy}: {len(own)} slots, "
          f"first start={own[0][0]:.1f}, last finish={own[-1][1]:.1f}")
    idle = sum(max(0.0, own[j + 1][0] - own[j][1]) for j in range(len(own) - 1))
    print(f"  idle time inside that node's span = {idle:.1f}")
