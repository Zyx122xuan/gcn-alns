"""Read-only diagnostic: why does RECR-D miss the deadline even on loose instances?
Compares RECR-D vs HEFT base makespan on medium-loose instances."""
import glob, os
from gcnalns.config import load_config
from gcnalns.data.instance_builder import load_instance
from gcnalns.data.links import get_factory
from gcnalns.algorithms.registry import get_scheduler
from gcnalns.core.schedule_eval import compute_schedule
from gcnalns.core.recovery import effective_budget, estimate_rmax

cfg = load_config(None)
paths = sorted(glob.glob("data/instances/medium/*_loose_*.json"))

import sys
if "--betasweep" in sys.argv:
    print("=== beta_deadline sweep on ALL medium-loose instances (deadline_ok rate) ===")
    for beta in [1.0, 2.0, 3.0, 5.0, 10.0]:
        cfg["reproductions"]["recr_d"]["beta_deadline"] = beta
        ok = 0; ratios = []
        for p in paths:
            inst = load_instance(p, cfg); factory = get_factory(inst, cfg)
            rsol = get_scheduler("RECR-D")(cfg, seed=0).schedule(inst)
            rs = compute_schedule(inst, rsol, factory, cfg)
            ok += 1 if rs.cmax <= inst.deadline + 1e-6 else 0
            ratios.append(rs.cmax / inst.deadline)
        import statistics
        print(f"  beta={beta:>5}: deadline_ok={100*ok/len(paths):5.1f}%  "
              f"mean(makespan/deadline)={statistics.mean(ratios):.2f}")
    sys.exit(0)

paths = paths[:6]

print(f"{'instance':<42} {'deadline':>9} {'HEFT_ms':>9} {'RECRD_ms':>9} {'dl_ok':>6} {'ovf':>4} {'RECRD_Rmax':>10} {'R_eff':>7} {'nact':>5}")
for p in paths:
    inst = load_instance(p, cfg)
    factory = get_factory(inst, cfg)
    D = inst.deadline
    row = os.path.basename(p)[:-5]
    # HEFT reference makespan
    heft = get_scheduler("HEFT")(cfg, seed=0).schedule(inst)
    hs = compute_schedule(inst, heft, factory, cfg)
    # RECR-D
    rec = get_scheduler("RECR-D")(cfg, seed=0)
    rsol = rec.schedule(inst)
    rs = compute_schedule(inst, rsol, factory, cfg)
    reff = effective_budget(inst, rs.cmax)
    rmax = estimate_rmax(inst, rsol, rs, factory, cfg)
    nact = sum(1 for i in inst.workflow.tasks if rsol.is_active(i))
    ovf = rsol.meta.get("capacity_overflow", False)
    dl_ok = rs.cmax <= D + 1e-6
    print(f"{row:<42} {D:>9.1f} {hs.cmax:>9.1f} {rs.cmax:>9.1f} {str(dl_ok):>6} {str(ovf):>4} "
          f"{rmax:>10.1f} {reff:>7.1f} {nact:>5}")

# Node-usage inspection on the first instance: does RECR-D pile tasks on few nodes?
print("\n--- node placement spread on first instance (RECR-D) ---")
inst = load_instance(paths[0], cfg)
factory = get_factory(inst, cfg)
rec = get_scheduler("RECR-D")(cfg, seed=0); rsol = rec.schedule(inst)
from collections import Counter
c = Counter()
for i, repl in rsol.placement.items():
    for k in repl: c[k]+=1
print(f"tasks={len(inst.workflow.tasks)} nodes_used={len(c)} / total_nodes={len(inst.continuum.nodes)}")
print("top-5 loaded nodes (node->#tasks):", c.most_common(5))
# node speed vs cost: are chosen nodes the slow/cheap ones?
speeds = {k: nd.mips if hasattr(nd,'mips') else getattr(nd,'speed',None) for k,nd in inst.continuum.nodes.items()}
costs = {k: nd.cost_coef for k,nd in inst.continuum.nodes.items()}
used = set(c)
allnodes = list(inst.continuum.nodes)
import statistics as st
def avg(d, keys): 
    vals=[d[k] for k in keys if d.get(k) is not None]
    return sum(vals)/len(vals) if vals else float('nan')
print(f"avg cost_coef: used={avg(costs,used):.3f}  all={avg(costs,allnodes):.3f}")
