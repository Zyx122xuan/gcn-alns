"""Read-only probe: RECR-D feasible rate on one instance tier after the
order-alignment fix, so the gain is known before committing to a full rerun."""
import glob
import sys
from collections import defaultdict

from gcnalns.algorithms.registry import get_scheduler
from gcnalns.config import load_config
from gcnalns.core.recovery import effective_budget, verify
from gcnalns.core.schedule_eval import compute_schedule
from gcnalns.data.instance_builder import load_instance
from gcnalns.data.links import get_factory

tier = sys.argv[1] if len(sys.argv) > 1 else "small"
algo = sys.argv[2] if len(sys.argv) > 2 else "RECR-D"
cfg = load_config(None)
paths = sorted(glob.glob(f"data/instances/{tier}/*.json"))

by_scn = defaultdict(list)
gates = defaultdict(lambda: [0, 0, 0])  # scenario -> [deadline_ok, rmax_ok, feasible]
budget = defaultdict(list)
for path in paths:
    inst = load_instance(path, cfg)
    factory = get_factory(inst, cfg)
    sol = get_scheduler(algo)(cfg, seed=0).schedule(inst)
    sched = compute_schedule(inst, sol, factory, cfg)
    v = verify(inst, sol, factory, cfg, base=sched)
    ok = bool(v.feasible)
    scn = path.replace("\\", "/").split("/")[-1][:-5].split("_")[-2]
    by_scn[scn].append(1.0 if ok else 0.0)
    g = gates[scn]
    g[0] += sched.cmax <= inst.deadline + 1e-6
    g[1] += v.r_max <= effective_budget(inst, sched.cmax) + 1e-6
    g[2] += ok
    # Which term of R_eff = min(R^max, D - C_base) binds, and by how much?
    budget[scn].append((inst.recovery_window, inst.deadline - sched.cmax,
                        v.r_max, sched.cmax / inst.deadline))

total = [x for v in by_scn.values() for x in v]
print(f"{algo} on {tier}: feasible={100 * sum(total) / len(total):.1f}% (n={len(total)})")
print(f"{'scenario':<9} {'n':>4} {'deadline_ok':>12} {'rmax_ok':>9} {'feasible':>9}")
for scn in ("tight", "medium", "loose"):
    if by_scn[scn]:
        n = len(by_scn[scn])
        d, r, f = gates[scn]
        print(f"{scn:<9} {n:>4} {100 * d / n:>11.1f}% {100 * r / n:>8.1f}% "
              f"{100 * f / n:>8.1f}%")

print(f"\nR_eff = min(R^max, D - C_base): which term binds?")
print(f"{'scenario':<9} {'R^max':>8} {'D-C_base':>9} {'r_max':>8} "
      f"{'C_base/D':>9} {'binding':>10}")
for scn in ("tight", "medium", "loose"):
    rows = budget[scn]
    if not rows:
        continue
    m = [sum(c) / len(rows) for c in zip(*rows)]
    binding = "D-C_base" if m[1] < m[0] else "R^max"
    print(f"{scn:<9} {m[0]:>8.1f} {m[1]:>9.1f} {m[2]:>8.1f} {m[3]:>9.2f} "
          f"{binding:>10}")
