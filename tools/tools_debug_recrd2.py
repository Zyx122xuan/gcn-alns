"""Read-only diagnostic: trace RECR-D sub-deadline slack consumption.

Question: are the per-task sub-deadlines individually satisfied while the
global makespan still blows past D (i.e. the structural slack is being
re-consumed by every task independently)?
"""
import glob

from gcnalns.algorithms.registry import get_scheduler
from gcnalns.config import load_config
from gcnalns.core.dag import compute_structural
from gcnalns.core.schedule_eval import compute_schedule
from gcnalns.data.instance_builder import load_instance
from gcnalns.data.links import get_factory
from gcnalns.reproductions._shared import sub_deadlines

cfg = load_config(None)
p = cfg["reproductions"]["recr_d"]
path = sorted(glob.glob("data/instances/medium/*_loose_*.json"))[0]

inst = load_instance(path, cfg)
factory = get_factory(inst, cfg)
compute_structural(inst.workflow, inst.continuum, cfg, factory)
tasks = inst.workflow.tasks
D = inst.deadline
cmax_struct = max(t.eft for t in tasks.values())
a = float(p["db"])
rho = min(1.0, max(0.0, (D - cmax_struct) / D))
print(f"instance={path}")
print(f"D={D:.1f}  T_min(structural cmax)={cmax_struct:.1f}  "
      f"slack={D - cmax_struct:.1f}  rho={rho:.3f}  a={a}  DB={a + (1 - a) * rho:.3f}")

_st, subdl = sub_deadlines(inst, a)
dl_vals = sorted(subdl.values())
print(f"sub-deadlines: min={dl_vals[0]:.1f} median={dl_vals[len(dl_vals) // 2]:.1f} "
      f"max={dl_vals[-1]:.1f}   (D={D:.1f})")
n_over = sum(1 for v in dl_vals if v > D + 1e-6)
print(f"#tasks whose sub-deadline already exceeds D: {n_over}/{len(dl_vals)}")

# How much slack does each task individually get, relative to its own EFT?
slacks = [subdl[i] - tasks[i].eft for i in tasks]
print(f"per-task slack dl_i-EFT_i: mean={sum(slacks) / len(slacks):.1f} "
      f"max={max(slacks):.1f}  (global slack={D - cmax_struct:.1f})")

# Actual RECR-D schedule: how many tasks met their own sub-deadline?
sol = get_scheduler("RECR-D")(cfg, seed=0).schedule(inst)
sched = compute_schedule(inst, sol, factory, cfg)
met = sum(1 for i in tasks if sched.eff_finish.get(i, 0.0) <= subdl[i] + 1e-6)
print(f"\nRECR-D makespan={sched.cmax:.1f} (D={D:.1f})  "
      f"tasks meeting own sub-deadline: {met}/{len(tasks)}")
lateness = [sched.eff_finish[i] - tasks[i].eft for i in tasks if i in sched.eff_finish]
print(f"finish - structural EFT: mean={sum(lateness) / len(lateness):.1f} "
      f"max={max(lateness):.1f}")
