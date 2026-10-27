"""Read-only diagnostic for HFWPF on medium-loose: is the bottleneck deadline or the recovery gate?"""
import glob, os
from gcnalns.config import load_config
from gcnalns.data.instance_builder import load_instance
from gcnalns.data.links import get_factory
from gcnalns.algorithms.registry import get_scheduler
from gcnalns.core.schedule_eval import compute_schedule
from gcnalns.core.recovery import effective_budget, estimate_rmax, verify

cfg = load_config(None)
paths = sorted(glob.glob("data/instances/medium/*_loose_*.json"))

print(f"{'instance':<40} {'D':>8} {'ms':>8} {'dl_ok':>6} {'Rmax':>7} {'R_eff':>7} {'r_ok':>6} {'nact':>5} {'feas':>6}")
n_dlok=n_rok=n_feas=0
for p in paths:
    inst = load_instance(p, cfg); factory = get_factory(inst, cfg)
    sol = get_scheduler("HFWPF")(cfg, seed=0).schedule(inst)
    rs = compute_schedule(inst, sol, factory, cfg)
    reff = effective_budget(inst, rs.cmax); rmax = estimate_rmax(inst, sol, rs, factory, cfg)
    nact = sum(1 for i in inst.workflow.tasks if sol.is_active(i))
    dl_ok = rs.cmax <= inst.deadline + 1e-6
    r_ok = rmax <= reff + 1e-6
    vr = verify(inst, sol, factory, cfg)
    feas = getattr(vr, "feasible", None)
    if feas is None and isinstance(vr, tuple): feas = vr[0]
    n_dlok+=dl_ok; n_rok+=r_ok; n_feas+=bool(feas)
    print(f"{os.path.basename(p)[:-5]:<40} {inst.deadline:>8.1f} {rs.cmax:>8.1f} {str(dl_ok):>6} "
          f"{rmax:>7.1f} {reff:>7.1f} {str(r_ok):>6} {nact:>5} {str(bool(feas)):>6}")
n=len(paths)
print(f"\nsummary: deadline_ok={100*n_dlok/n:.0f}%  rmax_ok={100*n_rok/n:.0f}%  feasible={100*n_feas/n:.0f}%  (n={n})")
print("active tasks per instance shows FT provisioning; if nact small & r_ok False -> under-provisioned recovery.")
