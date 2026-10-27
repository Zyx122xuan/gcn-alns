"""Fault-domain-isolation robustness study (read-only w.r.t. the main pipeline).

For each small instance, solve GCN-ALNS with Full vs w/o Fault-Domain, then check
SLA feasibility under (a) the default single-point fault set and (b) the same set
augmented with DOMAIN-level faults (whole fault domain fails at once). Expectation:
under domain faults, w/o Fault-Domain degrades sharply (co-located replicas both
die) while Full stays robust -- making the value of the isolation constraint visible.

Writes only to results/logs/domain_fault_exp.txt (no change to raw_json/csv).
"""
import glob, os, json
from concurrent.futures import ProcessPoolExecutor

def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass

def _work(args):
    inst_path, label, ablation, seed, coarse_k = args
    from gcnalns.config import load_config
    from gcnalns.data.instance_builder import load_instance
    from gcnalns.data.links import get_factory
    from gcnalns.algorithms.registry import get_scheduler
    from gcnalns.core.schedule_eval import compute_schedule
    from gcnalns.core.fault_set import build_fault_set, build_domain_faults
    from gcnalns.core.recovery import simulate_fault, effective_budget, INF

    cfg = load_config(None)
    inst = load_instance(inst_path, cfg)
    # Optional: coarsen fault domains to K per layer (models correlated infra:
    # few large racks/AZs). Default fine-grained domains ~= 1 node each, so the
    # isolation constraint rarely binds; coarse domains make it bite.
    if coarse_k and coarse_k > 0:
        for nid, nd in inst.continuum.nodes.items():
            nd.fault_domain = f"{nd.layer}-{nid % coarse_k}"
    factory = get_factory(inst, cfg)
    mp = cfg.get("gcn.model_path", "results/gcn_model.pt")
    sched = get_scheduler("GCN-ALNS")(cfg, seed=seed, ablation=ablation, model_path=mp)
    sol = sched.schedule(inst)
    base = compute_schedule(inst, sol, factory, cfg)
    r_eff = effective_budget(inst, base.cmax)
    deadline_ok = base.cmax <= inst.deadline + 1e-9

    def rmax_over(faults):
        r = 0.0
        for f in faults:
            c = simulate_fault(inst, sol, base, f, factory, cfg, ablation=ablation)
            r = INF if (c == INF or r == INF) else max(r, max(0.0, c - base.cmax))
        return r

    sp = build_fault_set(inst, sol, base)               # single-point Omega
    dom = build_domain_faults(inst, sol)                # domain-level faults
    r_sp = rmax_over(sp)
    r_aug = rmax_over(sp + dom)
    feas_sp = deadline_ok and (r_sp <= r_eff + 1e-9)
    feas_aug = deadline_ok and (r_aug <= r_eff + 1e-9)
    # count how many active-replica tasks are domain-isolated vs co-located
    nodes = inst.continuum.nodes
    n_active = n_isolated = 0
    for i, repl in sol.placement.items():
        if sol.is_active(i) and len(repl) >= 2:
            n_active += 1
            doms = {nodes[k].fault_domain for k in repl}
            if len(doms) == len(repl):
                n_isolated += 1
    return dict(iid=os.path.basename(inst_path)[:-5], label=label,
                scenario=inst.meta.get("sla_scenario", "?"),
                r_eff=r_eff, r_sp=r_sp, r_aug=r_aug,
                feas_sp=feas_sp, feas_aug=feas_aug,
                n_domain_faults=len(dom), n_active=n_active, n_isolated=n_isolated)

def main():
    import sys
    coarse_k = 0
    for a in sys.argv[1:]:
        if a.startswith("--coarse="):
            coarse_k = int(a.split("=")[1])
    paths = sorted(glob.glob("data/instances/small/*.json"))
    configs = [("Full", None), ("w/o Fault-Domain", {"fault_domain": False})]
    tasks = [(p, lbl, abl, 0, coarse_k) for p in paths for (lbl, abl) in configs]
    print(f"instances={len(paths)}  solves={len(tasks)} (GCN-ALNS, small, seed=0)  "
          f"coarse_k={coarse_k} ({'coarse domains' if coarse_k else 'original fine domains'})")
    results = []
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as ex:
        for i, r in enumerate(ex.map(_work, tasks), 1):
            results.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)} done")

    def agg(label, key):
        sub = [r for r in results if r["label"] == label]
        return 100 * sum(1 for r in sub if r[key]) / len(sub) if sub else float("nan")

    print("\n================ Fault-Domain isolation robustness (small) ================")
    print(f"{'config':<18} {'feas @ single-point':>20} {'feas @ +domain faults':>22}")
    for lbl, _ in configs:
        print(f"{lbl:<18} {agg(lbl,'feas_sp'):>19.1f}% {agg(lbl,'feas_aug'):>21.1f}%")

    # isolation stats
    print("\nactive-replica domain isolation (mean per instance):")
    for lbl, _ in configs:
        sub = [r for r in results if r["label"] == lbl]
        na = sum(r["n_active"] for r in sub); ni = sum(r["n_isolated"] for r in sub)
        print(f"  {lbl:<18} active-replica tasks={na}, domain-isolated={ni} "
              f"({100*ni/na if na else float('nan'):.0f}% isolated)")

    # by scenario (feas @ +domain)
    print("\nfeas @ +domain faults, by scenario:")
    scens = sorted({r["scenario"] for r in results})
    print(f"{'config':<18} " + " ".join(f"{s:>9}" for s in scens))
    for lbl, _ in configs:
        row = []
        for s in scens:
            sub = [r for r in results if r["label"] == lbl and r["scenario"] == s]
            row.append(100*sum(1 for r in sub if r["feas_aug"])/len(sub) if sub else float("nan"))
        print(f"{lbl:<18} " + " ".join(f"{v:>8.1f}%" for v in row))

    os.makedirs("results/logs", exist_ok=True)
    suffix = f"_coarse{coarse_k}" if coarse_k else "_fine"
    outp = f"results/logs/domain_fault_exp{suffix}.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved per-instance detail -> {outp}")

if __name__ == "__main__":
    main()
