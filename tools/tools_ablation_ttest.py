"""Paired significance tests for the ablation study (Full vs each variant).

Pairing is by INSTANCE (iid): for each shared instance we aggregate over the
stochastic repeats, then run a paired test across instances.

- Feasibility  : per-instance feasible-rate (over reps); paired t-test + Wilcoxon
                 over all shared instances (feasibility defined for every instance).
- Cost/Safety/ttf : per-instance feasible-only mean; paired test over the subset
                 of instances that have >=1 feasible rep in BOTH Full and variant.

Multiple-comparison note: 13 variants vs Full -> Holm-Bonferroni adjusted p also
reported for the feasibility metric.

Read-only. Prints a report to stdout.
"""
import json, os, math
from collections import defaultdict
from scipy import stats

RAW = "results/raw_json"

def load_abl():
    d = {}
    for fn in os.listdir(RAW):
        if fn.startswith("ablation_") and fn.endswith(".jsonl"):
            name = fn[len("ablation_"):-len(".jsonl")]
            with open(os.path.join(RAW, fn), encoding="utf-8") as f:
                d[name] = [json.loads(l) for l in f if l.strip()]
    return d

def per_instance(records, scale=None):
    """Return dict[iid] -> list of records (optionally filtered by scale)."""
    g = defaultdict(list)
    for r in records:
        if scale and r.get("scale") != scale:
            continue
        g[r["iid"]].append(r)
    return g

def feas_rate(recs):
    return sum(1.0 for r in recs if r.get("feasible")) / len(recs) if recs else float("nan")

def feas_mean(recs, field):
    xs = [r[field] for r in recs if r.get("feasible") and r.get(field) is not None]
    return sum(xs)/len(xs) if xs else None

def holm(pvals):
    """Holm-Bonferroni adjusted p-values (returns list aligned to input order)."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); adj = [0.0]*m; prev = 0.0
    for rank, i in enumerate(idx):
        a = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, a); adj[i] = prev
    return adj

def paired_feasibility(full_g, var_g, iids):
    a = [feas_rate(full_g[i]) for i in iids]
    b = [feas_rate(var_g[i]) for i in iids]
    diff = [x - y for x, y in zip(a, b)]  # Full - variant (>0 => Full better)
    mean_full, mean_var = sum(a)/len(a), sum(b)/len(b)
    md = sum(diff)/len(diff)
    sd = (sum((d-md)**2 for d in diff)/(len(diff)-1))**0.5 if len(diff) > 1 else 0.0
    se = sd/math.sqrt(len(diff)) if len(diff) else float("nan")
    try:
        t, pt = stats.ttest_rel(a, b)
    except Exception:
        t, pt = float("nan"), float("nan")
    try:
        if any(d != 0 for d in diff):
            w, pw = stats.wilcoxon(a, b, zero_method="wilcox")
        else:
            w, pw = float("nan"), 1.0
    except Exception:
        w, pw = float("nan"), float("nan")
    return dict(n=len(iids), mean_full=mean_full*100, mean_var=mean_var*100,
                dpp=md*100, ci=1.96*se*100, t=t, p_t=pt, p_w=pw)

def paired_metric(full_g, var_g, iids, field, lower_better):
    pairs = []
    for i in iids:
        fa = feas_mean(full_g[i], field); fb = feas_mean(var_g[i], field)
        if fa is not None and fb is not None:
            pairs.append((fa, fb))
    if len(pairs) < 3:
        return dict(n=len(pairs), mean_full=float("nan"), mean_var=float("nan"),
                    diff=float("nan"), p_t=float("nan"))
    a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
    t, pt = stats.ttest_rel(a, b)
    return dict(n=len(pairs), mean_full=sum(a)/len(a), mean_var=sum(b)/len(b),
                diff=(sum(a)/len(a))-(sum(b)/len(b)), p_t=pt)

def stars(p):
    if p != p: return "  -"
    if p < 0.001: return "***"
    if p < 0.01:  return " **"
    if p < 0.05:  return "  *"
    return " ns"

def run(scale_label, scale):
    abl = load_abl()
    full = per_instance(abl["Full"], scale)
    variants = [k for k in abl if k != "Full"]
    print(f"\n{'='*100}\nPAIRED TESTS vs Full   (scale = {scale_label})\n{'='*100}")
    # report rep count sanity
    reps = sorted({r["rep"] for r in abl["Full"] if (not scale or r.get('scale')==scale)})
    print(f"Full reps present: {reps}  (n_instances={len(full)})")

    # ---- Feasibility (primary) ----
    print(f"\n[Feasibility]  Full - variant (>0 => removing the component HURTS feasibility)")
    print(f"{'variant':<26} {'n':>3} {'Full%':>6} {'var%':>6} {'Δpp':>7} {'95%CI':>7} {'p(t)':>9} {'p(Wil)':>9} {'sig':>4}")
    fea_res = {}
    for v in variants:
        vg = per_instance(abl[v], scale)
        iids = sorted(set(full) & set(vg))
        fea_res[v] = paired_feasibility(full, vg, iids)
    # Holm adjust across variants
    padj = holm([fea_res[v]["p_t"] for v in variants])
    for v, pa in sorted(zip(variants, padj), key=lambda kv: fea_res[kv[0]]["dpp"], reverse=True):
        r = fea_res[v]
        print(f"{v:<26} {r['n']:>3} {r['mean_full']:>6.1f} {r['mean_var']:>6.1f} "
              f"{r['dpp']:>+7.1f} {r['ci']:>7.1f} {r['p_t']:>9.4f} {r['p_w']:>9.4f} {stars(r['p_t']):>4}"
              f"  Holm p={pa:.4f}{stars(pa)}")

    # ---- Cost / Safety / ttf (feasible-only, instances feasible in both) ----
    for field, lower, label in [("cost_total", True, "Cost (feasible-only, lower better)"),
                                 ("safety_margin", False, "Safety margin (feasible-only)"),
                                 ("first_feasible_iter", True, "First-feasible iter (lower=faster)")]:
        print(f"\n[{label}]  Full - variant")
        print(f"{'variant':<26} {'npair':>5} {'Full':>9} {'var':>9} {'diff':>9} {'p(t)':>9} {'sig':>4}")
        for v in variants:
            vg = per_instance(abl[v], scale)
            iids = sorted(set(full) & set(vg))
            r = paired_metric(full, vg, iids, field, lower)
            print(f"{v:<26} {r['n']:>5} {r['mean_full']:>9.2f} {r['mean_var']:>9.2f} "
                  f"{r['diff']:>+9.2f} {r['p_t']:>9.4f} {stars(r['p_t']):>4}")

if __name__ == "__main__":
    run("medium (discriminative)", "medium")
    run("small+medium (global)", None)
    print("\nLegend: *** p<0.001  ** p<0.01  * p<0.05  ns not significant.")
    print("Δpp = Full% - variant% (percentage points). Positive => component matters (removing it lowers feasibility).")
