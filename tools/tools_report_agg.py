"""Read-only aggregation of raw_json for the advisor report (new cost model)."""
import json, glob, os
from collections import defaultdict

RAW = "results/raw_json"

def load(tag):
    p = os.path.join(RAW, f"{tag}.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def agg_by(records, keyfn):
    g = defaultdict(list)
    for r in records:
        g[keyfn(r)].append(r)
    return g

def summarize(records):
    n = len(records)
    feas = [r for r in records if r.get("feasible")]
    return {
        "n": n,
        "feasible_rate": mean([1.0 if r.get("feasible") else 0.0 for r in records]),
        "n_feasible": len(feas),
        "cost_total_feas": mean([r["cost_total"] for r in feas]),
        "cost_compute_feas": mean([r["cost_compute"] for r in feas]),
        "cost_comm_feas": mean([r["cost_comm"] for r in feas]),
        "cost_reserve_feas": mean([r["cost_reserve"] for r in feas]),
        "cost_recovery_feas": mean([r["cost_recovery"] for r in feas]),
        "safety_margin_feas": mean([r["safety_margin"] for r in feas]),
        "runtime": mean([r["runtime"] for r in records]),
        "first_feasible_iter": mean([r["first_feasible_iter"] for r in records if r.get("first_feasible_iter", -1) >= 0]),
        "gcn_infer": mean([r.get("gcn_infer_time", 0.0) for r in records]),
    }

def print_table(title, groups, order=None):
    print(f"\n===== {title} =====")
    keys = order if order else sorted(groups.keys(), key=str)
    print(f"{'key':<28} {'n':>5} {'feas%':>7} {'nfeas':>6} {'cost_tot':>9} {'compute':>8} {'comm':>7} {'reserve':>7} {'recov':>6} {'safety':>7} {'ttf':>6} {'runtime':>8}")
    for k in keys:
        if k not in groups:
            continue
        s = summarize(groups[k])
        print(f"{str(k):<28} {s['n']:>5} {s['feasible_rate']*100:>6.1f} {s['n_feasible']:>6} "
              f"{s['cost_total_feas']:>9.1f} {s['cost_compute_feas']:>8.1f} {s['cost_comm_feas']:>7.1f} "
              f"{s['cost_reserve_feas']:>7.1f} {s['cost_recovery_feas']:>6.2f} {s['safety_margin_feas']:>7.2f} "
              f"{s['first_feasible_iter']:>6.1f} {s['runtime']:>8.2f}")

# ---- MAIN three-scale ----
main = load("main_small") + load("main_medium") + load("main_large")
ALGO_ORDER = ["HEFT","PEFT","No-FT","Reactive-Only","Full-Proactive",
              "Critical-Path Backup","Criticality-Based Hybrid","Reliability-Aware",
              "ALNS","GCN-ALNS","RECR-D","HFWPF","GA-Placement","RA-MOMA"]
print("\n########## MAIN (small+medium+large, new cost model) ##########")
print(f"total records: {len(main)}")
print_table("By algorithm (three-scale global)", agg_by(main, lambda r: r["algorithm"]), ALGO_ORDER)

for scale in ["small","medium","large"]:
    sub = [r for r in main if r["scale"] == scale]
    print_table(f"By algorithm @ scale={scale}", agg_by(sub, lambda r: r["algorithm"]), ALGO_ORDER)

for scen in ["tight","medium","loose"]:
    sub = [r for r in main if r["sla_scenario"] == scen]
    print_table(f"By algorithm @ scenario={scen}", agg_by(sub, lambda r: r["algorithm"]), ALGO_ORDER)

# GCN vs ALNS scale x scenario feasibility
print("\n===== GCN-ALNS vs ALNS: feasible% by scale x scenario =====")
for scale in ["small","medium","large"]:
    for scen in ["tight","medium","loose"]:
        for algo in ["ALNS","GCN-ALNS"]:
            sub = [r for r in main if r["scale"]==scale and r["sla_scenario"]==scen and r["algorithm"]==algo]
            if sub:
                fr = mean([1.0 if r["feasible"] else 0.0 for r in sub])
                print(f"  {scale:<7} {scen:<7} {algo:<9} feas%={fr*100:5.1f} (n={len(sub)})")

# ---- ABLATION ----
print("\n\n########## ABLATION (small+medium) ##########")
abl = {}
for p in glob.glob(os.path.join(RAW, "ablation_*.jsonl")):
    name = os.path.basename(p)[len("ablation_"):-len(".jsonl")]
    recs = load(f"ablation_{name}") if False else None
    with open(p, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    abl[name] = recs
print_table("Ablation variants", abl, order=["Full"]+sorted([k for k in abl if k!="Full"]))
# medium-only discriminative
print("\n----- Ablation @ medium scale only -----")
abl_m = {k:[r for r in v if r["scale"]=="medium"] for k,v in abl.items()}
print_table("Ablation @ medium", abl_m, order=["Full"]+sorted([k for k in abl_m if k!="Full"]))

# ---- BUDGET ----
print("\n\n########## BUDGET sweep (small) ##########")
bud = {}
for lvl in ["50","100","200","500","1000"]:
    bud[lvl] = load(f"budget_{lvl}")
print_table("Budget levels (all algos mixed)", bud, order=["50","100","200","500","1000"])
print("\n----- Budget: GCN-ALNS only -----")
budg = {k:[r for r in v if r["algorithm"]=="GCN-ALNS"] for k,v in bud.items()}
print_table("Budget GCN-ALNS", budg, order=["50","100","200","500","1000"])
print("\n----- Budget: ALNS only -----")
buda = {k:[r for r in v if r["algorithm"]=="ALNS"] for k,v in bud.items()}
print_table("Budget ALNS", buda, order=["50","100","200","500","1000"])

# ---- MODEL A/B ----
print("\n\n########## MODEL A/B ##########")
mab = load("model_ab")
print_table("model_ab by algorithm", agg_by(mab, lambda r: r["algorithm"]))

# ---- HARD tiers ----
print("\n\n########## HARD tiers ##########")
for tag in ["main_hard_small","main_hard_medium","main_hard_large"]:
    recs = load(tag)
    if not recs:
        print(f"\n[{tag}] (no data)")
        continue
    print(f"\n[{tag}] records={len(recs)}")
    for scen in ["tight2","tight3"]:
        for algo in ["ALNS","GCN-ALNS"]:
            sub = [r for r in recs if r["sla_scenario"]==scen and r["algorithm"]==algo]
            if sub:
                fr = mean([1.0 if r["feasible"] else 0.0 for r in sub])
                feas=[r for r in sub if r["feasible"]]
                ct = mean([r["cost_total"] for r in feas])
                sm = mean([r["safety_margin"] for r in feas])
                print(f"  {scen:<7} {algo:<9} feas%={fr*100:5.1f} cost_feas={ct:8.1f} safety={sm:6.2f} (n={len(sub)})")
