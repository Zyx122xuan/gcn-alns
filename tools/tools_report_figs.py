"""Generate advisor-report figures from raw_json (main three-scale, new cost model).
Read-only w.r.t. the running pipeline: writes to results/figures_report/ only.
English labels (matches existing viz style; avoids CJK font issues)."""
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW = "results/raw_json"
OUT = "results/figures_report"
os.makedirs(OUT, exist_ok=True)

def load(tag):
    p = os.path.join(RAW, f"{tag}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else float("nan")

def feas_rate(recs):
    return mean([1.0 if r.get("feasible") else 0.0 for r in recs])

main = load("main_small") + load("main_medium") + load("main_large")

C_GCN, C_ALNS = "#d62728", "#1f77b4"
C_OTHER = "#9aa0a6"

# ---------- Fig 1: feasibility by algorithm (global) ----------
g = defaultdict(list)
for r in main: g[r["algorithm"]].append(r)
# sort by feasible rate so the bars read monotonically
algos = sorted(g, key=lambda a: feas_rate(g[a]), reverse=True)
vals = [feas_rate(g[a])*100 for a in algos]
colors = [C_GCN if a=="GCN-ALNS" else C_ALNS if a=="ALNS" else C_OTHER for a in algos]
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(range(len(algos)), vals, color=colors)
ax.set_xticks(range(len(algos))); ax.set_xticklabels(algos, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("SLA-feasible rate (%)"); ax.set_title("Feasible rate by algorithm (all scales, corrected cost model)")
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.1f}", ha="center", fontsize=7)
ax.set_ylim(0, max(vals)*1.15)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_feasible_by_algo.png", dpi=150); plt.close()

# ---------- Fig 2: GCN vs ALNS feasibility by scale x scenario ----------
scales=["small","medium","large"]; scens=["tight","medium","loose"]
def fr(algo, sc, se):
    sub=[r for r in main if r["algorithm"]==algo and r["scale"]==sc and r["sla_scenario"]==se]
    return feas_rate(sub)*100 if sub else 0.0
fig, axes = plt.subplots(1,3, figsize=(13,4.2), sharey=True)
x=np.arange(len(scens)); w=0.38
for ax,sc in zip(axes,scales):
    gcn=[fr("GCN-ALNS",sc,se) for se in scens]
    aln=[fr("ALNS",sc,se) for se in scens]
    ax.bar(x-w/2, gcn, w, label="GCN-ALNS", color=C_GCN)
    ax.bar(x+w/2, aln, w, label="ALNS", color=C_ALNS)
    ax.set_title(f"scale = {sc}"); ax.set_xticks(x); ax.set_xticklabels(scens)
    for i,(gv,av) in enumerate(zip(gcn,aln)):
        ax.text(i-w/2, gv+1, f"{gv:.0f}", ha="center", fontsize=7)
        ax.text(i+w/2, av+1, f"{av:.0f}", ha="center", fontsize=7)
axes[0].set_ylabel("SLA-feasible rate (%)"); axes[0].legend(fontsize=9)
axes[0].set_ylim(0,112)
fig.suptitle("GCN-ALNS vs ALNS: feasible rate by scale x SLA scenario")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_gcn_vs_alns_scale_scenario.png", dpi=150); plt.close()

# ---------- Fig 3: cost breakdown GCN vs ALNS (global, feasible-only) ----------
def cost_parts(algo):
    feas=[r for r in main if r["algorithm"]==algo and r.get("feasible")]
    return (mean([r["cost_compute"] for r in feas]), mean([r["cost_comm"] for r in feas]),
            mean([r["cost_reserve"] for r in feas]), mean([r["cost_recovery"] for r in feas]))
parts_labels=["compute","comm","reserve","recovery"]
pc=[cost_parts("GCN-ALNS"), cost_parts("ALNS")]
fig, ax = plt.subplots(figsize=(6.2,5))
xpos=[0,1]; bottom=[0,0]
palette=["#4c72b0","#dd8452","#55a868","#c44e52"]
for j,lab in enumerate(parts_labels):
    vals=[pc[0][j], pc[1][j]]
    ax.bar(xpos, vals, 0.55, bottom=bottom, label=lab, color=palette[j])
    for xi,(v,b) in enumerate(zip(vals,bottom)):
        if v>8: ax.text(xi, b+v/2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
    bottom=[bottom[0]+vals[0], bottom[1]+vals[1]]
totals=[sum(pc[0]), sum(pc[1])]
for xi,t in zip(xpos,totals): ax.text(xi, t+8, f"total {t:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(xpos); ax.set_xticklabels(["GCN-ALNS","ALNS"])
ax.set_ylabel("cost (feasible-only mean)"); ax.set_ylim(0, max(totals)*1.15)
ax.set_title("Cost breakdown (all scales, backup excluded)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_cost_breakdown.png", dpi=150); plt.close()

# ---------- Fig 4: ablation feasibility @ medium ----------
abl={}
for fn in os.listdir(RAW):
    if fn.startswith("ablation_") and fn.endswith(".jsonl"):
        name=fn[len("ablation_"):-len(".jsonl")]
        with open(os.path.join(RAW,fn),encoding="utf-8") as f:
            abl[name]=[json.loads(l) for l in f if l.strip()]
med={k:[r for r in v if r["scale"]=="medium"] for k,v in abl.items()}
items=sorted(((k, feas_rate(v)*100) for k,v in med.items()), key=lambda kv: kv[1])
names=[k for k,_ in items]; vals=[v for _,v in items]
full=feas_rate(med["Full"])*100
colors=["#2ca02c" if n=="Full" else ("#c44e52" if v<full-4 else C_OTHER) for n,v in zip(names,vals)]
fig, ax = plt.subplots(figsize=(8.5,6))
ax.barh(range(len(names)), vals, color=colors)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.axvline(full, ls="--", color="#2ca02c", lw=1, label=f"Full = {full:.1f}%")
for i,v in enumerate(vals): ax.text(v+0.4, i, f"{v:.1f}", va="center", fontsize=7)
ax.set_xlabel("SLA-feasible rate @ medium (%)"); ax.set_title("Ablation (medium scale, discriminative)")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_ablation_medium.png", dpi=150); plt.close()

# ---------- Fig 5: budget convergence ----------
levels=["50","100","200","500","1000"]
def bfr(lvl, algo):
    recs=[r for r in load(f"budget_{lvl}") if r["algorithm"]==algo]
    return feas_rate(recs)*100
gcn=[bfr(l,"GCN-ALNS") for l in levels]; aln=[bfr(l,"ALNS") for l in levels]
fig, ax = plt.subplots(figsize=(6.5,4.6))
x=[int(l) for l in levels]
ax.plot(x, gcn, "o-", color=C_GCN, label="GCN-ALNS")
ax.plot(x, aln, "s--", color=C_ALNS, label="ALNS")
for xi,v in zip(x,gcn): ax.text(xi, v+0.6, f"{v:.1f}", ha="center", fontsize=7, color=C_GCN)
for xi,v in zip(x,aln): ax.text(xi, v-1.4, f"{v:.1f}", ha="center", fontsize=7, color=C_ALNS)
ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(levels)
ax.set_xlabel("ALNS iteration budget"); ax.set_ylabel("SLA-feasible rate (%)")
ax.set_title("Feasible rate vs iteration budget (small)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_budget.png", dpi=150); plt.close()

# ---------- Fig 6: safety margin by scale ----------
fig, ax = plt.subplots(figsize=(6.5,4.6))
x=np.arange(len(scales)); w=0.38
def sm(algo, sc):
    feas=[r for r in main if r["algorithm"]==algo and r["scale"]==sc and r.get("feasible")]
    return mean([r["safety_margin"] for r in feas])
gcn=[sm("GCN-ALNS",sc) for sc in scales]; aln=[sm("ALNS",sc) for sc in scales]
ax.bar(x-w/2, gcn, w, label="GCN-ALNS", color=C_GCN)
ax.bar(x+w/2, aln, w, label="ALNS", color=C_ALNS)
for i,(gv,av) in enumerate(zip(gcn,aln)):
    ax.text(i-w/2, gv+0.15, f"{gv:.1f}", ha="center", fontsize=8)
    ax.text(i+w/2, av+0.15, f"{av:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(scales); ax.set_ylabel("safety margin (feasible-only mean)")
ax.set_title("Safety margin by scale"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_safety_by_scale.png", dpi=150); plt.close()

print("Saved figures to", OUT)
for fn in sorted(os.listdir(OUT)): print(" ", fn)
