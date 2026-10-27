"""Extra advisor-report figures from raw_json (read-only). English labels.
Adds: feasibility-by-scale, convergence(ttf)-by-scale, model A/B, hard-tier."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW = "results/raw_json"; OUT = "results/figures_report"
os.makedirs(OUT, exist_ok=True)
C_GCN, C_ALNS = "#d62728", "#1f77b4"

def load(tag):
    p = os.path.join(RAW, f"{tag}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else float("nan")

main = load("main_small") + load("main_medium") + load("main_large")
scales = ["small", "medium", "large"]

def feas(algo, sc):
    sub = [r for r in main if r["algorithm"] == algo and r["scale"] == sc]
    return 100*mean([1.0 if r["feasible"] else 0.0 for r in sub]) if sub else float("nan")
def ttf(algo, sc):
    sub = [r for r in main if r["algorithm"] == algo and r["scale"] == sc
           and r.get("first_feasible_iter", -1) >= 0]
    return mean([r["first_feasible_iter"] for r in sub])

# ---- Fig A: feasibility by scale (GCN vs ALNS) ----
fig, ax = plt.subplots(figsize=(6.5, 4.6)); x = np.arange(3); w = 0.38
g = [feas("GCN-ALNS", s) for s in scales]; a = [feas("ALNS", s) for s in scales]
ax.bar(x-w/2, g, w, label="GCN-ALNS", color=C_GCN)
ax.bar(x+w/2, a, w, label="ALNS", color=C_ALNS)
for i,(gv,av) in enumerate(zip(g,a)):
    ax.text(i-w/2, gv+1, f"{gv:.1f}", ha="center", fontsize=8)
    ax.text(i+w/2, av+1, f"{av:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(scales); ax.set_ylabel("SLA-feasible rate (%)")
ax.set_ylim(0, 100); ax.set_title("Feasible rate by scale (GCN-ALNS vs ALNS)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_feasible_by_scale.png", dpi=150); plt.close()

# ---- Fig B: convergence (first-feasible iter) by scale ----
fig, ax = plt.subplots(figsize=(6.5, 4.6))
g = [ttf("GCN-ALNS", s) for s in scales]; a = [ttf("ALNS", s) for s in scales]
ax.bar(x-w/2, g, w, label="GCN-ALNS", color=C_GCN)
ax.bar(x+w/2, a, w, label="ALNS", color=C_ALNS)
for i,(gv,av) in enumerate(zip(g,a)):
    ax.text(i-w/2, gv+2, f"{gv:.0f}", ha="center", fontsize=8)
    ax.text(i+w/2, av+2, f"{av:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(scales); ax.set_ylabel("First-feasible iteration (lower=faster)")
ax.set_title("Convergence speed by scale (feasible-only mean)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_ttf_by_scale.png", dpi=150); plt.close()

# ---- Fig C: model A/B (feasibility) ----
mab = load("model_ab")
def mab_feas(algo):
    sub = [r for r in mab if r["algorithm"] == algo]
    return 100*mean([1.0 if r["feasible"] else 0.0 for r in sub]) if sub else float("nan")
labels = ["ALNS\n(no GCN)", "GCN-ALNS\n(A2-off)", "GCN-ALNS\n(A2-on, default)"]
algos = ["ALNS", "GCN-ALNS (A2-off)", "GCN-ALNS (A2-on)"]
vals = [mab_feas(a) for a in algos]
colors = [C_ALNS, "#ff9896", C_GCN]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
b = ax.bar(range(3), vals, color=colors)
for i,v in enumerate(vals): ax.text(i, v+0.4, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("SLA-feasible rate (%)"); ax.set_ylim(0, max(vals)*1.2)
ax.set_title("Model A/B (small): A2 mode-head has no feasibility gain")
ax.annotate("mode-head AUC 0.975", xy=(2, vals[2]), xytext=(1.1, max(vals)*1.12),
            fontsize=8, ha="center")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_model_ab.png", dpi=150); plt.close()

# ---- Fig D: hard tiers (tight2/tight3 x small/medium/large), GCN vs ALNS ----
hs, hm, hl = (load("main_hard_small"), load("main_hard_medium"),
              load("main_hard_large"))
def hfeas(recs, algo, scen):
    sub = [r for r in recs if r["algorithm"] == algo and r["sla_scenario"] == scen]
    return 100*mean([1.0 if r["feasible"] else 0.0 for r in sub]) if sub else float("nan")
def hsafety(recs, algo, scen):
    sub = [r for r in recs if r["algorithm"] == algo and r["sla_scenario"] == scen
           and r["feasible"] and isinstance(r.get("safety_margin"), (int, float))]
    return mean([r["safety_margin"] for r in sub]) if sub else float("nan")
groups = [("small", "tight2", hs), ("small", "tight3", hs),
          ("medium", "tight2", hm), ("medium", "tight3", hm),
          ("large", "tight2", hl), ("large", "tight3", hl)]
glabels = [f"{sc}\n{se}" for sc, se, _ in groups]
xx = np.arange(len(groups))
# left = feasible rate (where GCN wins 4/6), right = safety margin (6/6)
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
for ax, fn, ylab, title in (
        (axes[0], hfeas, "SLA-feasible rate (%)",
         "Hard tiers: feasible rate (GCN >= ALNS in 4/6)"),
        (axes[1], hsafety, "Safety margin (feasible only)",
         "Hard tiers: safety margin (GCN > ALNS in 6/6)")):
    gv = [fn(r, "GCN-ALNS", se) for sc, se, r in groups]
    av = [fn(r, "ALNS", se) for sc, se, r in groups]
    ax.bar(xx-w/2, gv, w, label="GCN-ALNS", color=C_GCN)
    ax.bar(xx+w/2, av, w, label="ALNS", color=C_ALNS)
    top = max([v for v in gv + av if v == v] + [1.0])
    fmt = "{:.0f}" if ax is axes[0] else "{:.2f}"
    for i, (g_, a_) in enumerate(zip(gv, av)):
        ax.text(i-w/2, g_+0.02*top, fmt.format(g_), ha="center", fontsize=8)
        ax.text(i+w/2, a_+0.02*top, fmt.format(a_), ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels(glabels, fontsize=8)
    ax.set_ylabel(ylab); ax.set_ylim(0, top*1.18)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_hardtier.png", dpi=150); plt.close()

print("saved:")
for fn in ["fig_r_feasible_by_scale", "fig_r_ttf_by_scale", "fig_r_model_ab", "fig_r_hardtier"]:
    print(" ", fn+".png")
