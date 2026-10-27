"""Four-panel ablation figure (medium scale): feasibility / cost / safety / convergence.
Read-only w.r.t. the running pipeline: writes to results/figures_report/ only."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "results/raw_json"; OUT = "results/figures_report"
os.makedirs(OUT, exist_ok=True)

def load_abl():
    d = {}
    for fn in os.listdir(RAW):
        if fn.startswith("ablation_") and fn.endswith(".jsonl"):
            name = fn[len("ablation_"):-len(".jsonl")]
            with open(os.path.join(RAW, fn), encoding="utf-8") as f:
                d[name] = [json.loads(l) for l in f if l.strip()]
    return d

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else float("nan")

abl = load_abl()
med = {k: [r for r in v if r["scale"] == "medium"] for k, v in abl.items()}

def feas(v):   return mean([1.0 if r.get("feasible") else 0.0 for r in v]) * 100
def cost(v):   return mean([r["cost_total"] for r in v if r.get("feasible")])
def safety(v): return mean([r["safety_margin"] for r in v if r.get("feasible")])
def ttf(v):    return mean([r["first_feasible_iter"] for r in v if r.get("first_feasible_iter", -1) >= 0])

# unified order: feasibility descending (same row order across all 4 panels)
order = sorted(med.keys(), key=lambda k: feas(med[k]), reverse=True)
full_feas = feas(med["Full"])

fig, axes = plt.subplots(1, 4, figsize=(20, 6.2), sharey=True)

# Panel 1: feasibility (highlight Full green; the two pillars that drop >4pt red)
vals = [feas(med[k]) for k in order]
colors = ["#2ca02c" if n == "Full" else ("#c44e52" if v < full_feas - 4 else "#9aa0a6")
          for n, v in zip(order, vals)]
axes[0].barh(range(len(order)), vals, color=colors)
axes[0].axvline(full_feas, ls="--", color="#2ca02c", lw=1, label=f"Full = {full_feas:.1f}%")
axes[0].set_xlabel("SLA-feasible rate @ medium (%)"); axes[0].legend(fontsize=8, loc="lower right")
axes[0].set_title("(a) Feasibility  [higher better]", fontsize=10)

# Panel 2: cost on feasible (lower better) -- Full green
vals = [cost(med[k]) for k in order]
colors = ["#2ca02c" if n == "Full" else "#dd8452" for n in order]
axes[1].barh(range(len(order)), vals, color=colors)
axes[1].axvline(cost(med["Full"]), ls="--", color="#2ca02c", lw=1)
axes[1].set_xlabel("Cost on feasible solutions"); axes[1].set_xlim(left=min(vals)*0.97)
axes[1].set_title("(b) Cost  [lower better]", fontsize=10)

# Panel 3: safety margin (feasible-only)
vals = [safety(med[k]) for k in order]
colors = ["#2ca02c" if n == "Full" else "#55a868" for n in order]
axes[2].barh(range(len(order)), vals, color=colors)
axes[2].axvline(safety(med["Full"]), ls="--", color="#2ca02c", lw=1)
axes[2].set_xlabel("Safety margin (feasible-only)")
axes[2].set_title("(c) Safety margin  [feasible-only*]", fontsize=10)

# Panel 4: first-feasible iter (convergence; lower = faster)
vals = [ttf(med[k]) for k in order]
colors = ["#2ca02c" if n == "Full" else "#8172b3" for n in order]
axes[3].barh(range(len(order)), vals, color=colors)
axes[3].axvline(ttf(med["Full"]), ls="--", color="#2ca02c", lw=1)
axes[3].set_xlabel("First-feasible iteration")
axes[3].set_title("(d) Convergence  [lower=faster, feasible-only*]", fontsize=10)

axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=8)
axes[0].invert_yaxis()
for ax in axes:
    xvals = [b.get_width() for b in ax.patches]
    for i, v in enumerate(xvals):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=6.5)

fig.suptitle("Ablation (medium scale, GCN-ALNS, repeats=10): feasibility / cost / safety / convergence\n"
             "*panels (c)(d) are feasible-only means -> compare only across variants with similar feasibility",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
path = f"{OUT}/fig_r_ablation_4panel.png"
plt.savefig(path, dpi=150); plt.close()
print("saved", path)
