"""Honest multi-objective scatter for the ablation (medium scale, rep10):
feasibility (y, higher better) vs cost (x, lower better), marker size = safety margin.
Full is starred; the region strictly better than Full (higher feas AND lower cost)
is shaded so any 2D-edge variants are shown truthfully. Read-only."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "results/raw_json"; OUT = "results/figures_report"
os.makedirs(OUT, exist_ok=True)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else float("nan")

rows = {}
for fn in os.listdir(RAW):
    if fn.startswith("ablation_") and fn.endswith(".jsonl"):
        name = fn[len("ablation_"):-len(".jsonl")]
        recs = [json.loads(l) for l in open(os.path.join(RAW, fn), encoding="utf-8") if l.strip()]
        recs = [r for r in recs if r["scale"] == "medium"]
        feas = [r for r in recs if r.get("feasible")]
        rows[name] = dict(
            feas=100*len(feas)/len(recs),
            cost=mean([r["cost_total"] for r in feas]),
            safety=mean([r["safety_margin"] for r in feas]),
        )

full = rows["Full"]
fig, ax = plt.subplots(figsize=(9, 6))

# shade region strictly better than Full: cost < Full.cost AND feas > Full.feas (upper-left)
xmin, xmax = 555, 610
ax.axvspan(xmin, full["cost"], ymin=(full["feas"]-25)/(65-25), ymax=1.0, color="#ffe0e0", alpha=0.5, zorder=0)
ax.axvline(full["cost"], ls=":", color="gray", lw=1)
ax.axhline(full["feas"], ls=":", color="gray", lw=1)
ax.text(full["cost"]-1, 64, "<- cheaper", fontsize=8, ha="right", color="gray")

for name, r in rows.items():
    is_full = (name == "Full")
    # points strictly better than Full on (feas,cost)
    edge = (r["feas"] > full["feas"] + 1e-9) and (r["cost"] < full["cost"] - 1e-9)
    color = "#2ca02c" if is_full else ("#ff7f0e" if edge else "#8899aa")
    size = 60 + r["safety"] * 90  # marker size ~ safety margin
    ax.scatter(r["cost"], r["feas"], s=size, c=color, marker=("*" if is_full else "o"),
               edgecolors="black", linewidths=0.6, zorder=5,
               label=None)
    lbl = name.replace("wo_", "w/o ").replace("_", " ")
    dy = 0.5 if not is_full else -1.3
    ax.annotate(("Full" if is_full else lbl), (r["cost"], r["feas"]),
                xytext=(0, 7 if not is_full else -12), textcoords="offset points",
                fontsize=7.5, ha="center", fontweight=("bold" if is_full or edge else "normal"),
                color=("#2ca02c" if is_full else ("#d95f02" if edge else "black")))

ax.set_xlim(xmin, xmax); ax.set_ylim(25, 65)
ax.set_xlabel("Cost on feasible solutions (lower = better) → left is cheaper")
ax.set_ylabel("SLA-feasible rate @ medium (%)  (higher = better)")
ax.set_title("Ablation multi-objective view (medium, rep10)\n"
             "marker size ∝ safety margin;  pink = region strictly better than Full (higher feas & lower cost)")
# legend proxies
from matplotlib.lines import Line2D
proxies = [
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#2ca02c", markersize=15, markeredgecolor="k", label="Full (balanced)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=10, markeredgecolor="k", label="2D-edges Full on (feas,cost)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#8899aa", markersize=10, markeredgecolor="k", label="other variants"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=7, markeredgecolor="k", label="small marker = low safety"),
]
ax.legend(handles=proxies, fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_r_ablation_pareto.png", dpi=150); plt.close()

# print the honest domination check
print(f"Full: feas={full['feas']:.1f}%  cost={full['cost']:.1f}  safety={full['safety']:.2f}")
print("\nvariants in the 'strictly better than Full on (feas,cost)' region:")
for name, r in rows.items():
    if r["feas"] > full["feas"] and r["cost"] < full["cost"]:
        print(f"  {name:<22} feas={r['feas']:.1f} cost={r['cost']:.1f} safety={r['safety']:.2f} "
              f"(Full safety {full['safety']:.2f})")
