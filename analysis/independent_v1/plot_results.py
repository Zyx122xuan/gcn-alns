"""Export scientific figures exclusively from the audited derived metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
DATA = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
SCALES = ("small", "medium", "large")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained")
for ax, metric, title, xlabel in (
    (axes[0], "feasible", "SLA-feasible rate: GCN-ALNS minus ALNS", "Paired difference (percentage points)"),
    (axes[1], "cost_total", "Cost on matched successful runs", "Paired relative cost difference (%)"),
):
    rows = [DATA["main_paired"][scale + "/" + metric] for scale in SCALES]
    means = np.array([r["mean_difference"] for r in rows])
    low, high = np.array([r["ci95"] for r in rows]).T
    yy = np.arange(3)
    ax.errorbar(means, yy, xerr=[means - low, high - means], fmt="o", color="#235d81", capsize=4)
    ax.axvline(0, color="0.5", linestyle="--", linewidth=1)
    ax.set_yticks(yy, [f"{scale.capitalize()} (n={r['n_instances']})" for scale, r in zip(SCALES, rows)])
    ax.invert_yaxis()
    ax.set_ylim(2.7, -.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Problem scale (paired instances)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", color="0.9", linewidth=.7)
    for y, r in zip(yy, rows):
        ax.annotate(f"p={r['p_t']:.3g}", (r["mean_difference"], y), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9)
fig.suptitle("Points: instance means; error bars: 95% instance-bootstrap intervals", fontsize=11)
fig.supxlabel("Source: independent_v1/main_{small,medium,large}; completed 2026-11-04 (CST). Two-sided paired t-test p shown.", fontsize=8)
save(fig, "main_paired_effects")

fig, ax = plt.subplots(figsize=(7.4, 4.4), layout="constrained")
budgets = [50, 100, 200, 500, 1000]
for alg, color, marker in (("ALNS", "0.4", "s"), ("GCN-ALNS", "#235d81", "o")):
    ax.plot(budgets, [DATA["budgets"][str(b)][alg]["feasible_percent"] for b in budgets],
            marker=marker, color=color, label=alg)
ax.set_xticks(budgets)
ax.set_ylim(0, 40)
ax.set_xlabel("Maximum iteration budget (iterations; linear axis)")
ax.set_ylabel("SLA-feasible rate (instance macro mean, %)")
ax.set_title("Small instances: feasibility versus iteration budget")
ax.legend(frameon=False)
ax.grid(axis="y", color="0.9")
fig.supxlabel("Source: independent_v1/budget_*; 45 instances x 10 seeds per solver/budget. Descriptive curves; no significant paired feasibility effect at any budget.", fontsize=8)
save(fig, "budget_feasibility")

fig, ax = plt.subplots(figsize=(7.4, 4.4), layout="constrained")
tasks = [50, 150, 500]
for alg, color, marker in (("ALNS", "0.4", "s"), ("GCN-ALNS", "#235d81", "o")):
    ax.plot(tasks, [DATA["main"][scale][alg]["runtime_mean_s"] for scale in SCALES],
            marker=marker, color=color, label=alg)
ax.set_yscale("log")
ax.set_xticks(tasks)
ax.set_xlabel("Tasks / resource nodes per instance (count)")
ax.set_ylabel("Full solver runtime (mean seconds, logarithmic scale)")
ax.set_title("Main comparison only: complete solver runtime")
ax.legend(frameon=False)
ax.grid(axis="y", color="0.9")
fig.supxlabel("Source: independent_v1/main_* only; 10 seeds per instance for small/medium, 5 for large. Main limits: 1000 / 1000 / 800 iterations. Training time excluded.", fontsize=8)
save(fig, "runtime_main_only")
print("Exported three audited figures, each as PNG and editable SVG.")
