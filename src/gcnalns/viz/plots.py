"""Plotting (Spec sec.19, figures). Reads aggregated CSVs / runs.csv, writes PNGs.

Design notes (addresses the four experiment issues):

* Dimensional metrics (cost / makespan / recovery / runtime ...) are plotted
  **per scale** (small/medium/large) from ``summary_by_algo_scale.csv`` so the
  small(50)/medium(150)/large(500) magnitudes are never mixed into one mean.
* Dimensionless metrics (feasible / SLA-violation / backup-ratio / prune-ratio)
  stay as a single aggregated bar.
* A task-normalised cost figure (``cost_total / num_tasks``) gives a single
  cross-scale comparable view.
* Recovery-quality figures use the **feasible-only** aggregates
  (``*_feas_mean``) so infeasible INF/huge R_max no longer pull the mean
  negative.
* Main comparison figures keep only the core algorithms; ablation / sensitivity
  / budget families are read from their own tagged runs.

Each plot is guarded so a missing CSV / column simply skips that figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SCALE_ORDER = ["small", "medium", "large"]


def _load(csv_dir: Path, name: str) -> pd.DataFrame | None:
    p = csv_dir / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if not df.empty else None


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def _is_core(algo) -> bool:
    """Core (main) algorithms carry no ``[...]`` ablation/sensitivity suffix."""
    return "[" not in str(algo)


def _core(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or "algorithm" not in df.columns:
        return df
    out = df[df["algorithm"].map(_is_core)]
    return out if not out.empty else None


def _scale_sorted(values) -> list:
    vals = list(dict.fromkeys(values))
    known = [s for s in _SCALE_ORDER if s in vals]
    rest = [v for v in vals if v not in _SCALE_ORDER]
    return known + rest


# ---------------------------------------------------------------------------
# Dimensionless single-bar metrics
# ---------------------------------------------------------------------------
def _bar_by_algorithm(df, mean_col, ylabel, title, out_dir, name, logy=False):
    if df is None or "algorithm" not in df.columns or mean_col not in df.columns:
        return
    d = df.sort_values(mean_col)
    fig, ax = plt.subplots(figsize=(9, 5))
    std_col = mean_col.replace("_mean", "_std")
    yerr = d[std_col] if std_col in d.columns else None
    ax.bar(d["algorithm"], d[mean_col], yerr=yerr, capsize=3, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_dir, name)


def plot_dimensionless(csv_dir: Path, out_dir: Path) -> None:
    by_algo = _core(_load(csv_dir, "summary_by_algorithm"))
    specs = [
        ("feasible_mean", "Feasible ratio", "SLA-feasible solution ratio", "fig_feasible_ratio", False),
        ("sla_violation_mean", "SLA violation rate", "SLA violation rate", "fig_sla_violation", False),
        ("active_backup_ratio_mean", "Active-backup ratio", "Active-backup ratio", "fig_backup_ratio", False),
        ("prune_ratio_mean", "Prune ratio", "Candidate prune ratio", "fig_prune_ratio", False),
        ("gcn_infer_time_mean", "GCN inference (s)", "GCN inference time", "fig_gcn_infer", False),
        ("convergence_iter_mean", "Convergence iteration", "ALNS convergence iteration", "fig_convergence", False),
    ]
    for col, ylabel, title, name, logy in specs:
        _bar_by_algorithm(by_algo, col, ylabel, title, out_dir, name, logy)


# ---------------------------------------------------------------------------
# Dimensional metrics: one grouped bar per metric, hue = scale
# ---------------------------------------------------------------------------
def _grouped_bar_by_scale(df, value_col, ylabel, title, out_dir, name, logy=False):
    if df is None or "scale" not in df.columns or value_col not in df.columns:
        return
    d = df.dropna(subset=[value_col])
    if d.empty:
        return
    pivot = d.pivot_table(index="algorithm", columns="scale", values=value_col)
    scales = _scale_sorted(pivot.columns.tolist())
    pivot = pivot[scales]
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend(title="scale", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_dir, name)


def plot_by_scale(csv_dir: Path, out_dir: Path) -> None:
    by_scale = _core(_load(csv_dir, "summary_by_algo_scale"))
    if by_scale is None:
        return
    # prefer feasible-only aggregate where it exists (recovery/cost metrics)
    specs = [
        ("cost_total_feas_mean", "cost_total_mean", "Total cost (feasible)",
         "Total cost by scale (feasible only)", "fig_cost_total_by_scale", False),
        ("c_base_mean", "c_base_mean", "No-fault makespan",
         "No-fault makespan by scale", "fig_makespan_base_by_scale", False),
        ("c_worst_feas_mean", "c_worst_mean", "Worst-case makespan",
         "Worst-case makespan by scale (feasible only)", "fig_makespan_worst_by_scale", False),
        ("r_max_feas_mean", "r_max_mean", "Worst recovery delay (R_max)",
         "Worst recovery delay by scale (feasible only)", "fig_r_max_by_scale", False),
        ("avg_recovery_feas_mean", "avg_recovery_mean", "Average recovery delay",
         "Average recovery delay by scale (feasible only)", "fig_avg_recovery_by_scale", False),
        ("n_faults_mean", "n_faults_mean", "|Omega|",
         "Controllable fault-set size by scale", "fig_fault_set_by_scale", False),
        ("runtime_mean", "runtime_mean", "Runtime (s)",
         "Scheduling runtime by scale", "fig_runtime_by_scale", True),
        ("verify_time_mean", "verify_time_mean", "Verify time (s)",
         "Recovery-verification time by scale", "fig_verify_time_by_scale", True),
    ]
    for pref, fallback, ylabel, title, name, logy in specs:
        col = pref if pref in by_scale.columns else fallback
        _grouped_bar_by_scale(by_scale, col, ylabel, title, out_dir, name, logy)


def plot_safety_margin(csv_dir: Path, out_dir: Path) -> None:
    by_scale = _core(_load(csv_dir, "summary_by_algo_scale"))
    col = "safety_margin_feas_mean"
    if by_scale is not None and col in by_scale.columns:
        _grouped_bar_by_scale(by_scale, col, "Safety margin (feasible)",
                              "Recovery safety margin by scale (feasible only)",
                              out_dir, "fig_safety_margin_by_scale")
    # also a single aggregated feasible-only bar
    by_algo = _core(_load(csv_dir, "summary_by_algorithm"))
    if by_algo is not None and "safety_margin_feas_mean" in by_algo.columns:
        _bar_by_algorithm(by_algo, "safety_margin_feas_mean", "Safety margin (feasible)",
                          "Recovery safety margin (feasible solutions only)",
                          out_dir, "fig_safety_margin")


def plot_cost_normalized(csv_dir: Path, out_dir: Path) -> None:
    """Task-normalised cost (cost_total / num_tasks), feasible runs, by scale."""
    runs = _core(_load(csv_dir, "runs"))
    if runs is None or "cost_total" not in runs.columns or "num_tasks" not in runs.columns:
        return
    d = runs.copy()
    if "feasible" in d.columns:
        d = d[d["feasible"].astype(bool)]
    d = d[np.isfinite(pd.to_numeric(d["cost_total"], errors="coerce"))]
    d = d[d["num_tasks"] > 0]
    if d.empty:
        return
    d["cost_per_task"] = d["cost_total"] / d["num_tasks"]
    grp_keys = ["algorithm", "scale"] if "scale" in d.columns else ["algorithm"]
    norm = d.groupby(grp_keys)["cost_per_task"].mean().reset_index()
    if "scale" in norm.columns:
        pivot = norm.pivot_table(index="algorithm", columns="scale", values="cost_per_task")
        pivot = pivot[_scale_sorted(pivot.columns.tolist())]
        fig, ax = plt.subplots(figsize=(11, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.legend(title="scale", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(norm["algorithm"], norm["cost_per_task"], color="#4C72B0")
    ax.set_ylabel("Cost per task")
    ax.set_title("Task-normalised cost (feasible solutions)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_dir, "fig_cost_per_task")


def plot_cost_breakdown(csv_dir: Path, out_dir: Path) -> None:
    df = _core(_load(csv_dir, "summary_by_algorithm"))
    if df is None or "algorithm" not in df.columns:
        return
    parts = ["cost_compute_mean", "cost_backup_mean", "cost_comm_mean",
             "cost_reserve_mean", "cost_recovery_mean"]
    parts = [p for p in parts if p in df.columns]
    if not parts:
        return
    d = df.sort_values("cost_total_mean") if "cost_total_mean" in df.columns else df
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0.0] * len(d)
    labels = [p.replace("cost_", "").replace("_mean", "") for p in parts]
    for p, lab in zip(parts, labels):
        ax.bar(d["algorithm"], d[p], bottom=bottom, label=lab)
        bottom = [b + v for b, v in zip(bottom, d[p])]
    ax.set_ylabel("Cost")
    ax.set_title("Cost breakdown by algorithm")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_dir, "fig_cost_breakdown")


def plot_sla_by_scenario(csv_dir: Path, out_dir: Path) -> None:
    df = _core(_load(csv_dir, "summary_by_algo_scenario"))
    if df is None or "sla_violation_mean" not in df.columns:
        return
    pivot = df.pivot_table(index="algorithm", columns="sla_scenario",
                           values="sla_violation_mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("SLA violation rate")
    ax.set_title("SLA violation rate by algorithm and scenario")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_dir, "fig_sla_by_scenario")


def plot_scalability(csv_dir: Path, out_dir: Path) -> None:
    runs = _core(_load(csv_dir, "runs"))
    if runs is None or "num_tasks" not in runs.columns or "runtime" not in runs.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, grp in runs.groupby("algorithm"):
        g = grp.groupby("num_tasks")["runtime"].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=algo)
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.set_title("Scheduling runtime vs. problem size")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_scalability_runtime")


# ---------------------------------------------------------------------------
# Ablation / sensitivity / budget families (from runs.csv via tag/label)
# ---------------------------------------------------------------------------
def _runs_by_tag_prefix(runs: pd.DataFrame, prefix: str) -> pd.DataFrame | None:
    if runs is None or "tag" not in runs.columns:
        return None
    sub = runs[runs["tag"].astype(str).str.startswith(prefix)]
    return sub if not sub.empty else None


def _feasible_finite(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df
    if "feasible" in d.columns:
        d = d[d["feasible"].astype(bool)]
    if col in d.columns:
        d = d[np.isfinite(pd.to_numeric(d[col], errors="coerce"))]
    return d


def plot_ablation(csv_dir: Path, out_dir: Path) -> None:
    runs = _load(csv_dir, "runs")
    sub = _runs_by_tag_prefix(runs, "ablation")
    if sub is None:
        return
    feas = sub.groupby("algorithm")["feasible"].mean()
    cost = _feasible_finite(sub, "cost_total").groupby("algorithm")["cost_total"].mean()
    order = list(feas.sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar([a.replace("GCN-ALNS ", "") for a in order],
                [feas[a] for a in order], color="#55A868")
    axes[0].set_ylabel("Feasible ratio")
    axes[0].set_title("Ablation: SLA-feasible ratio")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
    corder = [a for a in order if a in cost.index]
    axes[1].bar([a.replace("GCN-ALNS ", "") for a in corder],
                [cost[a] for a in corder], color="#C44E52")
    axes[1].set_ylabel("Mean cost (feasible)")
    axes[1].set_title("Ablation: cost on feasible solutions")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_dir, "fig_ablation")


def plot_sensitivity(csv_dir: Path, out_dir: Path) -> None:
    runs = _load(csv_dir, "runs")
    sub = _runs_by_tag_prefix(runs, "sens")
    if sub is None:
        return
    # tag form: sens_<param_with_underscores>_<value>
    def parse(tag: str):
        body = tag[len("sens_"):]
        param, _, val = body.rpartition("_")
        try:
            return param, float(val)
        except ValueError:
            return param, None
    parsed = sub["tag"].astype(str).map(parse)
    sub = sub.assign(param=[p for p, _ in parsed], pval=[v for _, v in parsed])
    sub = sub.dropna(subset=["pval"])
    if sub.empty:
        return
    for param, grp in sub.groupby("param"):
        fig, ax = plt.subplots(figsize=(8, 5))
        feas = grp.groupby("pval")["feasible"].mean().sort_index()
        ax.plot(feas.index, feas.values, marker="o", color="#4C72B0", label="feasible ratio")
        ax.set_xlabel(param)
        ax.set_ylabel("Feasible ratio")
        cfin = _feasible_finite(grp, "cost_total")
        if not cfin.empty:
            ax2 = ax.twinx()
            cost = cfin.groupby("pval")["cost_total"].mean().sort_index()
            ax2.plot(cost.index, cost.values, marker="s", color="#C44E52", label="cost (feasible)")
            ax2.set_ylabel("Mean cost (feasible)")
        ax.set_title(f"Sensitivity: {param}")
        _save(fig, out_dir, f"fig_sensitivity_{param}")


def plot_budget(csv_dir: Path, out_dir: Path) -> None:
    """Convergence / limited-budget comparison: ALNS vs GCN-ALNS over max-iter."""
    runs = _load(csv_dir, "runs")
    sub = _runs_by_tag_prefix(runs, "budget")
    if sub is None:
        return

    def budget_iter(tag: str):
        try:
            return int(str(tag).split("_")[-1])
        except ValueError:
            return None
    sub = sub.assign(budget=sub["tag"].map(budget_iter)).dropna(subset=["budget"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for algo, grp in sub.groupby("algorithm"):
        feas = grp.groupby("budget")["feasible"].mean().sort_index()
        axes[0].plot(feas.index, feas.values, marker="o", label=algo)
        cfin = _feasible_finite(grp, "cost_total")
        if not cfin.empty:
            cost = cfin.groupby("budget")["cost_total"].mean().sort_index()
            axes[1].plot(cost.index, cost.values, marker="s", label=algo)
    axes[0].set_xlabel("Iteration budget")
    axes[0].set_ylabel("Feasible ratio")
    axes[0].set_title("Feasibility vs. iteration budget")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("Iteration budget")
    axes[1].set_ylabel("Mean cost (feasible)")
    axes[1].set_title("Cost vs. iteration budget (feasible)")
    axes[1].legend(fontsize=9)
    _save(fig, out_dir, "fig_budget_convergence")


def make_all(csv_dir: str | Path, out_dir: str | Path) -> None:
    csv_dir, out_dir = Path(csv_dir), Path(out_dir)
    plot_dimensionless(csv_dir, out_dir)
    plot_by_scale(csv_dir, out_dir)
    plot_safety_margin(csv_dir, out_dir)
    plot_cost_normalized(csv_dir, out_dir)
    plot_cost_breakdown(csv_dir, out_dir)
    plot_sla_by_scenario(csv_dir, out_dir)
    plot_scalability(csv_dir, out_dir)
    plot_ablation(csv_dir, out_dir)
    plot_sensitivity(csv_dir, out_dir)
    plot_budget(csv_dir, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render experiment figures from CSV")
    ap.add_argument("--csv", default="results/csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    make_all(args.csv, args.out)


if __name__ == "__main__":
    main()
