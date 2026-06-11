"""Plotting (Spec sec.19, figures). Reads aggregated CSVs only, writes PNGs.

Each plot is guarded so a missing CSV / column simply skips that figure rather
than aborting the whole run. Produces the standard comparison figures:
per-algorithm cost / feasibility / recovery / safety-margin / backup-ratio /
runtime bars, an SLA-violation-by-scenario grouped bar, a stacked cost
breakdown, and a runtime-vs-scale scalability scatter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


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


def plot_overview(csv_dir: Path, out_dir: Path) -> None:
    by_algo = _load(csv_dir, "summary_by_algorithm")
    specs = [
        ("cost_total_mean", "Total cost", "Total cost by algorithm", "fig_cost_total", False),
        ("feasible_mean", "Feasible ratio", "SLA-feasible solution ratio", "fig_feasible_ratio", False),
        ("sla_violation_mean", "SLA violation rate", "SLA violation rate", "fig_sla_violation", False),
        ("r_max_mean", "Worst-case recovery delay", "Worst recovery delay (R_max)", "fig_r_max", False),
        ("avg_recovery_mean", "Average recovery delay", "Average recovery delay", "fig_avg_recovery", False),
        ("safety_margin_mean", "Safety margin", "Recovery safety margin", "fig_safety_margin", False),
        ("active_backup_ratio_mean", "Active-backup ratio", "Active-backup ratio", "fig_backup_ratio", False),
        ("c_base_mean", "No-fault makespan", "No-fault makespan", "fig_makespan_base", False),
        ("c_worst_mean", "Worst-case makespan", "Worst-case post-fault makespan", "fig_makespan_worst", False),
        ("runtime_mean", "Runtime (s)", "Scheduling runtime", "fig_runtime", True),
        ("verify_time_mean", "Verify time (s)", "Recovery-verification overhead", "fig_verify_time", True),
        ("gcn_infer_time_mean", "GCN inference (s)", "GCN inference time", "fig_gcn_infer", False),
        ("convergence_iter_mean", "Convergence iteration", "ALNS convergence iteration", "fig_convergence", False),
        ("prune_ratio_mean", "Prune ratio", "Candidate prune ratio", "fig_prune_ratio", False),
        ("n_faults_mean", "|Omega|", "Controllable fault-set size", "fig_fault_set", False),
    ]
    for col, ylabel, title, name, logy in specs:
        _bar_by_algorithm(by_algo, col, ylabel, title, out_dir, name, logy)


def plot_cost_breakdown(csv_dir: Path, out_dir: Path) -> None:
    df = _load(csv_dir, "summary_by_algorithm")
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
    df = _load(csv_dir, "summary_by_algo_scenario")
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
    runs = _load(csv_dir, "runs")
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


def make_all(csv_dir: str | Path, out_dir: str | Path) -> None:
    csv_dir, out_dir = Path(csv_dir), Path(out_dir)
    plot_overview(csv_dir, out_dir)
    plot_cost_breakdown(csv_dir, out_dir)
    plot_sla_by_scenario(csv_dir, out_dir)
    plot_scalability(csv_dir, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render experiment figures from CSV")
    ap.add_argument("--csv", default="results/csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    make_all(args.csv, args.out)


if __name__ == "__main__":
    main()
