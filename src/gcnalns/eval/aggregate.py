"""Aggregate raw experiment records into summary CSVs (Spec sec.18-20).

Reads the raw per-run JSON written by the experiment runners and produces:

* ``runs.csv``                     : every individual run (flat).
* ``summary_by_algorithm.csv``     : mean/std per algorithm.
* ``summary_by_algo_scenario.csv`` : mean/std per (algorithm, SLA scenario).
* ``summary_by_algo_scale.csv``    : mean/std per (algorithm, scale).

Plots read these CSVs only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import load_json

# Metrics summarised with mean (+std). Feasibility is a rate (mean of 0/1).
_NUMERIC = [
    "feasible", "sla_violation", "c_base", "c_worst", "r_max", "r_eff",
    "avg_recovery", "safety_margin", "n_faults", "active_backup_ratio",
    "cost_total", "cost_compute", "cost_backup", "cost_comm", "cost_reserve",
    "cost_recovery", "runtime", "verify_time", "gcn_infer_time",
    "convergence_iter", "first_feasible_iter", "prune_ratio",
]

# Recovery metrics only meaningful on SLA-feasible solutions: infeasible runs
# carry huge / INF R_max that drag the plain mean negative (issue #4). These get
# an extra ``*_feas_*`` aggregation restricted to feasible, finite-valued runs.
_FEASIBLE_ONLY = ["safety_margin", "r_max", "avg_recovery", "c_worst",
                  "cost_total", "first_feasible_iter"]

# The by-algorithm/scale/scenario summary CSVs describe the *main comparison*
# only. Other studies (budget sweep, hard-tier, model A/B, ablation, ...) reuse
# the plain ALNS / GCN-ALNS labels or add suffixes and would otherwise pollute
# these summaries; they are analysed separately from ``runs.csv`` via their own
# tags. ``runs.csv`` itself always keeps every run.
_MAIN_TAGS = {"main", "main_small", "main_medium", "main_large"}


def load_runs(raw_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all raw run records under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    records: list[dict] = []
    for jf in sorted(raw_dir.glob("*.json")):
        data = load_json(jf)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict) and "runs" in data:
            records.extend(data["runs"])
        else:
            records.append(data)
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    cols = [c for c in _NUMERIC if c in df.columns]
    grouped = df.groupby(group_keys, dropna=False)[cols]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()

    # feasible-only aggregation for recovery / cost metrics (issue #4).
    # Non-finite values (e.g. INF r_max) are turned into NaN and skipped
    # *per column* rather than dropping whole rows, so a column that is simply
    # missing for some runs (e.g. first_feasible_iter on pre-instrumentation
    # runs) does not wipe out the other feasible-only metrics.
    if "feasible" in df.columns:
        feas = df[df["feasible"].astype(bool)].copy()
        fcols = [c for c in _FEASIBLE_ONLY if c in feas.columns]
        if fcols and not feas.empty:
            feas[fcols] = (feas[fcols].apply(pd.to_numeric, errors="coerce")
                           .replace([np.inf, -np.inf], np.nan))
            fg = feas.groupby(group_keys, dropna=False)[fcols].agg(["mean", "std", "count"])
            fg.columns = [f"{m}_feas_{stat}" for m, stat in fg.columns]
            summary = summary.merge(fg.reset_index(), on=group_keys, how="left")
    return summary


def aggregate(raw_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(raw_dir)
    written: dict[str, Path] = {}

    runs_path = out_dir / "runs.csv"
    df.to_csv(runs_path, index=False)
    written["runs"] = runs_path

    # Summaries describe the main comparison only (see _MAIN_TAGS note). Fall
    # back to the full frame if there is no tag column / no main-tagged runs.
    if "tag" in df.columns:
        main_df = df[df["tag"].isin(_MAIN_TAGS)]
        if main_df.empty:
            main_df = df
    else:
        main_df = df

    specs = {
        "summary_by_algorithm": ["algorithm"],
        "summary_by_algo_scenario": ["algorithm", "sla_scenario"],
        "summary_by_algo_scale": ["algorithm", "scale"],
        "summary_by_algo_workflow": ["algorithm", "workflow_type"],
        "summary_by_algo_scale_scenario": ["algorithm", "scale", "sla_scenario"],
    }
    for name, keys in specs.items():
        if all(k in main_df.columns for k in keys) and not main_df.empty:
            s = summarize(main_df, keys)
            p = out_dir / f"{name}.csv"
            s.to_csv(p, index=False)
            written[name] = p
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate raw experiment JSON to CSV")
    ap.add_argument("--raw", default="results/raw_json")
    ap.add_argument("--out", default="results/csv")
    args = ap.parse_args()
    written = aggregate(args.raw, args.out)
    for name, path in written.items():
        print(f"  wrote {name}: {path}")


if __name__ == "__main__":
    main()
