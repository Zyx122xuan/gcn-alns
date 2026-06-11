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

import pandas as pd

from ..utils.io import load_json

# Metrics summarised with mean (+std). Feasibility is a rate (mean of 0/1).
_NUMERIC = [
    "feasible", "sla_violation", "c_base", "c_worst", "r_max", "r_eff",
    "avg_recovery", "safety_margin", "n_faults", "active_backup_ratio",
    "cost_total", "cost_compute", "cost_backup", "cost_comm", "cost_reserve",
    "cost_recovery", "runtime", "verify_time", "gcn_infer_time",
    "convergence_iter", "prune_ratio",
]


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
    return summary.reset_index()


def aggregate(raw_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(raw_dir)
    written: dict[str, Path] = {}

    runs_path = out_dir / "runs.csv"
    df.to_csv(runs_path, index=False)
    written["runs"] = runs_path

    specs = {
        "summary_by_algorithm": ["algorithm"],
        "summary_by_algo_scenario": ["algorithm", "sla_scenario"],
        "summary_by_algo_scale": ["algorithm", "scale"],
        "summary_by_algo_workflow": ["algorithm", "workflow_type"],
    }
    for name, keys in specs.items():
        if all(k in df.columns for k in keys) and not df.empty:
            s = summarize(df, keys)
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
