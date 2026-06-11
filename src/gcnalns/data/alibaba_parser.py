"""Parse Alibaba cluster-trace-v2018 into the raw machine schema.

Emits the same dict shape as
:func:`gcnalns.data.synthetic.generate_raw_machines`:
``{machine_id, cpu_cap, mem_cap, cpu_util, mem_util, cpu_var, mem_var}``.

Reads ``machine_meta.csv`` (capacities) and ``machine_usage.csv`` (utilization
time series), aggregates a snapshot within a time window, and samples
``n_nodes`` machines. The trace is huge, so usage is read in chunks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column layouts for cluster-trace-v2018 (no header in raw files).
META_COLS = ["machine_id", "time_stamp", "failure_domain_1", "failure_domain_2",
             "cpu_num", "mem_size", "status"]
USAGE_COLS = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
              "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]


def snapshot(meta_path: str | Path, usage_path: str | Path, n_nodes: int,
             window: tuple[int, int] | None = None, seed: int = 2026) -> list[dict]:
    """Build a resource snapshot of ``n_nodes`` machines."""
    rng = np.random.default_rng(seed)

    meta = pd.read_csv(meta_path, header=None, names=META_COLS)
    meta = meta.dropna(subset=["cpu_num", "mem_size"]).drop_duplicates("machine_id")
    if len(meta) > n_nodes:
        meta = meta.sample(n=n_nodes, random_state=seed)
    machine_ids = set(meta["machine_id"].tolist())

    # aggregate usage per machine within the window
    agg: dict[str, dict[str, list]] = {m: {"cpu": [], "mem": []} for m in machine_ids}
    for chunk in pd.read_csv(usage_path, header=None, names=USAGE_COLS, chunksize=1_000_000):
        if window is not None:
            chunk = chunk[(chunk["time_stamp"] >= window[0]) & (chunk["time_stamp"] <= window[1])]
        chunk = chunk[chunk["machine_id"].isin(machine_ids)]
        for mid, grp in chunk.groupby("machine_id"):
            agg[mid]["cpu"].extend(grp["cpu_util_percent"].tolist())
            agg[mid]["mem"].extend(grp["mem_util_percent"].tolist())

    machines: list[dict] = []
    for _, row in meta.iterrows():
        mid = row["machine_id"]
        cpu_series = np.array(agg[mid]["cpu"], dtype=float) / 100.0 if agg[mid]["cpu"] else np.array([rng.uniform(0.2, 0.6)])
        mem_series = np.array(agg[mid]["mem"], dtype=float) / 100.0 if agg[mid]["mem"] else np.array([rng.uniform(0.2, 0.6)])
        machines.append({
            "machine_id": str(mid),
            "cpu_cap": float(row["cpu_num"]),
            "mem_cap": float(row["mem_size"]),
            "cpu_util": float(np.clip(np.mean(cpu_series), 0.0, 0.98)),
            "mem_util": float(np.clip(np.mean(mem_series), 0.0, 0.98)),
            "cpu_var": float(np.var(cpu_series)),
            "mem_var": float(np.var(mem_series)),
        })
    return machines
