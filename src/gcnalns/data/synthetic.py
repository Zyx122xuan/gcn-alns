"""Synthetic data generators that mimic the schema of real datasets.

Two raw producers:

* :func:`generate_raw_workflow` - produces a Pegasus-like workflow dict
  (jobs with runtimes, dependency edges with file sizes) for one of the five
  workflow types, with a target number of tasks.
* :func:`generate_raw_machines` - produces an Alibaba-trace-like machine list
  (cpu/mem capacity, cpu/mem utilization, utilization variance).

These dicts have the *same shape* the real parsers emit, so the layering /
links / instance-builder stages are agnostic to whether data is synthetic or
parsed from WfCommons / Alibaba traces.
"""
from __future__ import annotations

import numpy as np

WORKFLOW_TYPES = ["Montage", "CyberShake", "Epigenomics", "LIGO", "SIPHT"]


# ---------------------------------------------------------------------------
# Workflow DAG generation (type-specific topologies)
# ---------------------------------------------------------------------------
def _add_edge(edges, files, i, j, rng):
    edges.append((i, j))
    files[(i, j)] = float(rng.uniform(5.0, 200.0))  # MB


def _layered_dag(n_tasks, width_fn, rng):
    """Build a layered DAG where ``width_fn(level)`` gives nodes per level."""
    levels: list[list[int]] = []
    tid = 0
    remaining = n_tasks
    level = 0
    while remaining > 0:
        w = max(1, min(remaining, width_fn(level)))
        levels.append(list(range(tid, tid + w)))
        tid += w
        remaining -= w
        level += 1
    return levels


def generate_raw_workflow(wf_type: str, n_tasks: int, seed: int) -> dict:
    """Generate a Pegasus-like workflow with ~``n_tasks`` jobs.

    Returns dict: {name, jobs:{id:{runtime}}, edges:[(i,j)], files:{(i,j):MB}}.
    Topologies loosely emulate the canonical Pegasus shapes.
    """
    rng = np.random.default_rng(seed)
    edges: list[tuple[int, int]] = []
    files: dict[tuple[int, int], float] = {}

    if wf_type == "Montage":
        # Fan-out -> per-tile processing -> heavy fan-in aggregation.
        levels = _layered_dag(n_tasks, lambda lv: [1, max(2, n_tasks // 3),
                                                   max(2, n_tasks // 3), 2, 1][min(lv, 4)], rng)
    elif wf_type == "Epigenomics":
        # Mostly parallel pipelines that split then merge (multi-stage).
        branches = max(2, int(np.sqrt(n_tasks)))
        levels = _layered_dag(n_tasks, lambda lv: 1 if lv == 0 else branches, rng)
    elif wf_type == "SIPHT":
        # Many independent short stages, modest fan-in at the end.
        levels = _layered_dag(n_tasks, lambda lv: max(3, n_tasks // 5), rng)
    elif wf_type == "LIGO":
        # Several parallel blocks with cross dependencies.
        levels = _layered_dag(n_tasks, lambda lv: max(2, n_tasks // 6), rng)
    else:  # CyberShake - complex, high fan-out from a couple of sources.
        levels = _layered_dag(n_tasks, lambda lv: [2, max(4, n_tasks // 2)][min(lv, 1)]
                              if lv < 2 else max(2, n_tasks // 8), rng)

    # Connect consecutive levels; each node links to >=1 predecessor.
    for li in range(1, len(levels)):
        prev, cur = levels[li - 1], levels[li]
        for j in cur:
            k = rng.integers(1, min(3, len(prev)) + 1)
            for i in rng.choice(prev, size=int(k), replace=False):
                _add_edge(edges, files, int(i), int(j), rng)
    # CyberShake/LIGO: add a few skip edges for structural complexity.
    if wf_type in {"CyberShake", "LIGO"} and len(levels) > 2:
        flat = [t for lv in levels for t in lv]
        for _ in range(max(1, n_tasks // 20)):
            i, j = sorted(rng.choice(flat, size=2, replace=False))
            if i != j and (int(i), int(j)) not in files:
                _add_edge(edges, files, int(i), int(j), rng)

    # Runtimes: heavy-tailed (lognormal) to mimic real job runtime spread.
    jobs = {}
    for lv in levels:
        for t in lv:
            jobs[int(t)] = {"runtime": float(rng.lognormal(mean=2.0, sigma=0.7))}

    return {"name": wf_type, "jobs": jobs, "edges": edges, "files": files}


# ---------------------------------------------------------------------------
# Resource (Alibaba-like) machine snapshot generation
# ---------------------------------------------------------------------------
def generate_raw_machines(n_nodes: int, load_level: dict, seed: int) -> list[dict]:
    """Generate Alibaba-trace-like machines.

    Each machine: {machine_id, cpu_cap, mem_cap, cpu_util, mem_util,
    cpu_var, mem_var}. ``load_level`` = {util_low, util_high} controls
    utilization range (low/medium/high load scenarios).
    """
    rng = np.random.default_rng(seed)
    lo, hi = load_level["util_low"], load_level["util_high"]
    machines = []
    # Heterogeneous capacities: mixture of small/medium/large machines.
    cpu_choices = np.array([8, 16, 32, 64, 96])
    mem_choices = np.array([16, 32, 64, 128, 256])
    for mid in range(n_nodes):
        idx = rng.integers(0, len(cpu_choices))
        cpu_cap = float(cpu_choices[idx])
        mem_cap = float(mem_choices[idx])
        cpu_util = float(np.clip(rng.uniform(lo, hi), 0.0, 0.98))
        mem_util = float(np.clip(rng.uniform(lo, hi), 0.0, 0.98))
        # Variance correlates with load and inversely with capacity.
        base_var = rng.uniform(0.0, 0.15) * (1.0 + cpu_util)
        machines.append({
            "machine_id": f"m{mid}",
            "cpu_cap": cpu_cap,
            "mem_cap": mem_cap,
            "cpu_util": cpu_util,
            "mem_util": mem_util,
            "cpu_var": float(base_var),
            "mem_var": float(base_var * rng.uniform(0.7, 1.3)),
        })
    return machines
