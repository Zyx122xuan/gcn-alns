"""Parse Pegasus / WfCommons workflow JSON into the raw workflow schema.

Emits the same dict shape as :func:`gcnalns.data.synthetic.generate_raw_workflow`:
``{name, jobs:{id:{runtime}}, edges:[(i,j)], files:{(i,j):MB}}`` so the rest of
the pipeline is agnostic to the data source.

Supports the common WfCommons schema variants (``workflow.jobs`` /
``workflow.tasks``; ``runtimeInSeconds`` / ``runtime``; parent/children links;
per-file ``size`` with input/output ``link``). Missing file sizes are imputed
with the median rule (Spec sec.2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import load_json


def _job_runtime(job: dict) -> float:
    for key in ("runtimeInSeconds", "runtime", "execTime"):
        if key in job and job[key] is not None:
            return float(job[key])
    return 1.0


def parse(path: str | Path, name: str | None = None) -> dict:
    """Parse a WfCommons / Pegasus workflow JSON file."""
    doc = load_json(path)
    wf = doc.get("workflow", doc)
    jobs_list = wf.get("jobs") or wf.get("tasks") or []

    # name -> integer id
    name_to_id: dict[str, int] = {}
    for idx, job in enumerate(jobs_list):
        name_to_id[job.get("name", job.get("id", str(idx)))] = idx

    jobs: dict[int, dict] = {}
    edges: list[tuple[int, int]] = []
    files: dict[tuple[int, int], float] = {}
    # output file size per producer job (name -> MB), used to size edges
    out_size: dict[str, float] = {}

    for idx, job in enumerate(jobs_list):
        jname = job.get("name", job.get("id", str(idx)))
        jobs[idx] = {"runtime": _job_runtime(job)}
        total_out = 0.0
        for f in job.get("files", []):
            size_mb = float(f.get("size", f.get("sizeInBytes", 0)) or 0)
            # bytes -> MB heuristic when size looks like bytes
            if size_mb > 1e6:
                size_mb /= 1e6
            link = (f.get("link") or f.get("type") or "").lower()
            if link in ("output", "out"):
                total_out += size_mb
        out_size[jname] = total_out

    # edges from parents/children
    for idx, job in enumerate(jobs_list):
        jname = job.get("name", job.get("id", str(idx)))
        parents = job.get("parents") or job.get("parentIds") or []
        for p in parents:
            if p in name_to_id:
                i, j = name_to_id[p], idx
                edges.append((i, j))
                files[(i, j)] = out_size.get(p, 0.0)
        children = job.get("children") or []
        for c in children:
            if c in name_to_id:
                i, j = idx, name_to_id[c]
                edges.append((i, j))
                files[(i, j)] = out_size.get(jname, 0.0)

    edges = sorted(set(edges))
    files = {e: files.get(e, 0.0) for e in edges}

    # impute missing/zero file sizes with median of known sizes
    known = [v for v in files.values() if v > 0]
    median = float(np.median(known)) if known else 10.0
    for e in files:
        if files[e] <= 0:
            files[e] = median

    return {"name": name or wf.get("name", Path(path).stem),
            "jobs": jobs, "edges": edges, "files": files}
