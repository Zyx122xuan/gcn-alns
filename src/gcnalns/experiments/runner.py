"""Experiment orchestration core (Spec sec.19-20).

Runs a set of algorithms on a set of instances, repeating stochastic methods
(ALNS / GCN-ALNS) with distinct seeds, scores every solution with the shared
recovery verifier (via :func:`gcnalns.eval.metrics.evaluate`), writes raw
per-run JSON, and aggregates to summary CSVs.

The entry-point scripts ``run_main`` / ``run_ablation`` / ``run_sensitivity`` /
``run_scalability`` are thin wrappers around :func:`run_experiment`.

Execution is parallelised across the independent ``(algorithm, instance, rep)``
work items with a process pool (one verifier/solve per worker). Results are
streamed to a crash-safe ``<tag>.jsonl`` so a long run can be resumed: on
restart, already-completed ``(tag, algorithm, iid, rep)`` items are skipped.
"""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..algorithms.registry import get_scheduler
from ..config import Config, load_config
from ..data.instance_builder import load_instance
from ..data.links import get_factory
from ..eval.aggregate import aggregate
from ..eval.metrics import evaluate
from ..utils.io import _NpEncoder, save_json

# Algorithms whose output depends on the random seed -> repeated.
STOCHASTIC = {"ALNS", "GCN-ALNS"}

DEFAULT_ALGORITHMS = [
    "HEFT", "PEFT", "No-FT", "Reactive-Only", "Full-Proactive",
    "Critical-Path Backup", "Criticality-Based Hybrid", "Reliability-Aware",
    "ALNS", "GCN-ALNS",
]


def make_scheduler(name: str, cfg: Config, seed: int = 0,
                   ablation: dict | None = None, model_path: str | None = None):
    """Instantiate a scheduler, passing seed/ablation/model where supported."""
    cls = get_scheduler(name)
    if name == "GCN-ALNS":
        return cls(cfg, seed=seed, ablation=ablation, model_path=model_path)
    if name == "ALNS":
        return cls(cfg, seed=seed, ablation=ablation)
    return cls(cfg, seed=seed)


def _collect_paths(instances_dir) -> list[Path]:
    """Return sorted instance JSON paths from one dir or a list of dirs."""
    if isinstance(instances_dir, (str, Path)):
        dirs = [instances_dir]
    else:
        dirs = list(instances_dir)
    paths: list[Path] = []
    for d in dirs:
        paths.extend(sorted(Path(d).glob("*.json")))
    return paths


def _default_workers() -> int:
    return min(30, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Worker (must be top-level for the spawn start method on Windows)
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    """Limit per-worker BLAS / torch threads to avoid oversubscription."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:  # torch may be absent for baseline-only runs
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _run_one(task: tuple) -> dict:
    """Solve + evaluate a single ``(algorithm, instance, rep)`` work item."""
    name, label, inst_path, rep, ablation, model_path, cfg_data, tag = task
    cfg = Config(cfg_data)
    inst = load_instance(inst_path, cfg)
    factory = get_factory(inst, cfg)
    sched = make_scheduler(name, cfg, seed=rep, ablation=ablation,
                           model_path=model_path)
    sol = sched.schedule(inst)
    rec = evaluate(inst, sol, cfg, algorithm=label, factory=factory,
                   scheduler=sched, rep=rep)
    rec["tag"] = tag
    return rec


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def _record_key(rec: dict) -> tuple:
    return (rec.get("tag"), rec.get("algorithm"), rec.get("iid"), rec.get("rep"))


def _load_existing(jsonl_path: Path, json_path: Path) -> list[dict]:
    """Load already-computed records (for resume) from jsonl and/or json."""
    records: list[dict] = []
    seen: set[tuple] = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a half-written trailing line
                k = _record_key(rec)
                if k not in seen:
                    seen.add(k)
                    records.append(rec)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                for rec in json.load(fh):
                    k = _record_key(rec)
                    if k not in seen:
                        seen.add(k)
                        records.append(rec)
        except (json.JSONDecodeError, TypeError):
            pass
    return records


def run_experiment(instances_dir, out_dir, cfg: Config,
                   algorithms: list[str] | None = None,
                   repeats: int | None = None,
                   ablation: dict | None = None,
                   model_path: str | None = None,
                   tag: str = "main",
                   algo_label_suffix: str = "",
                   skip_missing_gcn: bool = True,
                   workers: int | None = None,
                   max_iter: int | None = None,
                   verbose: bool = True) -> list[dict]:
    """Run ``algorithms`` over all instances; write raw JSON + summary CSVs.

    ``instances_dir`` may be a single directory or a list of directories.
    ``workers`` controls process-level parallelism (default ~min(30, cpus));
    ``workers=1`` runs sequentially (useful for deterministic parity checks).
    ``max_iter`` overrides ``cfg["alns"]["max_iter"]`` for this run.

    Returns the list of per-run metric records.
    """
    algorithms = algorithms or list(DEFAULT_ALGORITHMS)
    repeats = repeats if repeats is not None else int(cfg.get("alns.repeats", 10))
    workers = workers if workers is not None else _default_workers()
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw_json"
    csv_dir = out_dir / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_path or cfg.get("gcn.model_path", "results/gcn_model.pt")
    if "GCN-ALNS" in algorithms and skip_missing_gcn and not Path(model_path).exists():
        if verbose:
            print(f"[runner] GCN model {model_path} missing -> skipping GCN-ALNS")
        algorithms = [a for a in algorithms if a != "GCN-ALNS"]

    # Per-run config snapshot (deep-copied so caller's cfg is never mutated).
    cfg_data = copy.deepcopy(cfg.data)
    if max_iter is not None:
        cfg_data.setdefault("alns", {})["max_iter"] = int(max_iter)

    paths = _collect_paths(instances_dir)
    if verbose:
        print(f"[runner:{tag}] {len(paths)} instances x {len(algorithms)} "
              f"algorithms, repeats={repeats}, workers={workers}")

    # Resume: load anything already computed for this tag.
    jsonl_path = raw_dir / f"{tag}.jsonl"
    json_path = raw_dir / f"{tag}.json"
    existing = _load_existing(jsonl_path, json_path)
    done = {_record_key(r) for r in existing}
    if existing and verbose:
        print(f"[runner:{tag}] resuming: {len(existing)} records already done")

    # Build the flat task list, skipping completed items.
    tasks: list[tuple] = []
    for name in algorithms:
        n_rep = repeats if name in STOCHASTIC else 1
        label = f"{name}{algo_label_suffix}"
        for p in paths:
            iid = p.stem
            for rep in range(n_rep):
                if (tag, label, iid, rep) in done:
                    continue
                tasks.append((name, label, str(p), rep, ablation, model_path,
                              cfg_data, tag))

    records: list[dict] = list(existing)
    t0 = time.perf_counter()
    if tasks:
        # Append-mode jsonl writer in the parent (single writer -> no lock).
        with open(jsonl_path, "a", encoding="utf-8") as sink:
            if workers <= 1:
                _worker_init()
                for i, task in enumerate(tasks, 1):
                    rec = _run_one(task)
                    records.append(rec)
                    sink.write(json.dumps(rec, cls=_NpEncoder, ensure_ascii=False) + "\n")
                    sink.flush()
                    if verbose and (i % 20 == 0 or i == len(tasks)):
                        print(f"  [{tag}] {i}/{len(tasks)} "
                              f"({time.perf_counter()-t0:.0f}s)")
            else:
                with ProcessPoolExecutor(max_workers=workers,
                                         initializer=_worker_init) as ex:
                    futures = [ex.submit(_run_one, t) for t in tasks]
                    for i, fut in enumerate(as_completed(futures), 1):
                        rec = fut.result()
                        records.append(rec)
                        sink.write(json.dumps(rec, cls=_NpEncoder,
                                              ensure_ascii=False) + "\n")
                        sink.flush()
                        if verbose and (i % 20 == 0 or i == len(tasks)):
                            print(f"  [{tag}] {i}/{len(tasks)} "
                                  f"({time.perf_counter()-t0:.0f}s)")
    elif verbose:
        print(f"[runner:{tag}] nothing to do (all items already computed)")

    save_json(records, json_path)
    aggregate(raw_dir, csv_dir)
    if verbose:
        print(f"[runner:{tag}] wrote {len(records)} runs -> {json_path}; "
              f"CSVs -> {csv_dir} ({time.perf_counter()-t0:.0f}s)")
    return records


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run a scheduling experiment")
    ap.add_argument("--instances", nargs="+", default=["data/instances"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--algorithms", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(args.instances, args.out, cfg, algorithms=args.algorithms,
                   repeats=args.repeats, tag=args.tag, workers=args.workers,
                   max_iter=args.max_iter)


if __name__ == "__main__":
    main()
