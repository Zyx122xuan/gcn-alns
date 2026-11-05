"""Experiment orchestration core (Spec sec.19-20).

Runs a set of algorithms on a set of instances, repeating stochastic methods
(ALNS / GCN-ALNS) with distinct seeds, scores every solution with the shared
recovery verifier (via :func:`gcnalns.eval.metrics.evaluate`), writes raw
per-run JSON, and aggregates to summary CSVs.

The entry-point scripts ``run_main`` / ``run_ablation`` / ``run_sensitivity`` /
``run_scalability`` are thin wrappers around :func:`run_experiment`.

Execution is parallelised across the independent ``(algorithm, instance, rep)``
work items with a process pool (one verifier/solve per worker). Results are
streamed to ``<tag>.jsonl`` so a long run can be resumed. A frozen manifest
must match config, code, model and instance contents before completed work is
skipped. Every result also references a saved deployment solution.
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
from ..data.protocol import load_protocol
from ..eval.aggregate import aggregate
from ..eval.metrics import evaluate
from ..utils.io import _NpEncoder, save_json
from ..utils.provenance import (
    assert_disjoint, code_version, collect_instances, content_hash, file_hash,
    write_manifest,
)

# Algorithms whose output depends on the random seed -> repeated.
STOCHASTIC = {"ALNS", "GCN-ALNS", "GA-Placement", "RA-MOMA"}

DEFAULT_ALGORITHMS = [
    "HEFT", "PEFT", "No-FT", "Reactive-Only", "Full-Proactive",
    "Critical-Path Backup", "Criticality-Based Hybrid", "Reliability-Aware",
    "RECR-D", "HFWPF", "GA-Placement", "RA-MOMA",
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
    name, label, inst_path, rep, ablation, model_path, cfg_data, tag, context = task
    if file_hash(inst_path) != context["instance_sha256"]:
        raise ValueError(f"Instance changed after experiment preflight: {inst_path}")
    if name == "GCN-ALNS" and file_hash(model_path) != context["model_sha256"]:
        raise ValueError("GCN checkpoint changed after experiment preflight")
    cfg = Config(cfg_data)
    inst = load_instance(inst_path, cfg)
    factory = get_factory(inst, cfg)
    sched = make_scheduler(name, cfg, seed=rep, ablation=ablation,
                           model_path=model_path)
    sol = sched.schedule(inst)
    rec = evaluate(inst, sol, cfg, algorithm=label, factory=factory,
                   scheduler=sched, rep=rep)
    rec["tag"] = tag
    rec.update({k: v for k, v in context.items() if k != "out_dir"})
    key = content_hash(_record_key(rec))
    solution_path = Path("solutions") / tag / f"{key}.json"
    target = Path(context["out_dir"]) / solution_path
    save_json({"experiment_id": context["experiment_id"], "tag": tag,
               "algorithm": label, "iid": inst.iid, "rep": rep,
               "placement": sol.placement, "mode": sol.mode,
               "order": sol.order, "meta": sol.meta}, target)
    rec["solution_path"] = solution_path.as_posix()
    rec["solution_sha256"] = file_hash(target)
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


def _check_test_model(model_path: Path, protocol: dict) -> None:
    """Legacy checkpoints can run development studies, never certified tests."""
    sidecar = model_path.with_name(model_path.name + ".provenance.json")
    if not sidecar.exists():
        raise ValueError("Final test requires checkpoint training provenance; "
                         "rebuild the dataset with --protocol and retrain first")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if data.get("checkpoint_sha256") != file_hash(model_path):
        raise ValueError("Checkpoint does not match its training provenance")
    training = data.get("training") or {}
    dataset = training.get("dataset") or {}
    if dataset.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("Checkpoint was not trained with this frozen protocol")
    teacher_inputs = dataset.get("teacher_inputs")
    if not teacher_inputs:
        raise ValueError("Checkpoint has no auditable teacher input identities")
    assert_disjoint({"teacher": teacher_inputs, "test": protocol["splits"]["test"]})
    allowed = {r["iid"]: r for role in ("train", "val")
               for r in protocol["splits"][role]}
    for record in teacher_inputs:
        if record["iid"] not in allowed or any(
                record[key] != allowed[record["iid"]][key]
                for key in ("sha256", "workflow_sha256", "resource_sha256")):
            raise ValueError("Checkpoint teacher inputs are outside protocol train/val")
    sample_splits = training.get("sample_splits") or {}
    if sample_splits.get("test"):
        raise ValueError("Checkpoint used the solver test set during training")
    for role in ("train", "val"):
        ids = sample_splits.get(role) or []
        role_ids = {r["iid"] for r in protocol["splits"][role]}
        if not ids or not set(ids).issubset(role_ids):
            raise ValueError(f"Checkpoint sample assignments violate frozen {role} split")


def _freeze_experiment(out_dir: Path, tag: str, payload: dict,
                       existing: list[dict]) -> str:
    fingerprint = content_hash(payload)
    path = out_dir / "manifests" / f"{tag}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (previous.get("experiment_id") != fingerprint
                or content_hash({k: v for k, v in previous.items()
                                 if k != "experiment_id"}) != fingerprint):
            raise ValueError("Experiment inputs/config/model/code changed; "
                             "use a new --tag or output directory")
    elif existing:
        raise ValueError("Legacy results lack a version manifest; "
                         "use a new --tag or output directory")
    else:
        write_manifest({"experiment_id": fingerprint, **payload}, path)
    for record in existing:
        if record.get("experiment_id") != fingerprint:
            raise ValueError("Existing result has incompatible experiment provenance")
        target = (out_dir / record.get("solution_path", "")).resolve()
        if (not target.is_relative_to(out_dir.resolve()) or not target.is_file()
                or file_hash(target) != record.get("solution_sha256")):
            raise ValueError("Saved solution missing or changed; use a new result tag")
    return fingerprint


def _check_output_scope(out_dir: Path, role: str, protocol_hash: str | None) -> None:
    """Keep raw aggregation from combining legacy, development and test results."""
    scope = {"evaluation_split": role, "protocol_sha256": protocol_hash}
    scope_path = out_dir / "evaluation_scope.json"
    if scope_path.exists() and json.loads(scope_path.read_text(encoding="utf-8")) != scope:
        raise ValueError("Output evaluation split/protocol changed; use a new output directory")
    raw = out_dir / "raw_json"
    tags = {p.stem for pattern in ("*.json", "*.jsonl") for p in raw.glob(pattern)}
    for tag in tags:
        for row in _load_existing(raw / f"{tag}.jsonl", raw / f"{tag}.json"):
            if not row.get("experiment_id"):
                raise ValueError("Legacy results lack evaluation provenance; "
                                 "use a new output directory")
            if any(row.get(key) != value for key, value in scope.items()):
                raise ValueError("Output evaluation split/protocol changed; "
                                 "use a new output directory")
    if not scope_path.exists():
        write_manifest(scope, scope_path)


def add_instance_source_args(parser, default=("data/instances",)) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--instances", nargs="+", default=list(default))
    source.add_argument("--protocol", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--instance-scales", nargs="+", choices=["small", "medium", "large"],
                        default=None, help="evaluate only these scales within the frozen split")


def experiment_source(args) -> dict:
    return {"instances_dir": None if args.protocol else args.instances,
            "protocol_path": args.protocol, "split": args.split,
            "instance_scales": args.instance_scales}


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
                   verbose: bool = True,
                   protocol_path: str | Path | None = None,
                   split: str | None = None,
                   instance_scales: list[str] | None = None) -> list[dict]:
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
    if not tag or Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("tag must be a single nonempty filename")
    out_dir = Path(out_dir).resolve()
    raw_dir = out_dir / "raw_json"
    csv_dir = out_dir / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)

    protocol = None
    if protocol_path is not None:
        if instances_dir is not None or split not in {"train", "val", "test"}:
            raise ValueError("Use protocol_path with an explicit train/val/test split, "
                             "without instances_dir")
        protocol = load_protocol(protocol_path)
        entries = protocol["splits"][split]
    else:
        if split is not None:
            raise ValueError("An evaluation split requires a frozen protocol")
        entries = collect_instances(instances_dir)
    selected_scales = sorted(set(instance_scales or []))
    if selected_scales:
        if any(scale not in {"small", "medium", "large"} for scale in selected_scales):
            raise ValueError("Unknown instance scale")
        entries = [r for r in entries if json.loads(Path(r["path"]).read_text(
            encoding="utf-8")).get("meta", {}).get("scale") in selected_scales]
        if not entries:
            raise ValueError("No instances in the requested scales")
    evaluation_split = split or "development"

    model_path = str(Path(model_path or cfg.get("gcn.model_path", "results/gcn_model.pt")).resolve())
    if protocol and split == "test" and "GCN-ALNS" in algorithms:
        if not Path(model_path).is_file():
            raise ValueError("Final test requested GCN-ALNS but its checkpoint is missing")
        _check_test_model(Path(model_path), protocol)
    if "GCN-ALNS" in algorithms and skip_missing_gcn and not Path(model_path).exists():
        if verbose:
            print(f"[runner] GCN model {model_path} missing -> skipping GCN-ALNS")
        algorithms = [a for a in algorithms if a != "GCN-ALNS"]

    # Per-run config snapshot (deep-copied so caller's cfg is never mutated).
    cfg_data = copy.deepcopy(cfg.data)
    if max_iter is not None:
        cfg_data.setdefault("alns", {})["max_iter"] = int(max_iter)

    if not algorithms:
        raise ValueError("No algorithms remain to run")
    paths = [Path(r["path"]) for r in entries]
    if verbose:
        print(f"[runner:{tag}] {len(paths)} instances x {len(algorithms)} "
              f"algorithms, repeats={repeats}, workers={workers}")

    # Resume: load anything already computed for this tag.
    jsonl_path = raw_dir / f"{tag}.jsonl"
    json_path = raw_dir / f"{tag}.json"
    existing = _load_existing(jsonl_path, json_path)
    _check_output_scope(out_dir, evaluation_split,
                        protocol["sha256"] if protocol else None)
    model = None
    if "GCN-ALNS" in algorithms:
        sidecar = Path(model_path + ".provenance.json")
        model = {"path": model_path, "sha256": file_hash(model_path),
                 "provenance_sha256": file_hash(sidecar) if sidecar.exists() else None}
    version = code_version()
    fingerprint = _freeze_experiment(out_dir, tag, {
        "schema_version": 1, "tag": tag, "config": cfg_data,
        "config_sha256": content_hash(cfg_data), "code": version,
        "model": model, "instances": entries,
        "protocol_sha256": protocol["sha256"] if protocol else None,
        "evaluation_split": evaluation_split, "algorithms": algorithms,
        "repeats": repeats, "ablation": ablation or {},
        "instance_scales": selected_scales,
        "algo_label_suffix": algo_label_suffix,
    }, existing)
    done = {_record_key(r) for r in existing}
    if existing and verbose:
        print(f"[runner:{tag}] resuming: {len(existing)} records already done")

    # Build the flat task list, skipping completed items.
    tasks: list[tuple] = []
    for name in algorithms:
        n_rep = repeats if name in STOCHASTIC else 1
        label = f"{name}{algo_label_suffix}"
        for entry in entries:
            p = Path(entry["path"])
            iid = entry["iid"]
            for rep in range(n_rep):
                if (tag, label, iid, rep) in done:
                    continue
                context = {
                    "out_dir": str(out_dir), "experiment_id": fingerprint,
                    "config_sha256": content_hash(cfg_data),
                    "code_sha256": version["sha256"],
                    "instance_sha256": entry["sha256"],
                    "model_sha256": model["sha256"] if name == "GCN-ALNS" else None,
                    "evaluation_split": evaluation_split,
                    "protocol_sha256": protocol["sha256"] if protocol else None,
                }
                tasks.append((name, label, str(p), rep, ablation, model_path,
                              cfg_data, tag, context))

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
    add_instance_source_args(ap)
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--algorithms", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(out_dir=args.out, cfg=cfg, **experiment_source(args),
                   algorithms=args.algorithms,
                   repeats=args.repeats, tag=args.tag, workers=args.workers,
                   max_iter=args.max_iter)


if __name__ == "__main__":
    main()
