"""Experiment orchestration core (Spec sec.19-20).

Runs a set of algorithms on a set of instances, repeating stochastic methods
(ALNS / GCN-ALNS) with distinct seeds, scores every solution with the shared
recovery verifier (via :func:`gcnalns.eval.metrics.evaluate`), writes raw
per-run JSON, and aggregates to summary CSVs.

The entry-point scripts ``run_main`` / ``run_ablation`` / ``run_sensitivity`` /
``run_scalability`` are thin wrappers around :func:`run_experiment`.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..algorithms.registry import get_scheduler
from ..config import Config, load_config
from ..core.model import Instance
from ..data.instance_builder import load_instance
from ..data.links import get_factory
from ..eval.aggregate import aggregate
from ..eval.metrics import evaluate
from ..utils.io import save_json

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


def _load_instances(instances_dir: str | Path, cfg: Config) -> list[Instance]:
    paths = sorted(Path(instances_dir).glob("*.json"))
    return [load_instance(p, cfg) for p in paths]


def run_experiment(instances_dir: str | Path, out_dir: str | Path, cfg: Config,
                   algorithms: list[str] | None = None,
                   repeats: int | None = None,
                   ablation: dict | None = None,
                   model_path: str | None = None,
                   tag: str = "main",
                   algo_label_suffix: str = "",
                   skip_missing_gcn: bool = True,
                   verbose: bool = True) -> list[dict]:
    """Run ``algorithms`` over all instances; write raw JSON + summary CSVs.

    Returns the list of per-run metric records.
    """
    algorithms = algorithms or list(DEFAULT_ALGORITHMS)
    repeats = repeats if repeats is not None else int(cfg.get("alns.repeats", 10))
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw_json"
    csv_dir = out_dir / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_path or cfg.get("gcn.model_path", "results/gcn_model.pt")
    if "GCN-ALNS" in algorithms and skip_missing_gcn and not Path(model_path).exists():
        if verbose:
            print(f"[runner] GCN model {model_path} missing -> skipping GCN-ALNS")
        algorithms = [a for a in algorithms if a != "GCN-ALNS"]

    instances = _load_instances(instances_dir, cfg)
    if verbose:
        print(f"[runner:{tag}] {len(instances)} instances x {len(algorithms)} algorithms")

    records: list[dict] = []
    for name in algorithms:
        n_rep = repeats if name in STOCHASTIC else 1
        label = f"{name}{algo_label_suffix}"
        t_algo = time.perf_counter()
        for inst in instances:
            for rep in range(n_rep):
                factory = get_factory(inst, cfg)
                sched = make_scheduler(name, cfg, seed=rep, ablation=ablation,
                                       model_path=model_path)
                sol = sched.schedule(inst)
                rec = evaluate(inst, sol, cfg, algorithm=label, factory=factory,
                               scheduler=sched, rep=rep)
                rec["tag"] = tag
                records.append(rec)
        if verbose:
            print(f"  [{label}] done in {time.perf_counter()-t_algo:.1f}s")

    raw_path = raw_dir / f"{tag}.json"
    save_json(records, raw_path)
    aggregate(raw_dir, csv_dir)
    if verbose:
        print(f"[runner:{tag}] wrote {len(records)} runs -> {raw_path}; CSVs -> {csv_dir}")
    return records


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run a scheduling experiment")
    ap.add_argument("--instances", default="data/instances")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--algorithms", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(args.instances, args.out, cfg, algorithms=args.algorithms,
                   repeats=args.repeats, tag=args.tag)


if __name__ == "__main__":
    main()
