"""Scalability study (Spec sec.19.4): runtime / quality vs. problem size.

Generates (if requested) instances at multiple scales and runs a representative
set of algorithms, recording scheduling runtime and recovery-verification time.
The ``summary_by_algo_scale.csv`` produced by aggregation captures the scaling
trend (instances also record ``num_tasks`` / ``num_nodes``).

CLI::

    # use an existing multi-scale instance directory
    python -m gcnalns.experiments.run_scalability --instances data/instances_scal --out results

    # generate small/medium/large instances first, then run
    python -m gcnalns.experiments.run_scalability --generate --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..data import synthetic
from ..data.instance_builder import build_instance, instance_to_dict
from ..utils.io import save_json
from .runner import run_experiment

SCALABILITY_ALGORITHMS = [
    "HEFT", "Criticality-Based Hybrid", "ALNS", "GCN-ALNS",
]


def generate_scalability_instances(out_dir: str | Path, cfg, scales: list[str],
                                   load: str = "medium", per_scale: int = 1,
                                   scenario: str = "medium") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_seed = int(cfg.get("seeds.global", 2026))
    wf_types = cfg["workflow_types"]
    count = 0
    for scale in scales:
        n_tasks = cfg["scales"]["workflow_tasks"][scale]
        n_nodes = cfg["scales"]["resource_nodes"][scale]
        load_level = cfg["scales"]["load_levels"][load]
        for rep in range(per_scale):
            wf_type = wf_types[(count) % len(wf_types)]
            seed = base_seed + count
            raw_wf = synthetic.generate_raw_workflow(wf_type, n_tasks, seed)
            machines = synthetic.generate_raw_machines(n_nodes, load_level, seed + 10_000)
            iid = f"{wf_type}_{scale}_{load}_{scenario}_r{rep}"
            inst = build_instance(raw_wf, machines, cfg, iid, scenario,
                                  meta={"scale": scale, "load": load, "seed": seed})
            d = instance_to_dict(inst, materialize_links=(n_nodes <= 60), cfg=cfg)
            save_json(d, out_dir / f"{iid}.json")
            count += 1
    print(f"generated {count} scalability instances into {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Scalability study")
    ap.add_argument("--instances", nargs="+", default=["data/instances_scal"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--scales", nargs="*", default=["small", "medium", "large"])
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    instances = args.instances
    if args.generate:
        # --generate writes to the first --instances path (single dir).
        gen_dir = args.instances[0] if isinstance(args.instances, list) else args.instances
        instances = generate_scalability_instances(gen_dir, cfg, args.scales)

    run_experiment(instances, args.out, cfg,
                   algorithms=SCALABILITY_ALGORITHMS, repeats=args.repeats,
                   tag="scalability", workers=args.workers, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
