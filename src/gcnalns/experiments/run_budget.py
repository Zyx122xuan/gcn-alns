"""Convergence / limited-budget experiment (addresses issue #1).

Runs ALNS and GCN-ALNS under a sweep of iteration budgets so the GCN's faster
convergence (reaching feasibility / lower cost with fewer iterations) becomes
visible. Each budget ``n`` is tagged ``budget_<n>``; the resulting runs feed the
``fig_budget_convergence`` figure (feasible ratio + feasible cost vs. budget).

CLI::

    python -m gcnalns.experiments.run_budget --instances data/instances/small \
        --out results --workers 30 --repeats 10
"""
from __future__ import annotations

import argparse

from ..config import load_config
from .runner import add_instance_source_args, experiment_source, run_experiment

DEFAULT_BUDGETS = [50, 100, 200, 500, 1000]
BUDGET_ALGORITHMS = ["ALNS", "GCN-ALNS"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Iteration-budget / convergence study")
    add_instance_source_args(ap)
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=DEFAULT_BUDGETS)
    ap.add_argument("--algorithms", nargs="*", default=BUDGET_ALGORITHMS)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--model-path", dest="model_path", default=None,
                    help="GCN model checkpoint (default: cfg gcn.model_path)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for n in args.budgets:
        run_experiment(out_dir=args.out, cfg=cfg, **experiment_source(args),
                       algorithms=args.algorithms, repeats=args.repeats,
                       tag=f"budget_{n}", max_iter=n, workers=args.workers,
                       model_path=args.model_path)


if __name__ == "__main__":
    main()
