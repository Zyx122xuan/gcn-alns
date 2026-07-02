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
from .runner import run_experiment

DEFAULT_BUDGETS = [50, 100, 200, 500, 1000]
BUDGET_ALGORITHMS = ["ALNS", "GCN-ALNS"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Iteration-budget / convergence study")
    ap.add_argument("--instances", nargs="+", default=["data/instances"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=DEFAULT_BUDGETS)
    ap.add_argument("--algorithms", nargs="*", default=BUDGET_ALGORITHMS)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    for n in args.budgets:
        run_experiment(args.instances, args.out, cfg,
                       algorithms=args.algorithms, repeats=args.repeats,
                       tag=f"budget_{n}", max_iter=n, workers=args.workers)


if __name__ == "__main__":
    main()
