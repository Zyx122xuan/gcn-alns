"""Main comparison experiment (Spec sec.19): all baselines + ALNS + GCN-ALNS.

CLI::

    python -m gcnalns.experiments.run_main --instances data/instances --out results
"""
from __future__ import annotations

import argparse

from ..config import load_config
from .runner import DEFAULT_ALGORITHMS, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Main scheduling comparison")
    ap.add_argument("--instances", default="data/instances")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--algorithms", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_experiment(args.instances, args.out, cfg,
                   algorithms=args.algorithms or list(DEFAULT_ALGORITHMS),
                   repeats=args.repeats, tag="main")


if __name__ == "__main__":
    main()
