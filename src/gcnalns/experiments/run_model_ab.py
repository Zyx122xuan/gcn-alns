"""GCN model A/B: quantify what the A2 (FT-mode) fix bought at solve time.

Runs GCN-ALNS twice on the same instances/seeds with two checkpoints:

* ``--old-model`` (mode head trained without class balancing -> effectively
  inert) labelled ``GCN-ALNS (A2-off)``
* ``--new-model`` (mode-BCE ``pos_weight`` fix -> mode AUC ~0.97) labelled
  ``GCN-ALNS (A2-on)``

plus a plain ``ALNS`` reference. Distinct algorithm labels let the aggregator
separate them in one summary. Tags use ``model_ab_a2_off``, ``model_ab_a2_on``
and ``model_ab_alns`` so distinct checkpoint versions never share a resume key.

CLI::

    python -m gcnalns.experiments.run_model_ab --instances data/instances/small \
        --out results --workers 8 --repeats 10 --max-iter 1000
"""
from __future__ import annotations

import argparse

from ..config import load_config
from .runner import add_instance_source_args, experiment_source, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="GCN model A/B (A2 on vs off)")
    add_instance_source_args(ap, default=("data/instances/small",))
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    ap.add_argument("--tag", default="model_ab")
    ap.add_argument("--old-model", dest="old_model",
                    default="results/gcn_model.pt")
    ap.add_argument("--new-model", dest="new_model",
                    default="results/gcn_model_a2.pt")
    ap.add_argument("--with-alns", dest="with_alns", action="store_true",
                    help="also run a plain ALNS reference")
    args = ap.parse_args()

    cfg = load_config(args.config)
    common = dict(repeats=args.repeats, **experiment_source(args),
                  workers=args.workers, max_iter=args.max_iter)

    run_experiment(out_dir=args.out, cfg=cfg, algorithms=["GCN-ALNS"],
                   tag=f"{args.tag}_a2_off",
                   model_path=args.old_model,
                   algo_label_suffix=" (A2-off)", **common)
    run_experiment(out_dir=args.out, cfg=cfg, algorithms=["GCN-ALNS"],
                   tag=f"{args.tag}_a2_on",
                   model_path=args.new_model,
                   algo_label_suffix=" (A2-on)", **common)
    if args.with_alns:
        run_experiment(out_dir=args.out, cfg=cfg, algorithms=["ALNS"],
                       tag=f"{args.tag}_alns",
                       **common)


if __name__ == "__main__":
    main()
