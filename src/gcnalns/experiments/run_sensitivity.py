"""Sensitivity analysis (Spec sec.19.3): sweep a single config parameter.

Sweeps a dotted config key over a list of values, re-running the chosen
algorithms for each value (tagged with the parameter value). Default sweeps the
recovery-penalty weight and the GCN insertion-cost weight ``w6`` for GCN-ALNS.

CLI::

    python -m gcnalns.experiments.run_sensitivity --instances data/instances --out results
    python -m gcnalns.experiments.run_sensitivity --param alns.recovery_penalty_weight --values 10 50 100 200
"""
from __future__ import annotations

import argparse
import copy

from ..config import Config, load_config
from .runner import add_instance_source_args, run_experiment


def _set_dotted(cfg: Config, dotted: str, value) -> None:
    node = cfg.data
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


def sweep(instances, out, base_cfg: Config, param: str, values: list,
          algorithms: list[str], repeats: int | None,
          workers: int | None = None, max_iter: int | None = None,
          protocol_path=None, split=None, instance_scales=None) -> None:
    if split == "test" or (protocol_path is not None and split != "val"):
        raise ValueError("Tune parameters on --split val, not the frozen test set")
    for v in values:
        cfg = Config(copy.deepcopy(base_cfg.data))
        _set_dotted(cfg, param, v)
        safe = param.replace(".", "_")
        run_experiment(instances, out, cfg, algorithms=algorithms,
                       repeats=repeats, tag=f"sens_{safe}_{v}",
                       algo_label_suffix=f" [{param}={v}]",
                       workers=workers, max_iter=max_iter,
                       protocol_path=protocol_path, split=split,
                       instance_scales=instance_scales)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensitivity sweep")
    add_instance_source_args(ap)
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--param", default=None, help="dotted config key to sweep")
    ap.add_argument("--values", nargs="*", type=float, default=None)
    ap.add_argument("--algorithms", nargs="*", default=["ALNS"])
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    instances = None if args.protocol else args.instances
    source = dict(protocol_path=args.protocol, split=args.split,
                  instance_scales=args.instance_scales)
    if args.param and args.values:
        sweep(instances, args.out, cfg, args.param, args.values,
              args.algorithms, args.repeats, args.workers, args.max_iter, **source)
    else:
        # default: two representative sweeps
        sweep(instances, args.out, cfg, "alns.recovery_penalty_weight",
              [10.0, 50.0, 100.0, 200.0], ["ALNS"], args.repeats,
              args.workers, args.max_iter, **source)
        sweep(instances, args.out, cfg, "insert_cost.w6",
              [0.0, 0.5, 1.0, 2.0], ["GCN-ALNS"], args.repeats,
              args.workers, args.max_iter, **source)


if __name__ == "__main__":
    main()
