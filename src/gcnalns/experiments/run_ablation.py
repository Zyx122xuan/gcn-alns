"""Ablation study (Spec sec.19.2): isolate each GCN-ALNS component.

Variants (toggled via the ALNS ``ablation`` dict):

* Full                  - the complete GCN-ALNS
* w/o GCN               - drop the learned guidance (plain ALNS search)
* w/o Recovery Penalty  - remove the R_max>budget penalty from the objective
* w/o Fault-Domain      - disable fault-domain isolation in repair
* w/o Deployment Risk   - drop the risk term from InsertCost
* w/o Dynamic Resched   - disable post-fault dynamic rescheduling preference
* w/o Hybrid            - force a single fault-tolerance mode

Note: ``gcn`` and ``recovery_penalty`` are fully wired in the ALNS core; the
remaining switches are passed through the ``ablation`` dict and take effect
where the operators/verifier consult them (see PROGRESS.md known-issues #2).

CLI::

    python -m gcnalns.experiments.run_ablation --instances data/instances --out results
"""
from __future__ import annotations

import argparse

from ..config import load_config
from .runner import run_experiment

VARIANTS = {
    "Full": {},
    "w/o GCN": {"gcn": False},
    "w/o Recovery Penalty": {"recovery_penalty": False},
    "w/o Fault-Domain": {"fault_domain": False},
    "w/o Deployment Risk": {"deployment_risk": False},
    "w/o Dynamic Resched": {"dynamic_resched": False},
    "w/o Hybrid": {"hybrid": False},
}


def main() -> None:
    ap = argparse.ArgumentParser(description="GCN-ALNS ablation study")
    ap.add_argument("--instances", nargs="+", default=["data/instances"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    for variant, ablation in VARIANTS.items():
        tag = "ablation_" + variant.replace("/", "").replace(" ", "_")
        run_experiment(args.instances, args.out, cfg,
                       algorithms=["GCN-ALNS"], repeats=args.repeats,
                       ablation=ablation, tag=tag,
                       algo_label_suffix=f" [{variant}]",
                       workers=args.workers, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
