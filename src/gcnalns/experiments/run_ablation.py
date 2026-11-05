"""Ablation study (Spec sec.19.2): isolate each GCN-ALNS component.

Variants (toggled via the ALNS ``ablation`` dict):

* Full                  - the complete GCN-ALNS
* w/o GCN               - drop the learned guidance (plain ALNS search)
* w/o Recovery Penalty  - remove the R_max>budget penalty from the objective
* w/o Fault-Domain      - disable fault-domain isolation in repair
* w/o Deployment Risk   - drop the risk term from InsertCost
* w/o Dynamic Resched   - disable post-fault global rescheduling (local only)
* w/o Hybrid (all-passive) - force passive recompute for every task
* w/o Hybrid (all-active)  - force active backup for every task
* w/o Warm Start        - skip the GCN-guided greedy warm start (A1)
* w/o GCN Mode          - drop the learned FT-mode / replica guidance (A2)
* w/o GCN Destroy       - disable the low_gcn directed destroy operator (A3)
* w/o Feasibility-First - always optimise cost+penalty (drop B1 phase)
* w/o Restart           - no diversifying restart on stagnation (drop B2)
* w/o Init-Diversify    - single warm start, no HEFT-fastest candidate (drop B3)

All switches are now wired into the ALNS core / operators / verifier
(insertion.py, operators.py, recovery.py), so every variant differs from Full.

CLI::

    python -m gcnalns.experiments.run_ablation --instances data/instances --out results
"""
from __future__ import annotations

import argparse

from ..config import load_config
from .runner import add_instance_source_args, experiment_source, run_experiment

VARIANTS = {
    "Full": {},
    "w/o GCN": {"gcn": False},
    "w/o Recovery Penalty": {"recovery_penalty": False},
    "w/o Fault-Domain": {"fault_domain": False},
    "w/o Deployment Risk": {"deployment_risk": False},
    "w/o Dynamic Resched": {"dynamic_resched": False},
    "w/o Hybrid (all-passive)": {"hybrid": False, "hybrid_mode": "all-passive"},
    "w/o Hybrid (all-active)": {"hybrid": False, "hybrid_mode": "all-active"},
    "w/o Warm Start": {"warm_start": False},
    "w/o GCN Mode": {"gcn_mode": False},
    "w/o GCN Destroy": {"gcn_destroy": False},
    # B-series search improvements (isolate them from the learned GCN guidance).
    "w/o Feasibility-First": {"feasibility_first": False},
    "w/o Restart": {"restarts": False},
    "w/o Init-Diversify": {"init_diversify": False},
}


def main() -> None:
    ap = argparse.ArgumentParser(description="GCN-ALNS ablation study")
    add_instance_source_args(ap)
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    ap.add_argument("--model-path", dest="model_path", default=None,
                    help="GCN model checkpoint (default: cfg gcn.model_path)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="subset of variant names to run (default: all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    variants = VARIANTS
    if args.variants:
        unknown = [v for v in args.variants if v not in VARIANTS]
        if unknown:
            raise SystemExit(f"unknown variants: {unknown}\n"
                             f"available: {list(VARIANTS)}")
        variants = {v: VARIANTS[v] for v in args.variants}
    for variant, ablation in variants.items():
        tag = "ablation_" + variant.replace("/", "").replace(" ", "_")
        run_experiment(out_dir=args.out, cfg=cfg, **experiment_source(args),
                       algorithms=["GCN-ALNS"], repeats=args.repeats,
                       ablation=ablation, tag=tag,
                       algo_label_suffix=f" [{variant}]",
                       model_path=args.model_path,
                       workers=args.workers, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
