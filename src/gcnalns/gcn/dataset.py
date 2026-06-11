"""Build the GCN training dataset from solver demonstrations (Spec sec.17).

For every training instance we run a panel of schedulers
(HEFT / Reactive-Only / Full-Proactive / Criticality-Based Hybrid / ALNS),
verify each with the shared recovery verifier, keep the *feasible* solutions and
take the lowest-cost ``high_quality_top_frac`` of them as high-quality
demonstrations. Their (task -> hosting-node) decisions become positive samples;
feasible-but-unchosen nodes become negatives (``neg_per_pos`` sampled per
positive at train time). Infeasible candidates are masked out.

Instances (not individual samples) are split train/val/test so no instance
leaks across splits.

CLI::

    python -m gcnalns.gcn.dataset --instances data/instances --out data/gcn
"""
from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..algorithms.baselines import (
    CriticalityHybrid,
    FullProactive,
    NoFT,
    ReactiveOnly,
)
from ..config import Config, load_config
from ..core.cost import compute_cost
from ..core.recovery import verify
from ..core.schedule_eval import compute_schedule
from ..data.instance_builder import load_instance
from ..data.links import get_factory
from .features import InstanceFeatures, build_features

# Schedulers used to generate demonstrations. ALNS is added separately (it has a
# different constructor and is the strongest teacher).
_DEMO_BASELINES = [NoFT, ReactiveOnly, FullProactive, CriticalityHybrid]


@dataclass
class InstanceSample:
    feats: InstanceFeatures
    pos: np.ndarray            # [K, 2] (task_pos, node_pos) positive pairs
    feas_mask: np.ndarray      # [T, N] uint8 feasible candidate mask
    n_feasible_solutions: int


def _feasible_mask(feats: InstanceFeatures) -> np.ndarray:
    """A node is a feasible candidate for a task if cpu/mem/deadline bits all hold."""
    names = ("cpu_fit", "mem_fit", "deadline_fit")
    from .features import INTERACTION_FEATURE_NAMES
    idx = [INTERACTION_FEATURE_NAMES.index(n) for n in names]
    bits = feats.inter[:, :, idx]            # [T, N, 3]
    return (bits.min(axis=2) > 0.5).astype(np.uint8)


def _run_panel(instance, cfg: Config, alns_iter: int | None):
    """Return list of (cost, solution) for feasible solutions on ``instance``."""
    factory = get_factory(instance, cfg)
    out = []
    schedulers = [cls(cfg) for cls in _DEMO_BASELINES]
    for sch in schedulers:
        sol = sch.schedule(instance)
        sched = compute_schedule(instance, sol, factory, cfg)
        res = verify(instance, sol, factory, cfg, sched)
        if res.feasible:
            bd = compute_cost(instance, sol, sched, factory, cfg,
                              recovery_cost=res.recovery_cost)
            out.append((bd.total, sol))

    # ALNS (strongest teacher); optionally cap iterations for speed
    from ..algorithms.alns.solver import ALNS
    alns = ALNS(cfg)
    if alns_iter is not None:
        alns.max_iter = alns_iter
    sol = alns.schedule(instance)
    sched = compute_schedule(instance, sol, factory, cfg)
    res = verify(instance, sol, factory, cfg, sched)
    if res.feasible:
        bd = compute_cost(instance, sol, sched, factory, cfg,
                          recovery_cost=res.recovery_cost)
        out.append((bd.total, sol))
    return out, factory


def build_instance_sample(path: Path, cfg: Config,
                          alns_iter: int | None) -> InstanceSample | None:
    instance = load_instance(path, cfg)
    feats = build_features(instance, cfg, get_factory(instance, cfg))
    feasible, _ = _run_panel(instance, cfg, alns_iter)
    if not feasible:
        return None

    feasible.sort(key=lambda x: x[0])
    frac = float(cfg.get("gcn.high_quality_top_frac", 0.2))
    n_keep = max(1, int(math.ceil(frac * len(feasible))))
    high_quality = feasible[:n_keep]

    pos: set[tuple[int, int]] = set()
    for _, sol in high_quality:
        for tid, repl in sol.placement.items():
            tp = feats.task_pos.get(tid)
            if tp is None:
                continue
            for k in repl:
                kp = feats.node_pos.get(k)
                if kp is not None:
                    pos.add((tp, kp))

    feas_mask = _feasible_mask(feats)
    # ensure positives are marked feasible (chosen => feasible)
    for (tp, kp) in pos:
        feas_mask[tp, kp] = 1

    pos_arr = np.array(sorted(pos), dtype=np.int64) if pos else np.zeros((0, 2), np.int64)
    return InstanceSample(feats=feats, pos=pos_arr, feas_mask=feas_mask,
                          n_feasible_solutions=len(feasible))


def split_instances(iids: list[str], cfg: Config) -> dict[str, list[str]]:
    """Deterministic 70/15/15 split over instances (Spec sec.17)."""
    sp = cfg["gcn"]["split"]
    rng = np.random.default_rng(int(cfg.get("seeds.split", 2026)))
    order = list(iids)
    rng.shuffle(order)
    n = len(order)
    n_tr = int(round(sp["train"] * n))
    n_val = int(round(sp["val"] * n))
    return {
        "train": order[:n_tr],
        "val": order[n_tr:n_tr + n_val],
        "test": order[n_tr + n_val:],
    }


def build_dataset(instances_dir: str | Path, cfg: Config,
                  alns_iter: int | None = None) -> dict:
    instances_dir = Path(instances_dir)
    paths = sorted(instances_dir.glob("*.json"))
    samples: dict[str, InstanceSample] = {}
    for p in paths:
        s = build_instance_sample(p, cfg, alns_iter)
        iid = p.stem
        if s is None:
            print(f"  [skip] {iid}: no feasible solution")
            continue
        samples[iid] = s
        print(f"  [ok]   {iid}: {len(s.pos)} positives, "
              f"{s.n_feasible_solutions} feasible solutions")
    split = split_instances(list(samples.keys()), cfg)
    return {"samples": samples, "split": split,
            "num_domains": _num_domains(cfg)}


def _num_domains(cfg: Config) -> int:
    from .features import num_domains
    return num_domains(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GCN training dataset")
    ap.add_argument("--instances", default="data/instances")
    ap.add_argument("--out", default="data/gcn")
    ap.add_argument("--config", default=None)
    ap.add_argument("--alns-iter", type=int, default=None,
                    help="cap ALNS iterations during demonstration generation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Building GCN dataset from {args.instances} ...")
    data = build_dataset(args.instances, cfg, args.alns_iter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dataset.pkl", "wb") as fh:
        pickle.dump(data, fh)
    sp = data["split"]
    print(f"Saved {len(data['samples'])} instances -> {out/'dataset.pkl'}")
    print(f"  split: train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")


if __name__ == "__main__":
    main()
