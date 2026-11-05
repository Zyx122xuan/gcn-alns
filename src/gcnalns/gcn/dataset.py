"""Build the GCN training dataset from solver demonstrations (Spec sec.17).

For every training instance we run a panel of schedulers
(HEFT / Reactive-Only / Full-Proactive / Criticality-Based Hybrid / ALNS),
verify each with the shared recovery verifier, keep the *feasible* solutions and
take the lowest-cost ``high_quality_top_frac`` of them as high-quality
demonstrations. Their (task -> hosting-node) decisions become positive samples;
feasible-but-unchosen nodes become negatives (``neg_per_pos`` sampled per
positive at train time). Infeasible candidates are masked out.

With --protocol, only frozen train/val instances enter the teacher. The solver
test set remains complete and untouched. Legacy development datasets still
support a local train/val/test split, assigned before teacher filtering.

CLI::

    python -m gcnalns.gcn.dataset --instances data/instances --out data/gcn
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from ..data.protocol import load_protocol
from ..utils.provenance import code_version, collect_instances, content_hash
from .features import InstanceFeatures, build_features
from .samples import InstanceSample

# Schedulers used to generate demonstrations. ALNS is added separately (it has a
# different constructor and is the strongest teacher).
_DEMO_BASELINES = [NoFT, ReactiveOnly, FullProactive, CriticalityHybrid]


def _feasible_mask(feats: InstanceFeatures) -> np.ndarray:
    """A node is a feasible candidate for a task if cpu/mem/deadline bits all hold."""
    names = ("cpu_fit", "mem_fit", "deadline_fit")
    from .features import INTERACTION_FEATURE_NAMES
    idx = [INTERACTION_FEATURE_NAMES.index(n) for n in names]
    bits = feats.inter[:, :, idx]            # [T, N, 3]
    return (bits.min(axis=2) > 0.5).astype(np.uint8)


def _run_panel(instance, cfg: Config, alns_iter: int | None):
    """Return list of ``(cost, solution, r_max, r_eff)`` for feasible solutions."""
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
            out.append((bd.total, sol, res.r_max, res.r_eff))

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
        out.append((bd.total, sol, res.r_max, res.r_eff))
    return out, factory


def _solution_safety(r_max: float, r_eff: float) -> float:
    """Recovery safety factor (1 - R_max/Delta_bar), clamped to [0, 1]."""
    if r_eff <= 1e-9:
        return 1.0 if r_max <= 1e-9 else 0.0
    return float(min(1.0, max(0.0, 1.0 - r_max / r_eff)))


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

    # Weighted labels (Modeling doc sec.13): a (task -> node) choice from a
    # high-quality solution is weighted by (1 + h_i) * (1 - R_max/Delta_bar) so
    # the GCN prioritises critical tasks placed in solutions with large safety
    # margin. A pair appearing in several solutions keeps its largest weight.
    tasks = instance.workflow.tasks
    pos_w: dict[tuple[int, int], float] = {}
    # FT-mode label per pair, taken from the solution that gave its max weight
    # (Part C): 1 => the task used an active backup, 0 => passive recompute.
    pos_mode: dict[tuple[int, int], float] = {}
    for _, sol, r_max, r_eff in high_quality:
        safety = _solution_safety(r_max, r_eff)
        for tid, repl in sol.placement.items():
            tp = feats.task_pos.get(tid)
            if tp is None:
                continue
            h_i = tasks[tid].criticality
            w = (1.0 + h_i) * safety
            active = 1.0 if sol.is_active(tid) else 0.0
            for k in repl:
                kp = feats.node_pos.get(k)
                if kp is not None:
                    key = (tp, kp)
                    if key not in pos_w or w > pos_w[key]:
                        pos_w[key] = w
                        pos_mode[key] = active

    pos = sorted(pos_w.keys())
    feas_mask = _feasible_mask(feats)
    # ensure positives are marked feasible (chosen => feasible)
    for (tp, kp) in pos:
        feas_mask[tp, kp] = 1

    if pos:
        pos_arr = np.array(pos, dtype=np.int64)
        w_arr = np.array([pos_w[p] for p in pos], dtype=np.float32)
        mode_arr = np.array([pos_mode[p] for p in pos], dtype=np.float32)
    else:
        pos_arr = np.zeros((0, 2), np.int64)
        w_arr = np.zeros((0,), np.float32)
        mode_arr = np.zeros((0,), np.float32)
    return InstanceSample(feats=feats, pos=pos_arr, feas_mask=feas_mask,
                          n_feasible_solutions=len(feasible), pos_w=w_arr,
                          pos_mode=mode_arr)


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


def _worker_init() -> None:
    """Limit per-worker BLAS / torch threads to avoid oversubscription."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _build_sample_worker(task: tuple):
    """Top-level worker (spawn-safe): build one instance sample."""
    path_str, cfg_data, alns_iter, iid = task
    cfg = Config(cfg_data)
    sample = build_instance_sample(Path(path_str), cfg, alns_iter)
    return iid, sample


def build_dataset(instances_dir, cfg: Config,
                  alns_iter: int | None = None,
                  workers: int | None = None,
                  protocol_path: str | Path | None = None) -> dict:
    if protocol_path is not None:
        if instances_dir is not None:
            raise ValueError("Use protocol_path or instances_dir, not both")
        protocol = load_protocol(protocol_path)
        source_splits = protocol["splits"]
        # The solver test set is not labelled, filtered or fed to the teacher.
        records = source_splits["train"] + source_splits["val"]
        assigned = {role: [r["iid"] for r in source_splits[role]]
                    for role in ("train", "val")}
        assigned["test"] = []
    else:
        protocol = None
        records = collect_instances(instances_dir)
        assigned = split_instances([r["iid"] for r in records], cfg)
        source_splits = {role: [r for r in records if r["iid"] in ids]
                         for role, ids in assigned.items()}
    paths = [Path(r["path"]) for r in records]
    workers = workers if workers is not None else min(30, os.cpu_count() or 1)
    cfg_data = copy.deepcopy(cfg.data)
    samples: dict[str, InstanceSample] = {}

    def _record(iid: str, s) -> None:
        if s is None:
            print(f"  [skip] {iid}: no feasible solution")
            return
        samples[iid] = s
        print(f"  [ok]   {iid}: {len(s.pos)} positives, "
              f"{s.n_feasible_solutions} feasible solutions")

    if workers <= 1:
        _worker_init()
        for record, p in zip(records, paths):
            _record(record["iid"], build_instance_sample(p, cfg, alns_iter))
    else:
        tasks = [(str(p), cfg_data, alns_iter, r["iid"]) for r, p in zip(records, paths)]
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_worker_init) as ex:
            futures = [ex.submit(_build_sample_worker, t) for t in tasks]
            for fut in as_completed(futures):
                iid, s = fut.result()
                _record(iid, s)

    samples = {k: samples[k] for k in sorted(samples)}  # stable ordering
    # Keep assignments fixed even when the teacher fails to find a feasible
    # demonstration. In particular, never replace missing train/val with test.
    split = {role: [iid for iid in ids if iid in samples]
             for role, ids in assigned.items()}
    return {"samples": samples, "split": split,
            "num_domains": _num_domains(cfg),
            "provenance": {
                "protocol_sha256": protocol["sha256"] if protocol else None,
                "source_splits": source_splits,
                "teacher_inputs": records,
                "config": copy.deepcopy(cfg.data),
                "config_sha256": content_hash(cfg.data),
                "code": code_version(), "alns_iter": alns_iter,
            }}


def _num_domains(cfg: Config) -> int:
    from .features import num_domains
    return num_domains(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GCN training dataset")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--instances", nargs="+", default=None)
    source.add_argument("--protocol", default=None)
    ap.add_argument("--out", default="data/gcn")
    ap.add_argument("--config", default=None)
    ap.add_argument("--alns-iter", type=int, default=None,
                    help="cap ALNS iterations during demonstration generation")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    instances = args.instances if args.protocol else (args.instances or ["data/instances"])
    print(f"Building GCN dataset from {args.protocol or instances} ...")
    data = build_dataset(instances, cfg, args.alns_iter, workers=args.workers,
                         protocol_path=args.protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dataset.pkl", "wb") as fh:
        pickle.dump(data, fh)
    sp = data["split"]
    print(f"Saved {len(data['samples'])} instances -> {out/'dataset.pkl'}")
    print(f"  split: train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")


if __name__ == "__main__":
    main()
