"""Freeze learning/validation inputs and a fresh, unfiltered solver test set.

CLI: python -m gcnalns.data.protocol --learning-instances data/instances/small
     data/instances/medium --out data/protocols/independent_v1 --test-seed 12026
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from ..config import Config, load_config
from ..utils.io import load_json
from ..utils.provenance import (
    assert_disjoint, collect_instances, content_hash, instance_record, write_manifest,
)
from .instance_builder import generate_instances

ROLES = ("train", "val", "test")


def write_protocol(path: str | Path, splits: dict[str, list[dict]],
                   metadata: dict | None = None) -> dict:
    """Write once. Input files remain protected by their recorded hashes."""
    path = Path(path).resolve()
    if set(splits) != set(ROLES) or any(not splits[r] for r in ROLES):
        raise ValueError("Protocol requires nonempty train, val and test splits")
    assert_disjoint(splits)
    stored = {
        role: [{**r, "path": Path(os.path.relpath(r["path"], path.parent)).as_posix()}
               for r in sorted(splits[role], key=lambda x: x["iid"])]
        for role in ROLES
    }
    payload = {"schema_version": 1, "splits": stored, "metadata": metadata or {}}
    payload["sha256"] = content_hash(payload)
    if path.exists():
        if load_json(path) != payload:
            raise ValueError(f"Refusing to replace frozen protocol: {path}")
    else:
        write_manifest(payload, path)
    return payload


def load_protocol(path: str | Path) -> dict:
    """Validate the manifest, files, identities and all cross-split boundaries."""
    path = Path(path).resolve()
    payload = load_json(path)
    expected = payload.get("sha256")
    if payload.get("schema_version") != 1 or expected != content_hash(
            {k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError(f"Invalid or modified protocol manifest: {path}")
    splits = payload["splits"]
    if set(splits) != set(ROLES) or any(not splits[r] for r in ROLES):
        raise ValueError("Protocol requires nonempty train, val and test splits")
    resolved = {}
    for role, entries in splits.items():
        resolved[role] = []
        for stored in entries:
            current = instance_record(path.parent / stored["path"])
            if any(current[k] != stored[k] for k in current if k != "path"):
                raise ValueError(f"Frozen {role} instance changed: {stored['iid']}")
            resolved[role].append(current)
    assert_disjoint(resolved)
    return {"sha256": expected, "splits": resolved, "manifest": payload}


def _learning_split(records: list[dict], val_fraction: float, seed: int) -> dict:
    """Assign related SLA/resource variants together, before teacher filtering."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        for field in ("workflow_sha256", "resource_sha256"):
            key = (field, record[field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(find(i), []).append(record)
    blocks = list(groups.values())
    if len(blocks) < 2:
        raise ValueError("Need at least two independent groups for train/val")
    random.Random(seed).shuffle(blocks)
    target = max(1, round(len(records) * val_fraction))
    val, train = [], []
    for block in blocks[:-1]:
        (val if len(val) < target else train).extend(block)
    train.extend(blocks[-1])
    return {"train": train, "val": val}


def prepare_protocol(learning_dirs, out_dir: str | Path, cfg: Config, *,
                     test_seed: int = 12026, split_seed: int = 2026,
                     val_fraction: float = 0.2,
                     test_counts: dict[str, int] | None = None) -> Path:
    """Use existing inputs only for learning; generate fresh test inputs once."""
    out = Path(out_dir).resolve()
    manifest_path = out / "protocol.json"
    if manifest_path.exists():
        raise ValueError(f"Protocol already frozen; load it instead: {manifest_path}")
    learning = collect_instances(learning_dirs)
    splits = _learning_split(learning, val_fraction, split_seed)
    counts = test_counts or {"small": 45, "medium": 30, "large": 15}
    test_paths = []
    for index, (scale, count) in enumerate(counts.items()):
        test_paths.extend(generate_instances(
            cfg, out / "test" / scale, scale=scale, n=count,
            seed=test_seed + index * 100_000, iid_prefix="heldout", overwrite=False,
        ))
    splits["test"] = collect_instances(test_paths)
    write_protocol(manifest_path, splits, {
        "test_seed": test_seed, "scale_seed_stride": 100_000,
        "split_seed": split_seed, "val_fraction": val_fraction,
        "generation_config_sha256": content_hash(cfg.data),
        "test_selection": "all generated instances; no solver/feasibility filtering",
    })
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare an independent evaluation protocol")
    ap.add_argument("--learning-instances", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--test-seed", type=int, default=12026)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()
    path = prepare_protocol(args.learning_instances, args.out, load_config(args.config),
                            test_seed=args.test_seed, split_seed=args.split_seed,
                            val_fraction=args.val_fraction)
    protocol = load_protocol(path)
    print(f"Frozen protocol -> {path}")
    print({role: len(entries) for role, entries in protocol["splits"].items()})


if __name__ == "__main__":
    main()
