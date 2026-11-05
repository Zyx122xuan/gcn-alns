"""Content identities for frozen data, model provenance and experiment resumes."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

from ..config import REPO_ROOT
from .io import load_json


def json_safe(value):
    """Keep manifests valid JSON, including configurations containing infinity."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def content_hash(value) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict:
    """Hash source contents as well as recording HEAD (includes uncommitted fixes)."""
    root = REPO_ROOT / "src" / "gcnalns"
    files = {p.relative_to(REPO_ROOT).as_posix(): file_hash(p)
             for p in sorted(root.rglob("*.py"))}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"sha256": content_hash(files), "git_commit": commit, "files": files}


def write_manifest(value: dict, path: str | Path) -> None:
    """Atomically publish a JSON sidecar outside raw_json aggregation inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def instance_record(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = load_json(path)
    workflow = {k: raw[k] for k in ("tasks", "edges", "data")}
    # Edge order has no semantic meaning. IDs and metadata are deliberately
    # excluded: renaming an instance or changing only its SLA cannot hide reuse.
    workflow["edges"] = sorted(workflow["edges"])
    resources = raw["nodes"]
    return {
        "iid": raw["iid"], "path": str(path), "sha256": file_hash(path),
        "workflow_sha256": content_hash(workflow),
        "resource_sha256": content_hash(resources),
        "content_sha256": content_hash({"workflow": workflow, "nodes": resources}),
    }


def collect_instances(directories) -> list[dict]:
    if isinstance(directories, (str, Path)):
        directories = [directories]
    paths = []
    for value in directories or []:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"Instance input does not exist: {path}")
        paths.extend([path] if path.is_file() else sorted(path.glob("*.json")))
    records = [instance_record(p) for p in paths]
    if not records:
        raise ValueError("No instance JSON files found")
    iids = [r["iid"] for r in records]
    if len(set(iids)) != len(iids):
        raise ValueError("Duplicate instance iid in inputs")
    return sorted(records, key=lambda r: r["iid"])


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    """Reject ID, workflow or resource-snapshot reuse between split roles."""
    fields = ("iid", "content_sha256", "workflow_sha256", "resource_sha256")
    owners = {field: {} for field in fields}
    for role, records in splits.items():
        if len({r["iid"] for r in records}) != len(records):
            raise ValueError(f"Duplicate instance iid in {role}")
        for record in records:
            for field in fields:
                value = record[field]
                previous = owners[field].get(value)
                if previous is not None and previous != role:
                    raise ValueError(
                        f"Data leakage: {record['iid']} shares {field} across "
                        f"{previous} and {role}"
                    )
                owners[field][value] = role
