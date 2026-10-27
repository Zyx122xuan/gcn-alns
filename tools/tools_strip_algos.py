"""Drop stale records for given algorithms from a tag's raw_json, so the
resume-safe runner recomputes them. Backs up both files first."""
import json
import shutil
import sys
from pathlib import Path

algos = set(sys.argv[1].split(","))
tags = sys.argv[2:]
raw = Path("results/raw_json")
stamp = "_prerecrdfix"

for tag in tags:
    for suffix in (".jsonl", ".json"):
        path = raw / f"{tag}{suffix}"
        if not path.exists():
            continue
        backup = raw / f"{tag}{stamp}{suffix}"
        if not backup.exists():
            shutil.copy2(path, backup)
        if suffix == ".jsonl":
            kept = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and json.loads(ln)["algorithm"] not in algos]
            dropped = sum(1 for _ in path.read_text(encoding="utf-8").splitlines()
                          if _.strip()) - len(kept)
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        else:
            recs = json.loads(path.read_text(encoding="utf-8"))
            kept = [r for r in recs if r["algorithm"] not in algos]
            dropped = len(recs) - len(kept)
            path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
        print(f"{path}: dropped {dropped}, kept {len(kept)}")
