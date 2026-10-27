"""Compare the external baselines before/after the RECR-D fixes, using the
_prerecrdfix backups written by tools_strip_algos.py.

The backups live outside results/raw_json/ on purpose: eval.aggregate globs
every *.json under that directory, so leaving them there silently doubles the
run count and blends pre/post-fix records into the CSV summaries.
"""
import json
from collections import defaultdict
from pathlib import Path

raw = Path("results/raw_json")
old_raw = Path("results/raw_json_backup_prerecrdfix")
ALGOS = ("RECR-D", "HFWPF")
KEYS = ("feasible", "makespan_base", "cost_total", "safety_margin")


def load(path):
    out = defaultdict(list)
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r["algorithm"] in ALGOS:
            out[r["algorithm"]].append(r)
    return out


def mean(recs, key):
    vals = [r.get(key) for r in recs if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


print(f"{'tag':<12} {'algo':<8} {'feasible%':>19} {'makespan':>19} "
      f"{'cost_total':>19} {'safety':>19}")
print(f"{'':<12} {'':<8} " + " ".join([f"{'old -> new':>19}"] * 4))
for tag in ("main_small", "main_medium", "main_large"):
    old = load(old_raw / f"{tag}_prerecrdfix.jsonl")
    new = load(raw / f"{tag}.jsonl")
    for a in ALGOS:
        o, n = old[a], new[a]
        cells = []
        for k in KEYS:
            if k == "feasible":
                ov = 100 * sum(1 for r in o if r.get("feasible")) / max(len(o), 1)
                nv = 100 * sum(1 for r in n if r.get("feasible")) / max(len(n), 1)
            else:
                ov, nv = mean(o, k), mean(n, k)
            cells.append(f"{ov:8.1f} ->{nv:8.1f}")
        print(f"{tag:<12} {a:<8} " + " ".join(cells))
