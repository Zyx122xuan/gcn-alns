"""Read-only: reconcile the external-baseline feasible counts between the
.jsonl stream, the consolidated .json, and the aggregated CSV."""
import csv
import json
from pathlib import Path

raw = Path("results/raw_json")
for algo in ("RECR-D", "HFWPF"):
    tot = {"jsonl": [0, 0], "json": [0, 0]}
    for tag in ("main_small", "main_medium", "main_large"):
        for kind in ("jsonl", "json"):
            path = raw / f"{tag}.{kind}"
            if kind == "jsonl":
                recs = [json.loads(l) for l in
                        path.read_text(encoding="utf-8").splitlines() if l.strip()]
            else:
                recs = json.loads(path.read_text(encoding="utf-8"))
            sub = [r for r in recs if r["algorithm"] == algo]
            tot[kind][0] += len(sub)
            tot[kind][1] += sum(1 for r in sub if r.get("feasible"))
    row = next(r for r in csv.DictReader(
        open("results/csv/summary_by_algorithm.csv", encoding="utf-8-sig"))
        if r["algorithm"] == algo)
    print(f"{algo}: jsonl {tot['jsonl'][1]}/{tot['jsonl'][0]}  "
          f"json {tot['json'][1]}/{tot['json'][0]}  "
          f"csv feasible_mean={row['feasible_mean']} n={row['feasible_count']}")
