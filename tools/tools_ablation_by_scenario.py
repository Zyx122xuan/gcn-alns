"""Read-only: ablation feasible rate split by SLA scenario (rep10 data)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RAW = Path("results/raw_json")
VARIANTS = [
    "Full",
    "wo_GCN",
    "wo_GCN_Mode",
    "wo_GCN_Destroy",
    "wo_Warm_Start",
    "wo_Feasibility-First",
    "wo_Restart",
    "wo_Init-Diversify",
    "wo_Dynamic_Resched",
    "wo_Hybrid_(all-active)",
    "wo_Hybrid_(all-passive)",
    "wo_Recovery_Penalty",
    "wo_Deployment_Risk",
    "wo_Fault-Domain",
]
SCEN = ["tight", "medium", "loose"]


def main() -> None:
    print(f"{'variant':<26} " + " ".join(f"{s:>8}" for s in SCEN) + f" {'n/scen':>8}")
    for v in VARIANTS:
        p = RAW / f"ablation_{v}.jsonl"
        if not p.exists():
            print(f"{v:<26} MISSING")
            continue
        hit: dict[str, list[int]] = defaultdict(list)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            hit[r.get("sla_scenario", "?")].append(1 if r.get("feasible") else 0)
        row = []
        n = 0
        for s in SCEN:
            xs = hit.get(s, [])
            n = max(n, len(xs))
            row.append(f"{(sum(xs) / len(xs)):>8.3f}" if xs else f"{'-':>8}")
        print(f"{v:<26} " + " ".join(row) + f" {n:>8}")


if __name__ == "__main__":
    main()
