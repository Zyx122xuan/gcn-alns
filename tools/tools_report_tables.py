"""Read-only: dump the exact summary_by_* fields the REPORT sections cite,
so §2-§6 can be refreshed from the regenerated CSVs."""
import csv
from pathlib import Path

CSV = Path("results/csv")
ORDER = ["GCN-ALNS", "ALNS", "HEFT", "No-FT", "PEFT", "Reactive-Only",
         "Reliability-Aware", "Critical-Path Backup", "Criticality-Based Hybrid",
         "Full-Proactive", "RECR-D", "HFWPF", "GA-Placement", "RA-MOMA"]


def rows(name):
    with open(CSV / name, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def g(r, key, nd=2):
    v = r.get(key, "")
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


print("=== summary_by_algorithm (global, three scales) ===")
hdr = ["feasible", "active_backup_ratio", "prune_ratio", "gcn_infer_time",
       "convergence_iter", "first_feasible_iter_feas", "cost_total_feas",
       "safety_margin_feas", "r_max_feas", "avg_recovery_feas", "c_base",
       "c_worst_feas", "n_faults", "runtime", "verify_time"]
print(f"{'algorithm':<26}" + "".join(f"{h[:13]:>15}" for h in hdr))
byname = {r["algorithm"]: r for r in rows("summary_by_algorithm.csv")}
for a in ORDER:
    r = byname.get(a)
    if not r:
        continue
    nd = {"feasible": 3, "active_backup_ratio": 3, "prune_ratio": 3,
          "gcn_infer_time": 3}
    print(f"{a:<26}" + "".join(f"{g(r, h + '_mean', nd.get(h, 2)):>15}" for h in hdr))

for csvname, keycol in (("summary_by_algo_scale.csv", "scale"),
                        ("summary_by_algo_scenario.csv", "sla_scenario")):
    print(f"\n=== {csvname} ===")
    data = rows(csvname)
    rank = ["small", "medium", "large", "tight", "loose"]
    keys = sorted({r[keycol] for r in data},
                  key=lambda s: rank.index(s) if s in rank else 99)
    fields = ["feasible", "cost_total_feas", "safety_margin_feas",
              "first_feasible_iter_feas", "r_max_feas", "avg_recovery_feas",
              "c_base", "c_worst_feas", "runtime", "verify_time",
              "cost_compute", "cost_comm", "cost_reserve", "cost_recovery"]
    for k in keys:
        print(f"\n--- {keycol}={k} ---")
        print(f"{'algorithm':<26}" + "".join(f"{f[:13]:>15}" for f in fields))
        sub = {r["algorithm"]: r for r in data if r[keycol] == k}
        for a in ORDER:
            r = sub.get(a)
            if not r:
                continue
            nd = {"feasible": 3}
            print(f"{a:<26}"
                  + "".join(f"{g(r, f + '_mean', nd.get(f, 2)):>15}" for f in fields))
