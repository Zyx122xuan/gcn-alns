"""Read-only: the exact per-scale figures Appendix C cites for the external
baselines (feasible rate, all-run cost, feasible-only cost, safety, backup ratio)."""
import csv

d = list(csv.DictReader(open("results/csv/summary_by_algo_scale.csv",
                             encoding="utf-8-sig")))
ALGOS = ["RECR-D", "HFWPF", "GA-Placement", "RA-MOMA", "HEFT", "ALNS", "GCN-ALNS"]
HDR = ("scale", "algo", "feas", "cost_all", "cost_feas", "safety", "bkp_ratio")
print("".join(f"{h:>12}" for h in HDR))
for sc in ("small", "medium", "large"):
    for a in ALGOS:
        r = next((x for x in d if x["scale"] == sc and x["algorithm"] == a), None)
        if not r:
            continue
        def f(key, nd=1):
            try:
                return f"{float(r[key + '_mean']):.{nd}f}"
            except (ValueError, TypeError, KeyError):
                return "-"
        print(f"{sc:>12}{a:>12}{f('feasible', 3):>12}{f('cost_total'):>12}"
              f"{f('cost_total_feas'):>12}{f('safety_margin_feas', 2):>12}"
              f"{f('active_backup_ratio', 3):>12}")
