"""Read-only audit: re-derive the load-bearing numbers quoted in the two
reports straight from raw_json, and flag any that disagree."""
import json
from collections import defaultdict
from pathlib import Path

RAW = Path("results/raw_json")


def load(tag):
    return [json.loads(l) for l in
            (RAW / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def rate(recs):
    return 100 * sum(1 for r in recs if r.get("feasible")) / len(recs) if recs else float("nan")


def favg(recs, key):
    vals = [r[key] for r in recs
            if r.get("feasible") and isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


main = load("main_small") + load("main_medium") + load("main_large")
by_algo = defaultdict(list)
for r in main:
    by_algo[r["algorithm"]].append(r)

CLAIMS = []
g, a = by_algo["GCN-ALNS"], by_algo["ALNS"]
CLAIMS += [
    ("GCN 全局可行率 52.1%", rate(g), 52.1),
    ("ALNS 全局可行率 49.9%", rate(a), 49.9),
    ("GCN 全局成本 595.0", favg(g, "cost_total"), 595.0),
    ("ALNS 全局成本 620.7", favg(a, "cost_total"), 620.7),
    ("GCN 全局安全余量 3.65", favg(g, "safety_margin"), 3.65),
    ("ALNS 全局安全余量 2.71", favg(a, "safety_margin"), 2.71),
    ("GCN 达可行迭代 91.4", favg(g, "first_feasible_iter"), 91.4),
    ("ALNS 达可行迭代 113.8", favg(a, "first_feasible_iter"), 113.8),
    ("HEFT 全局可行率 17.8%", rate(by_algo["HEFT"]), 17.8),
    ("RECR-D 全局可行率 1.1%", rate(by_algo["RECR-D"]), 1.1),
    ("HFWPF 全局可行率 4.4%", rate(by_algo["HFWPF"]), 4.4),
    ("RA-MOMA 全局可行率 9.8%", rate(by_algo["RA-MOMA"]), 9.8),
]

# cost components must sum to total under the corrected model (no backup)
comp = sum(favg(g, k) for k in
           ("cost_compute", "cost_comm", "cost_reserve", "cost_recovery"))
CLAIMS.append(("GCN 四项成本之和 = 总成本", comp, favg(g, "cost_total")))

# per scale
for tag, sc, gf, af in (("main_small", "small", 44.2, 40.9),
                        ("main_medium", "medium", 57.0, 55.3),
                        ("main_large", "large", 80.0, 82.7)):
    recs = load(tag)
    sub = defaultdict(list)
    for r in recs:
        sub[r["algorithm"]].append(r)
    CLAIMS.append((f"GCN {sc} 可行率 {gf}%", rate(sub["GCN-ALNS"]), gf))
    CLAIMS.append((f"ALNS {sc} 可行率 {af}%", rate(sub["ALNS"]), af))

# hard tiers incl. large
for tag, sc, rows in (
        ("main_hard_small", "small", [("tight2", 22.7, 16.7), ("tight3", 84.0, 63.3)]),
        ("main_hard_medium", "medium", [("tight2", 56.0, 56.0), ("tight3", 100.0, 100.0)]),
        ("main_hard_large", "large", [("tight2", 92.0, 100.0), ("tight3", 100.0, 88.0)])):
    recs = load(tag)
    for scen, gf, af in rows:
        gg = [r for r in recs if r["algorithm"] == "GCN-ALNS" and r["sla_scenario"] == scen]
        aa = [r for r in recs if r["algorithm"] == "ALNS" and r["sla_scenario"] == scen]
        CLAIMS.append((f"难档 {sc}-{scen} GCN {gf}%", rate(gg), gf))
        CLAIMS.append((f"难档 {sc}-{scen} ALNS {af}%", rate(aa), af))

bad = 0
for label, got, want in CLAIMS:
    ok = abs(got - want) <= max(0.06, abs(want) * 0.002)
    if not ok:
        bad += 1
    print(f"{'OK ' if ok else 'BAD'} {label:<34} got={got:9.2f} want={want:9.2f}")
print(f"\n{len(CLAIMS) - bad}/{len(CLAIMS)} 条一致" + ("" if not bad else f"，{bad} 条不一致"))
