"""Read-only audit of frozen experiment inputs; write derived analysis here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/independent_v1"
OUT = Path(__file__).resolve().parent
SCALES = ("small", "medium", "large")
FIELDS = ("feasible", "cost_total", "safety_margin", "first_feasible_iter")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [None] * len(values)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[i]))
        adjusted[i] = previous
    return adjusted


def paired(left, right, field):
    keys = ["iid", "rep"]
    data = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if field != "feasible":
        data = data[data.feasible_a & data.feasible_b]
    a, b = data[field + "_a"].astype(float), data[field + "_b"].astype(float)
    good = np.isfinite(a) & np.isfinite(b)
    if field == "first_feasible_iter":
        good &= (a >= 0) & (b >= 0)
    data = data[good]
    differences = a[good] - b[good]
    if field == "feasible":
        differences *= 100
    elif field == "cost_total":
        differences = 100 * differences / b[good].abs().clip(lower=1e-12)
    values = differences.groupby(data.iid).mean()
    if len(values) < 2:
        raise ValueError("At least two paired instances required for this analysis")
    rng = np.random.default_rng(2026)
    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    null = np.allclose(values, 0, rtol=0, atol=1e-12)
    return {
        "n_instances": len(values), "n_paired_runs": len(data),
        "mean_difference": float(values.mean()),
        "median_difference": float(values.median()),
        "ci95": np.quantile(draws, [.025, .975]).tolist(),
        "p_t": 1.0 if null else float(stats.ttest_1samp(values, 0).pvalue),
        "p_wilcoxon": 1.0 if null else float(stats.wilcoxon(values).pvalue),
        "negative_instances": int((values < -1e-9).sum()),
        "positive_instances": int((values > 1e-9).sum()),
        "tied_instances": int((abs(values) <= 1e-9).sum()),
        "by_instance": {str(i): float(v) for i, v in values.items()},
    }


def summarize(frame):
    feasible = frame[frame.feasible]
    cost = feasible.groupby("iid").cost_total.mean()
    return {
        "n_runs": len(frame), "n_instances": int(frame.iid.nunique()),
        "n_feasible_runs": int(frame.feasible.sum()),
        "feasible_percent": float(100 * frame.groupby("iid").feasible.mean().mean()),
        "cost_feasible_run_mean": float(feasible.cost_total.mean()) if len(feasible) else None,
        "cost_feasible_instance_mean": float(cost.mean()) if len(cost) else None,
        "runtime_mean_s": float(frame.runtime.mean()),
    }


def main():
    summary = read(SOURCE / "summary.json")
    plan = read(SOURCE / "pipeline_plan.json")
    state = read(SOURCE / "pipeline_state.json")
    assert state["state"] == "complete"
    groups = {}
    ledger = {}
    for tag, n in summary["run_counts"].items():
        path = SOURCE / "test/raw_json" / (tag + ".json")
        frame = pd.DataFrame(read(path))
        assert len(frame) == n
        assert not frame.duplicated(["tag", "algorithm", "iid", "rep"]).any()
        assert (frame.evaluation_split == "test").all()
        assert (frame.protocol_sha256 == plan["protocol_sha256"]).all()
        assert (frame.code_sha256 == plan["code"]["sha256"]).all()
        stage = read(SOURCE / "stages" / (tag + ".json"))
        assert stage["state"] == "complete"
        groups[tag] = frame
        ledger[str(path.relative_to(ROOT))] = digest(path)
    assert sum(map(len, groups.values())) == 20550
    source_files = dict(plan["code"]["files"])
    source_files["scripts/rerun_independent_v1.py"] = plan["driver_sha256"]
    source_files["tools/report_independent_v1.py"] = plan["report_sha256"]
    assert all(digest(ROOT / name) == sha for name, sha in source_files.items())
    main_pairs, main_summary, breakdowns = {}, {}, {}
    for scale in SCALES:
        frame = groups["main_" + scale]
        a, b = (frame[frame.algorithm == alg] for alg in ("GCN-ALNS", "ALNS"))
        for field in FIELDS:
            key = scale + "/" + field
            result = paired(a, b, field)
            expected = summary["main_paired"][key]
            for name in ("n_instances", "n_paired_runs", "mean_difference", "p_t", "p_wilcoxon"):
                assert np.isclose(result[name], expected[name]), (key, name)
            assert np.allclose(result["ci95"], expected["ci95"])
            main_pairs[key] = result
        main_summary[scale] = {str(alg): summarize(rows) for alg, rows in frame.groupby("algorithm")}
        runtime = {alg: main_summary[scale][alg]["runtime_mean_s"] for alg in ("ALNS", "GCN-ALNS")}
        main_summary[scale]["runtime_overhead_percent"] = 100 * (runtime["GCN-ALNS"] / runtime["ALNS"] - 1)
        main_summary[scale]["gcn_inference_mean_s"] = float(a.gcn_infer_time.mean())
        breakdowns[scale] = {}
        for field in ("sla_scenario", "workflow_type"):
            breakdowns[scale][field] = []
            for (value, alg), rows in frame[frame.algorithm.isin(["ALNS", "GCN-ALNS"])].groupby([field, "algorithm"]):
                breakdowns[scale][field].append({"group": value, "algorithm": alg, **summarize(rows)})
    # Post-hoc robustness check, explicitly separate from the registered tests.
    for key, adjusted in zip(main_pairs, holm([r["p_t"] for r in main_pairs.values()])):
        main_pairs[key]["p_t_holm12_sensitivity"] = adjusted
    budgets = {}
    for n in (50, 100, 200, 500, 1000):
        frame = groups[f"budget_{n}"]
        a, b = (frame[frame.algorithm == alg] for alg in ("GCN-ALNS", "ALNS"))
        budgets[str(n)] = {
            "ALNS": summarize(b), "GCN-ALNS": summarize(a),
            "paired_feasibility": paired(a, b, "feasible"),
            "paired_cost": paired(a, b, "cost_total"),
        }
    ablations = {}
    for tag, frame in groups.items():
        if not tag.startswith("ablation_"):
            continue
        result = {"summary": summarize(frame)}
        if tag != "ablation_Full":
            result["paired"] = {}
            for field in FIELDS:
                value = paired(groups["ablation_Full"], frame, field)
                expected = summary["ablation_paired"][tag][field]
                for name in ("n_instances", "n_paired_runs", "mean_difference", "p_t"):
                    assert np.isclose(value[name], expected[name]), (tag, field, name)
                value["p_t_holm"] = expected["p_t_holm"]
                result["paired"][field] = value
        ablations[tag] = result
    overlap = {}
    main_small_medium = pd.concat([groups["main_small"], groups["main_medium"]])
    for tag, alg in (("ablation_Full", "GCN-ALNS"), ("ablation_wo_GCN", "ALNS")):
        joined = groups[tag].merge(main_small_medium[main_small_medium.algorithm == alg], on=["iid", "rep"], suffixes=("_a", "_b"), validate="one_to_one")
        overlap[tag] = {field: bool(np.allclose(joined[field + "_a"].astype(float), joined[field + "_b"].astype(float), equal_nan=True)) for field in FIELDS}
        assert len(joined) == 750
    policy_audit = {}
    for tag, require_active in (("ablation_wo_Hybrid_(all-active)", True),
                                ("ablation_wo_Hybrid_(all-passive)", False)):
        frame = groups[tag]
        strict_all, strict_feasible = 0, 0
        for row in frame.itertuples(index=False):
            path = Path(row.solution_path)
            if not path.is_absolute():
                path = SOURCE / "test" / path
            assert digest(path) == row.solution_sha256
            solution = read(path)
            mode = solution["mode"]
            active = sum(bool(v) for v in mode.values())
            assert np.isclose(active / row.num_tasks, row.active_backup_ratio)
            placements = solution["placement"]
            strict = len(placements) == row.num_tasks and len(mode) == row.num_tasks
            strict &= all(bool(mode.get(i, False)) == require_active and
                          (len(nodes) >= 2 if require_active else len(nodes) == 1)
                          for i, nodes in placements.items())
            strict_all += bool(strict)
            strict_feasible += bool(strict and row.feasible)
        feasible = frame[frame.feasible]
        policy_audit[tag] = {
            "n_runs": len(frame), "n_feasible": len(feasible),
            "strict_policy_all_runs": strict_all,
            "strict_policy_feasible_runs": strict_feasible,
            "violating_policy_feasible_runs": len(feasible) - strict_feasible,
            "feasible_active_fraction_min": float(feasible.active_backup_ratio.min()),
            "feasible_active_fraction_max": float(feasible.active_backup_ratio.max()),
            "feasible_active_fraction_mean": float(feasible.active_backup_ratio.mean()),
            "saved_solution_hashes_verified": len(frame),
            "interpretation": "Repair-mode preference variant; not a strictly enforced homogeneous deployment policy.",
        }
    ledger["results/independent_v1/summary.json"] = digest(SOURCE / "summary.json")
    output = {
        "analysis_date": "2026-11-05", "protocol_sha256": plan["protocol_sha256"],
        "selected_model": summary["model_selection"]["selected"],
        "main": main_summary, "main_paired": main_pairs, "ablation": ablations,
        "budgets": budgets, "exploratory_breakdowns": breakdowns,
        "shared_main_ablation_outcomes": overlap,
        "protection_policy_output_audit": policy_audit,
        "audit": {"test_records": 20550, "paper_study_records": 19200,
                  "internal_diagnostic_records": 1350, "frozen_files_verified": len(source_files),
                  "source_file_sha256": ledger,
                  "notes": "Instance is the independent unit. Conditional metrics use matched successful iid/rep runs. Holm12 is a post-hoc sensitivity check, not the registered main testing plan."},
    }
    (OUT / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Verified all 20,550 test rows and reproduced registered main/ablation paired means and tests.")
    print("Frozen experiment files verified:", len(source_files))
    for scale in SCALES:
        r = main_pairs[scale + "/cost_total"]
        print(scale, "paired cost %", round(r["mean_difference"], 4), "Holm12 sensitivity p", r["p_t_holm12_sensitivity"])
    print("Derived metrics:", OUT / "metrics.json")


if __name__ == "__main__":
    main()
