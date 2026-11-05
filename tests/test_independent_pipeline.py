"""Registered study scopes, validation-only selection and report statistics."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_helper(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_helper("independent_driver", "scripts/rerun_independent_v1.py")
reporter = load_helper("independent_report", "tools/report_independent_v1.py")


def row(iid, rep, feasible=True, cost=10, role="val"):
    return {"iid": iid, "rep": rep, "feasible": feasible, "cost_total": cost,
            "evaluation_split": role}


def test_model_selection_prioritizes_validation_feasibility():
    on = [row("a", 0, True, 1), row("b", 0, False, 1)]
    off = [row("a", 0, True, 100), row("b", 0, True, 100)]
    assert driver.choose_model(on, off)["selected"] == "a2_off"


def test_teacher_progress_decodes_powershell_and_python_logs(tmp_path):
    value = "Building teacher\n  [ok] sample: 2 positives\nSaved dataset\n"
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        path = tmp_path / f"{encoding}.log"
        path.write_text(value, encoding=encoding)
        assert driver.read_log(path).count("  [ok]") == 1
        assert "Saved dataset" in driver.read_log(path)


def test_model_selection_uses_paired_cost_then_fixed_tie_break():
    on = [row("a", 0, cost=10), row("b", 0, False, 1)]
    off = [row("a", 0, cost=20), row("b", 0, False, 999)]
    value = driver.choose_model(on, off)
    assert value["selected"] == "a2_on" and value["n_common_feasible"] == 1
    assert value["paired_relative_cost_on_minus_off"] == pytest.approx(-0.5)
    assert driver.choose_model(on, on)["reason"] == "registered deterministic tie-break"


def test_model_selection_rejects_test_results_or_unpaired_seeds():
    with pytest.raises(ValueError, match="validation-only"):
        driver.choose_model([row("a", 0, role="test")], [row("a", 0)])
    with pytest.raises(ValueError, match="paired"):
        driver.choose_model([row("a", 0)], [row("a", 1)])


def test_original_experiment_budgets_count_20550_test_runs(tmp_path):
    entries = []
    for scale, n in (("small", 45), ("medium", 30), ("large", 15)):
        path = tmp_path / f"{scale}.json"
        path.write_text(json.dumps({"meta": {"scale": scale}}), encoding="utf-8")
        entries.extend([{"path": str(path)}] * n)
    counts = driver.study_counts({"splits": {"test": entries}})
    assert counts == {"main": {"small": 2250, "medium": 1500, "large": 450},
                      "ablation": 10500, "budget": 4500, "model_ab": 1350}
    assert sum(counts["main"].values()) + sum(v for k, v in counts.items() if k != "main") == 20550


def test_running_teacher_only_allows_unused_frontend_edits():
    old = {"src/gcnalns/core/recovery.py": "same", "src/gcnalns/experiments/runner.py": "old"}
    new = {**old, "src/gcnalns/experiments/runner.py": "new"}
    assert driver.teacher_source_changes(old, new) == ["src/gcnalns/experiments/runner.py"]
    with pytest.raises(ValueError, match="Teacher computation source changed"):
        driver.teacher_source_changes(old, {**new, "src/gcnalns/core/recovery.py": "different"})


def test_paired_statistics_use_instances_not_seed_pseudoreplication():
    a = pd.DataFrame([row("a", 0, True, 10), row("a", 1, False, 999),
                      row("b", 0), row("b", 1)])
    b = pd.DataFrame([row("a", 0, True, 20), row("a", 1, True, 20),
                      row("b", 0, False), row("b", 1, False)])
    feasible = reporter.paired(a, b, "feasible")
    assert feasible["n_instances"] == 2 and feasible["n_paired_runs"] == 4
    assert feasible["mean_difference"] == pytest.approx(25)
    cost = reporter.paired(a, b, "cost_total", relative=True)
    assert cost["n_instances"] == 1 and cost["n_paired_runs"] == 1
    assert cost["mean_difference"] == pytest.approx(-50)
    assert cost["p_t"] is None
    assert reporter.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_report_builds_only_from_complete_new_protocol_runs(tmp_path):
    raw = tmp_path / "test/raw_json"
    raw.mkdir(parents=True)
    plan = {"protocol_sha256": "protocol", "code": {"sha256": "code"},
            "expected_test_records": {"main": {"small": 2, "medium": 2, "large": 2},
                                      "ablation": 2, "budget": 2, "model_ab": 3}}
    (tmp_path / "pipeline_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "model_selection.json").write_text(
        json.dumps({"selected": "a2_on", "reason": "validation"}), encoding="utf-8")

    def save(tag, scale, algorithms):
        rows = [{**row(scale, 0, role="test"), "algorithm": algo, "tag": tag,
                 "scale": scale, "runtime": 1.0, "safety_margin": 2.0,
                 "first_feasible_iter": 0, "protocol_sha256": "protocol",
                 "code_sha256": "code"} for algo in algorithms]
        (raw / f"{tag}.json").write_text(json.dumps(rows), encoding="utf-8")

    for scale in ("small", "medium", "large"):
        save(f"main_{scale}", scale, ["ALNS", "GCN-ALNS"])
    save("ablation_Full", "small", ["GCN-ALNS [Full]"])
    save("ablation_wo_GCN", "small", ["GCN-ALNS [w/o GCN]"])
    save("budget_50", "small", ["ALNS", "GCN-ALNS"])
    for name in ("alns", "a2_off", "a2_on"):
        save(f"model_ab_{name}", "small", [name])
    value = reporter.build_report(tmp_path, plots=False)
    assert sum(value["run_counts"].values()) == 13
    assert (tmp_path / "REPORT.md").is_file()
    (raw / "budget_50.json").unlink()
    with pytest.raises(ValueError, match="Incomplete budget"):
        reporter.build_report(tmp_path, plots=False)


def test_coordinator_stage_resumes_and_detects_output_changes(tmp_path):
    pipeline = driver.Pipeline.__new__(driver.Pipeline)
    pipeline.root = tmp_path
    pipeline.logs = tmp_path / "logs"
    pipeline.logs.mkdir()
    pipeline.env = os.environ.copy()
    pipeline.version = {"sha256": "test-version"}
    pipeline.plan = {"driver_sha256": "test-driver"}
    pipeline.protocol = {"sha256": "test-protocol"}
    pipeline.check_sources = lambda: None
    output = tmp_path / "probe.txt"
    command = [sys.executable, "-c",
               f"from pathlib import Path; Path({str(output)!r}).write_text('completed')"]
    pipeline.stage("probe", command, [output])
    assert output.read_text() == "completed"
    first = (tmp_path / "stages/probe.json").read_bytes()
    pipeline.stage("probe", command, [output])
    assert (tmp_path / "stages/probe.json").read_bytes() == first
    output.write_text("changed")
    with pytest.raises(ValueError, match="Stage output changed"):
        pipeline.stage("probe", command, [output])
