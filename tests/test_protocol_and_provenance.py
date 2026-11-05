"""Exercise leakage rejection and durable experiment resume boundaries."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gcnalns.config import Config, load_config
from gcnalns.core.model import Continuum, Instance, Node, Task, Workflow
from gcnalns.data.instance_builder import instance_to_dict
from gcnalns.data.protocol import _learning_split, load_protocol, write_protocol
from gcnalns.experiments import runner
from gcnalns.gcn import dataset
from gcnalns.utils.io import load_json, save_json
from gcnalns.utils.provenance import collect_instances, instance_record


@pytest.fixture
def cfg():
    data = copy.deepcopy(load_config().data)
    data["gcn"].update(hidden_dim=8, max_epochs=1, dropout=0.0)
    data["alns"]["max_iter"] = 0
    return Config(data)


def _instance(path, iid, value):
    tasks = {i: Task(tid=i, workload=value + i) for i in range(2)}
    nodes = {k: Node(nid=k, compute=value + k, cpu_cap=100, mem_cap=100,
                     load=0, load_var=0, failure_rate=0, cost_coef=1,
                     layer="cloud", fault_domain=f"cloud-{k}") for k in range(3)}
    inst = Instance(iid=iid, workflow=Workflow("unit", tasks, [(0, 1)], {(0, 1): 1}),
                    continuum=Continuum(nodes, {}), deadline=100, recovery_window=100,
                    meta={"scale": "small", "sla_scenario": "loose"})
    save_json(instance_to_dict(inst), path)
    return path


@pytest.fixture
def protocol_path(tmp_path):
    # Deliberately use filenames different from iid: identity is in the JSON.
    splits = {role: [instance_record(_instance(tmp_path / f"{role}.json", role, value))]
              for role, value in (("train", 10), ("val", 20), ("test", 30))}
    path = tmp_path / "protocol.json"
    write_protocol(path, splits)
    return path


@pytest.mark.parametrize("reuse", ["instance", "workflow", "resources"])
def test_rejects_renamed_or_partially_reused_inputs(protocol_path, reuse):
    protocol = load_protocol(protocol_path)
    splits = protocol["splits"]
    train = load_json(splits["train"][0]["path"])
    test_path = Path(splits["test"][0]["path"])
    test = load_json(test_path)
    if reuse == "instance":
        test = train
        test["iid"] = "renamed_test"
        test["workflow_name"] = "also-renamed"
        test["deadline"] = 200
    elif reuse == "workflow":
        for key in ("tasks", "edges", "data"):
            test[key] = train[key]
    else:
        test["nodes"] = train["nodes"]
    save_json(test, test_path)
    splits["test"] = [instance_record(test_path)]
    with pytest.raises(ValueError, match="Data leakage"):
        write_protocol(protocol_path.parent / "leaked.json", splits)


def test_protocol_detects_edited_input_and_refuses_replacement(protocol_path):
    protocol = load_protocol(protocol_path)
    with pytest.raises(ValueError, match="replace frozen"):
        write_protocol(protocol_path, protocol["splits"], metadata={"edited": True})
    path = Path(protocol["splits"]["test"][0]["path"])
    raw = load_json(path)
    raw["deadline"] += 1
    save_json(raw, path)
    with pytest.raises(ValueError, match="Frozen test instance changed"):
        load_protocol(protocol_path)


def test_learning_split_keeps_sla_variants_together(tmp_path):
    paths = [_instance(tmp_path / f"{i}.json", str(i), 10 + i) for i in range(4)]
    duplicate = load_json(paths[0])
    duplicate["iid"] = "sla_variant"
    duplicate["deadline"] = 200
    save_json(duplicate, tmp_path / "variant.json")
    records = collect_instances([*paths, tmp_path / "variant.json"])
    splits = _learning_split(records, 0.5, 2026)
    owner = {r["iid"]: role for role, rows in splits.items() for r in rows}
    assert owner["0"] == owner["sla_variant"]
    assert splits["train"] and splits["val"]


def test_teacher_never_receives_solver_test_or_moves_failed_samples(
        protocol_path, cfg, monkeypatch):
    seen = []

    def teacher(path, *_):
        iid = load_json(path)["iid"]
        seen.append(iid)
        assert iid != "test"
        return None if iid == "train" else SimpleNamespace(pos=[(0, 0)],
                                                          n_feasible_solutions=1)

    monkeypatch.setattr(dataset, "build_instance_sample", teacher)
    monkeypatch.setattr(dataset, "_worker_init", lambda: None)
    data = dataset.build_dataset(None, cfg, workers=1, protocol_path=protocol_path)
    assert seen == ["train", "val"]
    assert data["split"] == {"train": [], "val": ["val"], "test": []}
    assert [r["iid"] for r in data["provenance"]["teacher_inputs"]] == ["train", "val"]
    assert [r["iid"] for r in data["provenance"]["source_splits"]["test"]] == ["test"]


def test_final_test_refuses_legacy_checkpoint(protocol_path, cfg, tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"legacy checkpoint without provenance")
    with pytest.raises(ValueError, match="requires checkpoint training provenance"):
        runner.run_experiment(None, tmp_path / "results", cfg,
                              algorithms=["GCN-ALNS"], model_path=str(checkpoint),
                              protocol_path=protocol_path, split="test", workers=1)


def test_training_checkpoint_final_test_and_resume(protocol_path, cfg, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from gcnalns.gcn.train import save_model, train

    data = dataset.build_dataset(None, cfg, workers=1, alns_iter=0,
                                 protocol_path=protocol_path)
    model, meta = train(data, cfg, verbose=False)
    checkpoint = tmp_path / "model.pt"
    save_model(model, meta, cfg, checkpoint)
    sidecar = load_json(str(checkpoint) + ".provenance.json")
    assert sidecar["training"]["sample_splits"] == {
        "train": ["train"], "val": ["val"], "test": [],
    }
    assert sidecar["training"]["dataset"]["config"]["alns"]["max_iter"] == 0
    kwargs = dict(instances_dir=None, out_dir=tmp_path / "results", cfg=cfg,
                  algorithms=["HEFT", "GCN-ALNS"], model_path=str(checkpoint),
                  protocol_path=protocol_path, split="test", workers=1, repeats=1,
                  max_iter=0, verbose=False)
    first = runner.run_experiment(**kwargs)
    assert len(first) == 2 and {r["iid"] for r in first} == {"test"}
    for record in first:
        assert record["evaluation_split"] == "test"
        saved = load_json(kwargs["out_dir"] / record["solution_path"])
        assert saved["iid"] == "test" and saved["placement"]
        assert saved["experiment_id"] == record["experiment_id"]
    manifest = load_json(kwargs["out_dir"] / "manifests/main.json")
    assert manifest["model"]["sha256"] == sidecar["checkpoint_sha256"]

    def should_not_schedule(*args, **kw):
        pytest.fail("An unchanged experiment must resume without scheduling")

    monkeypatch.setattr(runner, "make_scheduler", should_not_schedule)
    assert runner.run_experiment(**kwargs) == first

    # Even a provenance sidecar with the matching checkpoint hash must not
    # certify a model which was exposed to the final test inputs.
    sidecar["training"]["dataset"]["teacher_inputs"].append(
        load_protocol(protocol_path)["splits"]["test"][0])
    save_json(sidecar, str(checkpoint) + ".provenance.json")
    with pytest.raises(ValueError, match="Data leakage"):
        runner.run_experiment(**kwargs)


@pytest.mark.parametrize("change", ["config", "instance", "code", "solution"])
def test_resume_refuses_changed_experiment_inputs(tmp_path, cfg, monkeypatch, change):
    path = _instance(tmp_path / "input.json", "unit", 10)
    kwargs = dict(instances_dir=[path], out_dir=tmp_path / "results", cfg=cfg,
                  algorithms=["HEFT"], workers=1, repeats=1, verbose=False)
    first = runner.run_experiment(**kwargs)
    if change == "config":
        cfg.data["recovery"]["delta_detect"] += 0.1
    elif change == "instance":
        raw = load_json(path)
        raw["deadline"] += 1
        save_json(raw, path)
    elif change == "code":
        monkeypatch.setattr(runner, "code_version", lambda: {"sha256": "new-code"})
    else:
        target = kwargs["out_dir"] / first[0]["solution_path"]
        target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        runner.run_experiment(**kwargs)


def test_resume_refuses_changed_model(tmp_path, cfg, monkeypatch):
    # A scheduler stub isolates the resume check from torch deserialization.
    from gcnalns.algorithms.heft import HEFT
    monkeypatch.setattr(runner, "make_scheduler", lambda *a, **kw: HEFT(cfg))
    path = _instance(tmp_path / "input.json", "unit", 10)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"version-one")
    kwargs = dict(instances_dir=[path], out_dir=tmp_path / "results", cfg=cfg,
                  algorithms=["GCN-ALNS"], model_path=str(checkpoint),
                  workers=1, repeats=1, verbose=False)
    runner.run_experiment(**kwargs)
    checkpoint.write_bytes(b"version-two")
    with pytest.raises(ValueError, match="model/code changed"):
        runner.run_experiment(**kwargs)


def test_legacy_results_are_not_silently_resumed(tmp_path, cfg):
    path = _instance(tmp_path / "input.json", "unit", 10)
    out = tmp_path / "results"
    save_json([{"tag": "main", "algorithm": "HEFT", "iid": "unit", "rep": 0}],
              out / "raw_json/main.json")
    with pytest.raises(ValueError, match="Legacy results lack"):
        runner.run_experiment([path], out, cfg, algorithms=["HEFT"], workers=1)


def test_output_cannot_mix_validation_and_final_test(protocol_path, cfg, tmp_path):
    kwargs = dict(instances_dir=None, out_dir=tmp_path / "results", cfg=cfg,
                  algorithms=["HEFT"], protocol_path=protocol_path, workers=1,
                  verbose=False)
    runner.run_experiment(**kwargs, split="val", tag="validation")
    with pytest.raises(ValueError, match="Output evaluation split/protocol changed"):
        runner.run_experiment(**kwargs, split="test", tag="final")


def test_solver_test_includes_infeasible_instances(protocol_path, cfg, tmp_path):
    splits = load_protocol(protocol_path)["splits"]
    extra_path = _instance(tmp_path / "infeasible.json", "infeasible-test", 40)
    raw = load_json(extra_path)
    raw["recovery_window"] = 0.0
    save_json(raw, extra_path)
    splits["test"].append(instance_record(extra_path))
    path = tmp_path / "unfiltered_protocol.json"
    write_protocol(path, splits)
    rows = runner.run_experiment(None, tmp_path / "results", cfg,
                                 protocol_path=path, split="test", algorithms=["HEFT"],
                                 workers=1, verbose=False)
    assert {r["iid"] for r in rows} == {"test", "infeasible-test"}
    assert not next(r for r in rows if r["iid"] == "infeasible-test")["feasible"]


def test_parameter_sweep_cannot_tune_final_test(cfg, tmp_path):
    from gcnalns.experiments.run_sensitivity import sweep
    with pytest.raises(ValueError, match="Tune parameters"):
        sweep(None, tmp_path, cfg, "insert_cost.w6", [0.5], ["ALNS"], 1,
              protocol_path="unused", split="test")


def test_scale_selection_stays_in_original_protocol(protocol_path, cfg, tmp_path):
    splits = load_protocol(protocol_path)["splits"]
    path = _instance(tmp_path / "medium.json", "medium-test", 50)
    raw = load_json(path)
    raw["meta"]["scale"] = "medium"
    save_json(raw, path)
    splits["test"].append(instance_record(path))
    protocol = tmp_path / "multiscale.json"
    frozen = write_protocol(protocol, splits)
    out = tmp_path / "results"
    rows = runner.run_experiment(None, out, cfg, protocol_path=protocol, split="test",
                                 instance_scales=["medium"], algorithms=["HEFT"],
                                 workers=1, verbose=False)
    assert [r["iid"] for r in rows] == ["medium-test"]
    assert rows[0]["protocol_sha256"] == frozen["sha256"]
    with pytest.raises(ValueError, match="changed"):
        runner.run_experiment(None, out, cfg, protocol_path=protocol, split="test",
                              instance_scales=["small"], algorithms=["HEFT"],
                              workers=1, verbose=False)
