"""Durable train -> validation -> frozen test -> component-study pipeline.

Run from any directory. The existing teacher job is awaited, not duplicated.
Every subprocess has an independent log and verified, resumable stage record.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcnalns.config import load_config
from gcnalns.data.protocol import load_protocol
from gcnalns.experiments.run_ablation import VARIANTS
from gcnalns.experiments.runner import DEFAULT_ALGORITHMS, STOCHASTIC
from gcnalns.utils.provenance import code_version, content_hash, file_hash, json_safe, write_manifest

TEACHER_UNUSED_EDITS = {
    "src/gcnalns/experiments/runner.py",
    "src/gcnalns/experiments/run_scalability.py",
    "src/gcnalns/experiments/run_sensitivity.py",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_log(path):
    """Windows PowerShell Tee-Object writes UTF-16; subprocess logs use UTF-8."""
    path = Path(path)
    if not path.exists():
        return ""
    data = path.read_bytes()
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return data.decode(encoding, errors="replace")


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json_safe(value):
            raise ValueError(f"Frozen pipeline artifact changed: {path}")
    else:
        write_manifest(value, path)


def teacher_source_changes(before, after):
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if set(changed) - TEACHER_UNUSED_EDITS:
        raise ValueError(f"Teacher computation source changed: {changed}")
    return changed


def choose_model(on, off):
    """Registered rule: validation feasibility, paired cost, deterministic tie."""
    if not on or not off or any(r.get("evaluation_split") != "val" for r in on + off):
        raise ValueError("Model selection requires nonempty validation-only records")
    a = {(r["iid"], r["rep"]): r for r in on}
    b = {(r["iid"], r["rep"]): r for r in off}
    if len(a) != len(on) or len(b) != len(off) or set(a) != set(b):
        raise ValueError("Model validation runs are not identically paired")
    feasible_on = sum(bool(r["feasible"]) for r in a.values())
    feasible_off = sum(bool(r["feasible"]) for r in b.values())
    common = [key for key in a if a[key]["feasible"] and b[key]["feasible"]]
    paired_relative_cost = (sum((a[k]["cost_total"] - b[k]["cost_total"])
                               / max(abs(b[k]["cost_total"]), 1e-12) for k in common)
                            / len(common)) if common else None
    if feasible_on != feasible_off:
        selected = "a2_on" if feasible_on > feasible_off else "a2_off"
        reason = "higher validation feasibility"
    elif paired_relative_cost is not None and abs(paired_relative_cost) > 1e-9:
        selected = "a2_on" if paired_relative_cost < 0 else "a2_off"
        reason = "lower cost on identically paired feasible validation runs"
    else:
        selected, reason = "a2_on", "registered deterministic tie-break"
    return {"selected": selected, "reason": reason, "n_validation_runs_per_model": len(a),
            "feasible_on": feasible_on, "feasible_off": feasible_off,
            "n_common_feasible": len(common), "paired_relative_cost_on_minus_off": paired_relative_cost}


def study_counts(protocol):
    scales = {scale: sum(json.loads(Path(r["path"]).read_text(encoding="utf-8"))
                         ["meta"]["scale"] == scale for r in protocol["splits"]["test"])
              for scale in ("small", "medium", "large")}
    main = {scale: n * sum((5 if scale == "large" else 10) if a in STOCHASTIC else 1
                          for a in DEFAULT_ALGORITHMS) for scale, n in scales.items()}
    return {"main": main, "ablation": (scales["small"] + scales["medium"]) * 10 * len(VARIANTS),
            "budget": scales["small"] * 10 * 2 * 5,
            "model_ab": scales["small"] * 10 * 3}


class Pipeline:
    def __init__(self, root, workers=8, large_workers=12, device="auto"):
        self.root = Path(root).resolve()
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.workers, self.large_workers, self.device = workers, large_workers, device
        self.protocol_path = ROOT / "data/protocols/independent_v1/protocol.json"
        self.dataset_path = ROOT / "data/gcn_independent_v1/dataset.pkl"
        self.protocol = load_protocol(self.protocol_path)
        self.request = json.loads((self.logs / "teacher_build_request.json").read_text(encoding="utf-8"))
        self.version = code_version()
        self.teacher_edits = teacher_source_changes(self.request["code_at_start"]["files"],
                                                    self.version["files"])
        self.plan = {
            "schema_version": 1, "protocol_sha256": self.protocol["sha256"],
            "code": self.version, "driver_sha256": file_hash(__file__),
            "report_sha256": file_hash(ROOT / "tools/report_independent_v1.py"),
            "workers": workers, "large_workers": large_workers,
            "device_policy": device, "base_config": self.request["config"],
            "teacher_unused_infrastructure_edits": self.teacher_edits,
            "validation": {"repeats": 5, "max_iter": 1000, "models": ["a2_on", "a2_off"],
                           "selection_rule": "feasibility count; matched feasible relative cost; a2_on tie"},
            "test_main": {"algorithms": DEFAULT_ALGORITHMS,
                          "small": {"repeats": 10, "max_iter": 1000},
                          "medium": {"repeats": 10, "max_iter": 1000},
                          "large": {"repeats": 5, "max_iter": 800}},
            "ablation": {"scales": ["small", "medium"], "repeats": 10,
                         "max_iter": 1000, "variants": VARIANTS},
            "budget": {"scales": ["small"], "repeats": 10,
                       "budgets": [50, 100, 200, 500, 1000]},
            "model_ab": {"scales": ["small"], "repeats": 10, "max_iter": 1000},
            "expected_test_records": study_counts(self.protocol),
        }
        frozen_json(self.root / "pipeline_plan.json", self.plan)
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1",
                        PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.env[key] = "1"

    def status(self, phase, state="running", **extra):
        write_manifest({"pid": os.getpid(), "updated_utc": now(), "phase": phase,
                        "state": state, "protocol_sha256": self.protocol["sha256"], **extra},
                       self.root / "pipeline_state.json")

    def check_sources(self):
        if (code_version()["sha256"] != self.version["sha256"]
                or file_hash(__file__) != self.plan["driver_sha256"]
                or file_hash(ROOT / "tools/report_independent_v1.py") != self.plan["report_sha256"]):
            raise ValueError("Pipeline source changed; refusing to mix versions")

    def await_teacher(self):
        print("WAIT teacher build already running", flush=True)
        log_path = self.logs / "teacher_build.log"
        while True:
            text = read_log(log_path)
            progress = {"teacher_finished": text.count("  [ok]") + text.count("  [skip]"),
                        "teacher_with_demonstrations": text.count("  [ok]")}
            self.status("teacher", "waiting", **progress)
            if self.dataset_path.exists() and "Saved " in text and "split: train=" in text:
                break
            if "Traceback (most recent call last)" in text:
                raise RuntimeError("Teacher build failed; see logs/teacher_build.log")
            time.sleep(30)
        self.check_sources()
        with self.dataset_path.open("rb") as fh:
            data = pickle.load(fh)
        provenance = data.get("provenance") or {}
        expected = {r["iid"] for role in ("train", "val") for r in self.protocol["splits"][role]}
        actual = [r["iid"] for r in provenance.get("teacher_inputs", [])]
        if (provenance.get("protocol_sha256") != self.protocol["sha256"]
                or len(actual) != len(expected) or set(actual) != expected
                or provenance.get("alns_iter") is not None
                or content_hash(provenance.get("config")) != content_hash(self.request["config"])):
            raise ValueError("Teacher dataset provenance does not match the registered request")
        if data["split"].get("test") or not data["split"].get("train") or not data["split"].get("val"):
            raise ValueError("Teacher must have train/val demonstrations and no solver-test samples")
        for role in ("train", "val"):
            if not set(data["split"][role]).issubset({r["iid"] for r in self.protocol["splits"][role]}):
                raise ValueError(f"Teacher split violation: {role}")
        teacher_source_changes(self.request["code_at_start"]["files"],
                               provenance["code"]["files"])
        self.dataset_hash = file_hash(self.dataset_path)
        summary = {"state": "complete", "dataset_sha256": self.dataset_hash,
                   "protocol_sha256": self.protocol["sha256"],
                   "n_teacher_inputs": len(actual),
                   "n_samples": len(data["samples"]),
                   "split_counts": {k: len(v) for k, v in data["split"].items()},
                   "source_changes_limited_to_unused_experiment_infrastructure": self.teacher_edits}
        write_manifest(summary, self.logs / "teacher_build_status.json")
        print("TEACHER VERIFIED", summary, flush=True)

    def config_file(self, name, values):
        import yaml
        path = self.root / "configs" / f"{name}.yaml"
        content = yaml.safe_dump(json_safe(values), sort_keys=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Frozen configuration changed: {path}")
        path.write_text(content, encoding="utf-8")
        return path

    def stage(self, name, command, outputs, inputs=()):
        self.check_sources()
        signature = content_hash({"command": list(map(str, command)),
                                  "inputs": {str(p): file_hash(p) for p in inputs},
                                  "code_sha256": self.version["sha256"],
                                  "driver_sha256": self.plan["driver_sha256"]})
        stage_file = self.root / "stages" / f"{name}.json"
        previous = json.loads(stage_file.read_text(encoding="utf-8")) if stage_file.exists() else None
        if previous and previous["signature"] != signature:
            raise ValueError(f"Stage inputs changed: {name}")
        if previous and previous.get("state") == "complete":
            for path, digest in previous["outputs"].items():
                if not Path(path).is_file() or file_hash(path) != digest:
                    raise ValueError(f"Stage output changed: {path}")
                if "/raw_json/" in Path(path).as_posix():
                    for record in json.loads(Path(path).read_text(encoding="utf-8")):
                        solution = Path(path).parent.parent / record["solution_path"]
                        if not solution.is_file() or file_hash(solution) != record["solution_sha256"]:
                            raise ValueError(f"Saved solution changed: {solution}")
            print("RESUME completed", name, flush=True)
            return
        if not previous and any(Path(p).exists() for p in outputs):
            raise ValueError(f"Unowned outputs already exist for {name}")
        state = {"signature": signature, "state": "running", "started_utc": now(),
                 "command": list(map(str, command))}
        write_manifest(state, stage_file)
        log = self.logs / f"{name}.log"
        print("START", name, flush=True)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log.open("a", encoding="utf-8") as sink:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=self.env,
                                     stdout=sink, stderr=subprocess.STDOUT, creationflags=flags)
            while True:
                self.status(name, child_pid=child.pid, log=str(log))
                try:
                    code = child.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if code:
            write_manifest({**state, "state": "failed", "exit_code": code}, stage_file)
            raise RuntimeError(f"Stage {name} failed (exit {code}); see {log}")
        self.check_sources()
        digests = {str(p): file_hash(p) for p in outputs}
        write_manifest({**state, "state": "complete", "completed_utc": now(),
                        "outputs": digests}, stage_file)
        print("DONE", name, flush=True)

    def experiment(self, name, tag, config, algorithms, *, role="test", scales=None,
                   repeats=10, iterations=1000, model=None, extra=()):
        out = self.root / ("validation" if role == "val" else "test")
        command = [sys.executable, "-u", "-m", "gcnalns.experiments.run_main",
                   "--protocol", self.protocol_path, "--split", role, "--out", out,
                   "--config", config, "--algorithms", *algorithms, "--repeats", repeats,
                   "--max-iter", iterations, "--tag", tag, "--workers",
                   self.large_workers if scales == ["large"] else self.workers]
        inputs = [config]
        if model:
            command += ["--model-path", model]
            inputs += [model, Path(str(model) + ".provenance.json")]
        if scales:
            command += ["--instance-scales", *scales]
        command += list(extra)
        self.stage(name, command, [out / "raw_json" / f"{tag}.json",
                                  out / "manifests" / f"{tag}.json"], inputs)

    def run(self):
        self.await_teacher()
        import torch
        device = ("cuda:0" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        configs, models = {}, {}
        for name, weight in (("a2_on", 0.5), ("a2_off", 0.0)):
            values = copy.deepcopy(self.request["config"])
            values["gcn"]["mode_loss_weight"] = weight
            configs[name] = self.config_file(name, values)
            models[name] = self.root / "models" / f"{name}.pt"
            self.stage(f"train_{name}", [sys.executable, "-u", "-m", "gcnalns.gcn.train",
                       "--data", self.dataset_path.parent, "--out", models[name],
                       "--config", configs[name], "--device", device],
                       [models[name], Path(str(models[name]) + ".provenance.json")],
                       [self.dataset_path, configs[name]])
        self.experiment("validation_alns", "validation_alns", configs["a2_on"], ["ALNS"],
                        role="val", repeats=5)
        for name in ("a2_on", "a2_off"):
            self.experiment(f"validation_{name}", f"validation_{name}", configs[name],
                            ["GCN-ALNS"], role="val", repeats=5, model=models[name])
        validation = self.root / "validation/raw_json"
        on = json.loads((validation / "validation_a2_on.json").read_text(encoding="utf-8"))
        off = json.loads((validation / "validation_a2_off.json").read_text(encoding="utf-8"))
        selection = choose_model(on, off)
        chosen = selection["selected"]
        selection.update(protocol_sha256=self.protocol["sha256"], dataset_sha256=self.dataset_hash,
                         model_sha256=file_hash(models[chosen]),
                         validation_outputs={p.name: file_hash(p) for p in validation.glob("*.json")})
        frozen_json(self.root / "model_selection.json", selection)
        selected_model = self.root / "gcn_model.pt"
        for source, target in ((models[chosen], selected_model),
                               (Path(str(models[chosen]) + ".provenance.json"),
                                Path(str(selected_model) + ".provenance.json"))):
            if target.exists() and file_hash(target) != file_hash(source):
                raise ValueError(f"Selected model output changed: {target}")
            shutil.copy2(source, target)
        selected_config = configs[chosen]
        print("MODEL FROZEN", selection, flush=True)
        for scale, repeats, iterations in (("small", 10, 1000), ("medium", 10, 1000),
                                            ("large", 5, 800)):
            self.experiment(f"main_{scale}", f"main_{scale}", selected_config,
                            DEFAULT_ALGORITHMS, scales=[scale], repeats=repeats,
                            iterations=iterations, model=selected_model)
        # Stage 4: all scopes/budgets were registered before any test results.
        for variant in VARIANTS:
            tag = "ablation_" + variant.replace("/", "").replace(" ", "_")
            out = self.root / "test"
            self.stage(tag, [sys.executable, "-u", "-m", "gcnalns.experiments.run_ablation",
                       "--protocol", self.protocol_path, "--split", "test",
                       "--instance-scales", "small", "medium", "--out", out,
                       "--config", selected_config, "--model-path", selected_model,
                       "--variants", variant, "--repeats", "10", "--max-iter", "1000",
                       "--workers", self.workers],
                       [out / "raw_json" / f"{tag}.json", out / "manifests" / f"{tag}.json"],
                       [selected_config, selected_model, Path(str(selected_model) + ".provenance.json")])
        for budget in (50, 100, 200, 500, 1000):
            self.experiment(f"budget_{budget}", f"budget_{budget}", selected_config,
                            ["ALNS", "GCN-ALNS"], scales=["small"], iterations=budget,
                            model=selected_model)
        for name in ("a2_off", "a2_on"):
            self.experiment(f"model_ab_{name}", f"model_ab_{name}", configs[name],
                            ["GCN-ALNS"], scales=["small"], model=models[name],
                            extra=["--algo-suffix", f" ({name})"])
        self.experiment("model_ab_alns", "model_ab_alns", selected_config,
                        ["ALNS"], scales=["small"])
        self.stage("report", [sys.executable, ROOT / "tools/report_independent_v1.py",
                   "--root", self.root], [self.root / "REPORT.md", self.root / "summary.json"],
                   [self.root / "model_selection.json"])
        self.status("complete", "complete", report=str(self.root / "REPORT.md"))
        print("PIPELINE COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT / "results/independent_v1"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--large-workers", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Windows releases this OS lock even if the process dies; stale PID files
    # therefore cannot prevent a later resume or allow duplicate coordinators.
    lock = (root / "pipeline.lock").open("a+b")
    if os.name == "nt":
        import msvcrt
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Another coordinator is already running")
    pipeline = None
    try:
        pipeline = Pipeline(root, args.workers, args.large_workers, args.device)
        if args.prepare_only:
            print("PLAN PREPARED", pipeline.plan["expected_test_records"], flush=True)
            return
        pipeline.run()
    except Exception as exc:
        if pipeline is not None:
            pipeline.status("failed", "failed", error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
