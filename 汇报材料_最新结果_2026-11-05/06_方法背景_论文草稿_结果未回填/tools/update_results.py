#!/usr/bin/env python3
"""Export completed, provenance-checked main results to the IEEE draft.

Run from the repository with Python 3.10+; only the standard library is used.
This script never launches experiments or writes to the experiment directory.
Incomplete stages, legacy results, changed inputs, and unpaired seed grids are
errors.  ``--check-only`` performs validation without publishing paper files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from datetime import datetime, timezone


SCALES = ("small", "medium", "large")
ALGORITHMS = (
    "HEFT", "PEFT", "No-FT", "Reactive-Only", "Full-Proactive",
    "Critical-Path Backup", "Criticality-Based Hybrid", "Reliability-Aware",
    "RECR-D", "HFWPF", "GA-Placement", "RA-MOMA", "ALNS", "GCN-ALNS",
)
# This is the frozen runner's seed policy, not a dependency on its Python code.
STOCHASTIC = frozenset(("GA-Placement", "RA-MOMA", "ALNS", "GCN-ALNS"))
IDENTITIES = ("iid", "sha256", "workflow_sha256", "resource_sha256", "content_sha256")
PAPER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PAPER_ROOT.parent.parent


class ValidationError(ValueError):
    """The recorded study cannot safely be exported."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def content_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file(), f"Required completed artifact is missing: {path}")
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def same(actual, expected, context):
    def preview(value):
        text = repr(value)
        return text if len(text) <= 240 else text[:237] + "..."
    require(actual == expected,
            f"Mismatch in {context}: expected {preview(expected)}, got {preview(actual)}")


def integer(value, context):
    require(type(value) is int and value > 0, f"{context} must be a positive integer")
    return value


def identity(record):
    return {key: record[key] for key in IDENTITIES}


def checked_path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def command_values(command, flag):
    require(command.count(flag) == 1, f"Stage command must contain {flag} exactly once")
    start = command.index(flag) + 1
    stop = start
    while stop < len(command) and not command[stop].startswith("--"):
        stop += 1
    require(stop > start, f"Stage command has no value for {flag}")
    return command[start:stop]


def command_value(command, flag):
    values = command_values(command, flag)
    require(len(values) == 1, f"Stage command has multiple values for {flag}")
    return values[0]


def check_code(version, expected, context):
    same(version.get("sha256"), expected["sha256"], f"{context} code fingerprint")
    same(version.get("files"), expected["files"], f"{context} source file identities")
    same(content_hash(version["files"]), version["sha256"], f"{context} code content hash")


def load_protocol(path, expected_sha):
    protocol = read_json(path)
    same(protocol.get("schema_version"), 1, "protocol schema")
    same(protocol.get("sha256"), expected_sha, "plan/protocol identity")
    same(content_hash({k: v for k, v in protocol.items() if k != "sha256"}),
         expected_sha, "protocol content hash")
    splits = protocol["splits"]
    same(set(splits), {"train", "val", "test"}, "protocol split roles")
    owners = {key: {} for key in IDENTITIES if key != "sha256"}
    resolved, raw_instances = {}, {}
    for role, entries in splits.items():
        require(entries, f"Protocol {role} split is empty")
        resolved[role] = {}
        for entry in entries:
            iid = entry["iid"]
            require(iid not in resolved[role], f"Duplicate protocol iid: {iid}")
            current_path = checked_path(entry["path"], path.parent)
            raw = read_json(current_path)
            workflow = {key: raw[key] for key in ("tasks", "edges", "data")}
            workflow["edges"] = sorted(workflow["edges"])
            current = {
                "iid": raw["iid"], "sha256": file_hash(current_path),
                "workflow_sha256": content_hash(workflow),
                "resource_sha256": content_hash(raw["nodes"]),
                "content_sha256": content_hash({"workflow": workflow, "nodes": raw["nodes"]}),
            }
            same(current, identity(entry), f"frozen {role} instance {iid}")
            for field, seen in owners.items():
                previous = seen.get(current[field])
                require(previous is None or previous == role,
                        f"Cross-split reuse of {field} for {iid}: {previous}/{role}")
                seen[current[field]] = role
            resolved[role][iid] = current
            raw_instances[iid] = raw
    return resolved, raw_instances


def check_completed_stages(root):
    stages, incomplete = {}, []
    for scale in SCALES:
        tag = f"main_{scale}"
        path = root / "stages" / f"{tag}.json"
        if not path.is_file():
            incomplete.append(f"{tag}: missing stage")
            continue
        stage = read_json(path)
        if stage.get("state") != "complete":
            incomplete.append(f"{tag}: {stage.get('state', 'unknown state')}")
        else:
            stages[scale] = stage
    require(not incomplete, "Three-scale main results are incomplete (" + "; ".join(incomplete) + "). No paper files were written.")
    return stages


def check_selection(root, plan, splits):
    selection = read_json(root / "model_selection.json")
    same(selection.get("protocol_sha256"), plan["protocol_sha256"], "model-selection protocol")
    chosen = selection.get("selected")
    require(chosen in ("a2_on", "a2_off"), "Unknown selected model variant")
    model = root / "gcn_model.pt"
    sidecar = Path(str(model) + ".provenance.json")
    model_sha = file_hash(model)
    same(model_sha, selection.get("model_sha256"), "selected checkpoint")
    same(file_hash(root / "models" / f"{chosen}.pt"), model_sha, "selected variant checkpoint")
    same(file_hash(root / "models" / f"{chosen}.pt.provenance.json"),
         file_hash(sidecar), "selected variant training sidecar")
    provenance = read_json(sidecar)
    same(provenance.get("schema_version"), 1, "checkpoint provenance schema")
    same(provenance.get("checkpoint_sha256"), model_sha, "checkpoint provenance identity")
    training = provenance["training"]
    check_code(training["code"], plan["code"], "model training")
    same(training.get("dataset_sha256"), selection.get("dataset_sha256"), "selected training dataset")
    dataset = training["dataset"]
    same(dataset.get("protocol_sha256"), plan["protocol_sha256"], "teacher dataset protocol")
    same(content_hash(training["config"]), training["config_sha256"], "training configuration hash")
    expected_config = copy.deepcopy(plan["base_config"])
    expected_config["gcn"]["mode_loss_weight"] = 0.5 if chosen == "a2_on" else 0.0
    same(training["config"], expected_config, "registered selected training configuration")
    teacher_inputs = dataset.get("teacher_inputs")
    require(teacher_inputs, "Checkpoint has no teacher input provenance")
    allowed = {**splits["train"], **splits["val"]}
    seen = set()
    for entry in teacher_inputs:
        iid = entry["iid"]
        require(iid not in seen, f"Duplicate teacher input: {iid}")
        seen.add(iid)
        require(iid in allowed, f"Teacher input is outside frozen train/val: {iid}")
        same(identity(entry), allowed[iid], f"teacher input {iid}")
    for role in ("train", "val"):
        ids = training["sample_splits"].get(role)
        require(ids and len(ids) == len(set(ids)), f"Missing/duplicate retained {role} samples")
        require(set(ids).issubset(splits[role]), f"Checkpoint samples violate frozen {role} role")
        require(set(ids).issubset(seen), f"Checkpoint {role} samples lack teacher provenance")
    require(not training["sample_splits"].get("test"), "Checkpoint used held-out test samples")
    # Selection outputs are frozen before testing. Validate and reproduce the
    # registered rule instead of accepting a freely edited selected-model name.
    outputs = selection.get("validation_outputs") or {}
    same(set(outputs), {"validation_alns.json", "validation_a2_on.json", "validation_a2_off.json"},
         "model-selection validation snapshots")
    validations = {}
    for filename, digest in outputs.items():
        snapshot = root / "validation" / "raw_json" / filename
        same(file_hash(snapshot), digest, f"selection input {filename}")
        rows = read_json(snapshot)
        require(isinstance(rows, list) and rows, f"Empty validation snapshot: {filename}")
        tag = filename[:-5]
        stage = read_json(root / "stages" / f"{tag}.json")
        same(stage.get("state"), "complete", f"selection stage {tag}")
        stage_outputs = {checked_path(k, REPO_ROOT): v for k, v in stage.get("outputs", {}).items()}
        same(stage_outputs.get(snapshot.resolve()), digest, f"selection stage snapshot {tag}")
        manifest_path = root / "validation" / "manifests" / filename
        same(stage_outputs.get(manifest_path.resolve()), file_hash(manifest_path),
             f"selection stage manifest {tag}")
        manifest = read_json(manifest_path)
        same(manifest.get("experiment_id"),
             content_hash({k: v for k, v in manifest.items() if k != "experiment_id"}),
             f"{tag} validation experiment identity")
        check_code(manifest["code"], plan["code"], f"{tag} validation")
        expected_algo = "ALNS" if tag == "validation_alns" else "GCN-ALNS"
        expected_model_sha = None if expected_algo == "ALNS" else file_hash(
            root / "models" / f"{tag.removeprefix('validation_')}.pt")
        if expected_model_sha is not None:
            same(manifest["model"]["sha256"], expected_model_sha, f"{tag} validation checkpoint")
        same(manifest.get("algorithms"), [expected_algo], f"{tag} validation algorithm")
        same(manifest.get("evaluation_split"), "val", f"{tag} validation manifest role")
        same(manifest.get("protocol_sha256"), plan["protocol_sha256"], f"{tag} validation manifest protocol")
        same(content_hash(manifest["config"]), manifest["config_sha256"], f"{tag} validation config identity")
        same({entry["iid"]: identity(entry) for entry in manifest["instances"]},
             splits["val"], f"{tag} unfiltered validation instances")
        index = {}
        repeats = integer(plan["validation"]["repeats"], "validation repeats")
        expected = {(iid, rep) for iid in splits["val"] for rep in range(repeats)}
        for row in rows:
            key = (row["iid"], row["rep"])
            require(key not in index, f"Duplicate validation pair in {tag}: {key}")
            require(key in expected and type(row["rep"]) is int,
                    f"Unexpected validation pair in {tag}: {key}")
            same(row.get("tag"), tag, f"{tag} row tag")
            same(row.get("algorithm"), expected_algo, f"{tag} row algorithm")
            same(row.get("experiment_id"), manifest["experiment_id"], f"{tag} row experiment")
            same(row.get("config_sha256"), manifest["config_sha256"], f"{tag} row config")
            same(row.get("instance_sha256"), splits["val"][row["iid"]]["sha256"], f"{tag} row instance")
            same(row.get("model_sha256"), expected_model_sha, f"{tag} row checkpoint")
            same(row.get("evaluation_split"), "val", f"{tag} split")
            same(row.get("protocol_sha256"), plan["protocol_sha256"], f"{tag} protocol")
            same(row.get("code_sha256"), plan["code"]["sha256"], f"{tag} code")
            require(type(row.get("feasible")) is bool, f"Non-Boolean feasibility in {tag}")
            if row["feasible"]:
                finite_cost(row, tag)
            index[key] = row
        same(set(index), expected, f"{tag} complete validation seed grid")
        validations[tag] = index
    on, off = validations["validation_a2_on"], validations["validation_a2_off"]
    n_on = sum(row["feasible"] for row in on.values())
    n_off = sum(row["feasible"] for row in off.values())
    common = [key for key in on if on[key]["feasible"] and off[key]["feasible"]]
    delta = statistics.mean((on[k]["cost_total"] - off[k]["cost_total"])
                            / max(abs(off[k]["cost_total"]), 1e-12) for k in common) if common else None
    expected_chosen = ("a2_on" if n_on > n_off else "a2_off") if n_on != n_off else (
        ("a2_on" if delta < 0 else "a2_off") if delta is not None and abs(delta) > 1e-9 else "a2_on")
    same(chosen, expected_chosen, "registered validation selection rule")
    same(selection.get("n_validation_runs_per_model"), len(on), "selection validation run count")
    same(selection.get("feasible_on"), n_on, "selection feasible-on count")
    same(selection.get("feasible_off"), n_off, "selection feasible-off count")
    same(selection.get("n_common_feasible"), len(common), "selection common-feasible count")
    recorded_delta = selection.get("paired_relative_cost_on_minus_off")
    require((delta is None and recorded_delta is None) or
            (type(recorded_delta) in (int, float) and delta is not None
             and math.isclose(recorded_delta, delta, rel_tol=1e-12, abs_tol=1e-12)),
            "Mismatch in validation paired-cost selection statistic")
    return selection, training["config"], model, sidecar


def finite_cost(row, context):
    value = row.get("cost_total")
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Feasible row has invalid cost_total in {context}")
    return float(value)


def summarize(rows, iids):
    """First average seeds per instance; then give instances equal weight."""
    grouped = {iid: [] for iid in iids}
    for row in rows:
        grouped[row["iid"]].append(row)
    rates, conditional_costs = [], []
    for iid in sorted(grouped):
        runs = grouped[iid]
        require(runs, f"Missing instance in summary: {iid}")
        feasible = [row for row in runs if row["feasible"]]
        rates.append(len(feasible) / len(runs))
        if feasible:
            conditional_costs.append(statistics.mean(finite_cost(row, iid) for row in feasible))
    return {
        "n_instances": len(grouped), "n_runs": len(rows),
        "n_feasible_runs": sum(row["feasible"] for row in rows),
        "n_instances_with_feasible_run": len(conditional_costs),
        "feasible_rate": statistics.mean(rates),
        "feasible_conditional_cost": statistics.mean(conditional_costs) if conditional_costs else None,
    }


def validate(root):
    plan = read_json(root / "pipeline_plan.json")
    same(plan.get("schema_version"), 1, "pipeline plan schema")
    same(plan["test_main"]["algorithms"], list(ALGORITHMS), "registered fourteen-algorithm panel")
    stages = check_completed_stages(root)
    expected_code = plan["code"]
    same(content_hash(expected_code["files"]), expected_code["sha256"], "plan code content hash")
    current_files = {p.relative_to(REPO_ROOT).as_posix(): file_hash(p)
                     for p in sorted((REPO_ROOT / "src" / "gcnalns").rglob("*.py"))}
    same(current_files, expected_code["files"], "current frozen source tree")
    same(file_hash(REPO_ROOT / "scripts" / "rerun_independent_v1.py"),
         plan["driver_sha256"], "frozen pipeline driver")
    same(file_hash(REPO_ROOT / "tools" / "report_independent_v1.py"),
         plan["report_sha256"], "frozen report generator")
    protocol_paths = {checked_path(command_value(stage["command"], "--protocol"), REPO_ROOT)
                      for stage in stages.values()}
    require(len(protocol_paths) == 1, "Main stages name different protocols")
    protocol_path = protocol_paths.pop()
    splits, instances = load_protocol(protocol_path, plan["protocol_sha256"])
    selection, selected_config, model_path, sidecar_path = check_selection(root, plan, splits)
    selected_sha = selection["model_sha256"]
    ledger_paths = [root / "pipeline_plan.json", root / "model_selection.json", protocol_path,
                    model_path, sidecar_path]
    statistics_by_scale, solution_digests = {}, {}
    for scale in SCALES:
        tag = f"main_{scale}"
        stage = stages[scale]
        cmd = stage["command"]
        snapshot_path = root / "test" / "raw_json" / f"{tag}.json"
        manifest_path = root / "test" / "manifests" / f"{tag}.json"
        paths = (snapshot_path, manifest_path)
        expected_outputs = {p.resolve() for p in paths}
        output_hashes = {checked_path(k, REPO_ROOT): v for k, v in stage.get("outputs", {}).items()}
        same(set(output_hashes), expected_outputs, f"{tag} completed output paths")
        for path in paths:
            same(file_hash(path), output_hashes[path.resolve()], f"{tag} completed output {path.name}")
        manifest, rows = read_json(manifest_path), read_json(snapshot_path)
        require(isinstance(rows, list), f"{tag} snapshot must be a JSON list, not JSONL")
        same(len(rows), plan["expected_test_records"]["main"][scale], f"{tag} planned record count")
        same(manifest.get("schema_version"), 1, f"{tag} manifest schema")
        same(manifest.get("experiment_id"), content_hash({k: v for k, v in manifest.items() if k != "experiment_id"}),
             f"{tag} experiment content identity")
        check_code(manifest["code"], expected_code, tag)
        for field, expected in (("tag", tag), ("evaluation_split", "test"),
                                ("protocol_sha256", plan["protocol_sha256"]),
                                ("algorithms", list(ALGORITHMS)), ("instance_scales", [scale]),
                                ("ablation", {}), ("algo_label_suffix", "")):
            same(manifest.get(field), expected, f"{tag} {field}")
        repeats = integer(plan["test_main"][scale]["repeats"], f"{scale} repeats")
        max_iter = integer(plan["test_main"][scale]["max_iter"], f"{scale} max_iter")
        same(manifest.get("repeats"), repeats, f"{tag} repeats")
        config = copy.deepcopy(selected_config)
        config["alns"]["max_iter"] = max_iter
        same(manifest["config"], config, f"{tag} registered configuration")
        same(manifest["config_sha256"], content_hash(config), f"{tag} configuration hash")
        same(manifest["model"]["sha256"], selected_sha, f"{tag} selected model fingerprint")
        same(manifest["model"]["provenance_sha256"], file_hash(sidecar_path), f"{tag} model sidecar")
        same(checked_path(manifest["model"]["path"], REPO_ROOT), model_path, f"{tag} model path")
        for flag, value in (("--tag", tag), ("--split", "test"), ("--repeats", str(repeats)),
                            ("--max-iter", str(max_iter))):
            same(command_value(cmd, flag), value, f"{tag} stage {flag}")
        same(command_values(cmd, "--algorithms"), list(ALGORITHMS), f"{tag} stage algorithms")
        same(command_values(cmd, "--instance-scales"), [scale], f"{tag} stage scale")
        same(checked_path(command_value(cmd, "--out"), REPO_ROOT), root / "test", f"{tag} stage output root")
        same(checked_path(command_value(cmd, "--model-path"), REPO_ROOT), model_path, f"{tag} stage model")
        config_path = checked_path(command_value(cmd, "--config"), REPO_ROOT)
        same(config_path, root / "configs" / f"{selection['selected']}.yaml", f"{tag} selected config path")
        stage_inputs = {str(config_path): file_hash(config_path), str(model_path): file_hash(model_path),
                        str(sidecar_path): file_hash(sidecar_path)}
        signature = content_hash({"command": cmd, "inputs": stage_inputs,
                                  "code_sha256": expected_code["sha256"],
                                  "driver_sha256": plan["driver_sha256"]})
        same(stage.get("signature"), signature, f"{tag} registered stage signature")
        scale_instances = {iid: entry for iid, entry in splits["test"].items()
                           if instances[iid]["meta"]["scale"] == scale}
        require(scale_instances, f"No held-out protocol instances for {scale}")
        manifest_instances = {}
        for entry in manifest["instances"]:
            iid = entry["iid"]
            require(iid not in manifest_instances, f"Duplicate manifest instance: {iid}")
            manifest_instances[iid] = identity(entry)
        same(manifest_instances, scale_instances, f"{tag} full unfiltered test scope")
        expected_keys = {(tag, algo, iid, rep) for algo in ALGORITHMS for iid in scale_instances
                         for rep in range(repeats if algo in STOCHASTIC else 1)}
        same(len(expected_keys), plan["expected_test_records"]["main"][scale], f"{tag} plan/seed-grid count")
        seen, by_algorithm = set(), {algo: [] for algo in ALGORITHMS}
        for row in rows:
            key = (row.get("tag"), row.get("algorithm"), row.get("iid"), row.get("rep"))
            require(type(row.get("rep")) is int, f"Non-integer repetition key in {tag}")
            require(key in expected_keys, f"Unexpected main-result key in {tag}: {key}")
            require(key not in seen, f"Duplicate main-result key in {tag}: {key}")
            seen.add(key)
            iid, algo = row["iid"], row["algorithm"]
            for field, expected in (("scale", scale), ("experiment_id", manifest["experiment_id"]),
                                    ("evaluation_split", "test"), ("protocol_sha256", plan["protocol_sha256"]),
                                    ("config_sha256", manifest["config_sha256"]),
                                    ("code_sha256", expected_code["sha256"]),
                                    ("instance_sha256", scale_instances[iid]["sha256"]),
                                    ("model_sha256", selected_sha if algo == "GCN-ALNS" else None)):
                same(row.get(field), expected, f"{key} {field}")
            require(type(row.get("feasible")) is bool, f"Non-Boolean feasibility for {key}")
            if row["feasible"]:
                finite_cost(row, str(key))
            solution_path = checked_path(row["solution_path"], root / "test")
            solution_dir = (root / "test" / "solutions" / tag).resolve()
            require(solution_dir in solution_path.parents, f"Solution path outside {tag} directory")
            digest = file_hash(solution_path)
            same(digest, row.get("solution_sha256"), f"saved deployment {key}")
            require(str(solution_path) not in solution_digests, f"Reused saved-deployment path: {key}")
            solution_digests[str(solution_path)] = digest
            by_algorithm[algo].append(row)
        same(seen, expected_keys, f"{tag} complete paired instance/repetition grid")
        statistics_by_scale[scale] = {algo: summarize(by_algorithm[algo], scale_instances) for algo in ALGORITHMS}
        ledger_paths.extend((root / "stages" / f"{tag}.json", snapshot_path, manifest_path, config_path))
    ledger = {str(path): file_hash(path) for path in ledger_paths}
    return {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "exporter_sha256": file_hash(Path(__file__).resolve()),
        "results_root": str(root), "protocol_sha256": plan["protocol_sha256"],
        "code_sha256": expected_code["sha256"], "selected_variant": selection["selected"],
        "selected_model_sha256": selected_sha, "input_sha256": ledger,
        "validated_solution_count": len(solution_digests),
        "solution_hash_index_sha256": content_hash(solution_digests),
        "aggregation": {
            "feasible_rate": "Mean of per-instance feasible-seed fractions; all planned instances included.",
            "feasible_conditional_cost": "Mean of per-instance mean cost over feasible seeds, restricted to instances with at least one feasible seed.",
            "cost_interpretation": "Descriptive, algorithm-specific feasible subsets; not a matched-feasible cost effect or a significance test.",
        },
        "statistics": statistics_by_scale,
    }


def latex_escape(text):
    return "".join({"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "\\": r"\textbackslash{}"}.get(char, char) for char in text)


def render_table(export):
    lines = [
        "% Generated by tools/update_results.py after complete three-scale validation.",
        "% Protocol SHA256: " + export["protocol_sha256"],
        "% Code SHA256: " + export["code_sha256"],
        r"\begin{table*}[!t]",
        r"\caption{Main comparison on all held-out instances. Feasibility is the mean of per-instance feasible-seed fractions (\%). Conditional cost first averages feasible seeds within each instance, then averages instances with at least one feasible run; ``--'' means no feasible run. Costs describe algorithm-specific feasible subsets and are not a paired cost comparison or evidence of statistical significance.}",
        r"\label{tab:main_results}", r"\centering", r"\small", r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{Small} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Large} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Algorithm & Feasible (\%) & Cond. cost & Feasible (\%) & Cond. cost & Feasible (\%) & Cond. cost \\",
        r"\midrule",
    ]
    for algo in ALGORITHMS:
        cells = [latex_escape(algo)]
        for scale in SCALES:
            stats = export["statistics"][scale][algo]
            cost = stats["feasible_conditional_cost"]
            cells.extend((f"{100 * stats['feasible_rate']:.2f}", "--" if cost is None else f"{cost:.2f}"))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines)


def publish(export):
    # Recheck the export inputs immediately before publishing. Completed stage
    # hashes already bind snapshots/manifests, and the manifest commits the table.
    for path, digest in export["input_sha256"].items():
        same(file_hash(path), digest, f"input changed during export: {path}")
    table = render_table(export).encode("utf-8")
    export["table_sha256"] = hashlib.sha256(table).hexdigest()
    export["table_file"] = "main_results.tex"
    manifest = (json.dumps(export, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    generated = PAPER_ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        for name, data in (("main_results.tex", table), ("result_export_manifest.json", manifest)):
            with tempfile.NamedTemporaryFile(dir=generated, prefix=name + ".", suffix=".tmp", delete=False) as stream:
                stream.write(data)
                temporary.append((Path(stream.name), generated / name))
        # Publish the manifest last: it is the completion marker for the table.
        for source, destination in temporary:
            os.replace(source, destination)
    finally:
        for source, _ in temporary:
            source.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/independent_v1"))
    parser.add_argument("--check-only", action="store_true", help="Validate completed inputs without writing paper files")
    args = parser.parse_args(argv)
    try:
        export = validate(args.results_root.resolve())
        if not args.check_only:
            publish(export)
        counts = ", ".join(f"{scale}={sum(s['n_runs'] for s in export['statistics'][scale].values())}" for scale in SCALES)
        print(f"Validated all fourteen algorithms and paired grids: {counts}.")
        print("Check only: no files written." if args.check_only else
              f"Exported main_results.tex and result_export_manifest.json to {PAPER_ROOT / 'generated'}.")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
