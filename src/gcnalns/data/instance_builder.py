"""Build complete experiment instances and (de)serialize them as JSON.

Combines a workflow source (synthetic or parsed Pegasus) with a resource
snapshot (synthetic or parsed Alibaba trace), applies layering + links, fills
structural task attributes, and assigns SLA parameters.

CLI::

    python -m gcnalns.data.instance_builder --out data/instances --n 20
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from ..config import Config, load_config
from ..core.dag import compute_structural
from ..core.model import Continuum, Instance, Node, Task, Workflow
from ..utils.io import load_json, save_json
from . import synthetic
from .layering import build_nodes
from .links import build_continuum, get_factory
from .sla import generate_sla


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_workflow(raw: dict, cfg: Config) -> Workflow:
    """Convert a raw workflow dict (jobs/edges/files) into a Workflow.

    w_i = runtime_i * f_std  (Spec sec.2).
    """
    f_std = float(cfg["f_std"])
    tasks: dict[int, Task] = {}
    for jid, job in raw["jobs"].items():
        tid = int(jid)
        tasks[tid] = Task(tid=tid, workload=float(job["runtime"]) * f_std,
                          resource=float(job.get("cpu", 1.0)),
                          mem=float(job.get("mem", 1.0)))
    edges = [(int(i), int(j)) for (i, j) in raw["edges"]]
    data = {(int(i), int(j)): float(v) for (i, j), v in raw["files"].items()}
    return Workflow(name=raw["name"], tasks=tasks, edges=edges, data=data)


def build_instance(raw_workflow: dict, machines: list[dict], cfg: Config,
                   iid: str, scenario: str = "medium",
                   meta: dict | None = None) -> Instance:
    wf = build_workflow(raw_workflow, cfg)
    nodes = build_nodes(machines, cfg)
    materialize = len(nodes) <= 60
    continuum, factory = build_continuum(nodes, cfg, materialize=materialize)
    inst = Instance(iid=iid, workflow=wf, continuum=continuum,
                    deadline=0.0, recovery_window=0.0, meta=meta or {})
    compute_structural(wf, continuum, cfg, factory)
    t_ref, deadline, rwin = generate_sla(inst, cfg, scenario)
    inst.deadline = deadline
    inst.recovery_window = rwin
    inst.meta.update({"t_ref": t_ref, "sla_scenario": scenario,
                      "workflow_type": wf.name})
    return inst


# ---------------------------------------------------------------------------
# Serialization (Spec sec.20)
# ---------------------------------------------------------------------------
def instance_to_dict(inst: Instance, materialize_links: bool = False,
                     cfg: Config | None = None) -> dict:
    wf = inst.workflow
    d = {
        "iid": inst.iid,
        "workflow_name": wf.name,
        "tasks": {str(t): {"workload": tk.workload, "resource": tk.resource, "mem": tk.mem}
                  for t, tk in wf.tasks.items()},
        "edges": [[i, j] for (i, j) in wf.edges],
        "data": {f"{i},{j}": v for (i, j), v in wf.data.items()},
        "nodes": {str(k): {
            "compute": n.compute, "cpu_cap": n.cpu_cap, "mem_cap": n.mem_cap,
            "load": n.load, "load_var": n.load_var, "failure_rate": n.failure_rate,
            "cost_coef": n.cost_coef, "layer": n.layer, "fault_domain": n.fault_domain,
        } for k, n in inst.continuum.nodes.items()},
        "deadline": inst.deadline,
        "recovery_window": inst.recovery_window,
        "meta": inst.meta,
    }
    if materialize_links and cfg is not None:
        factory = get_factory(inst, cfg)
        links = {}
        ids = list(inst.continuum.nodes.keys())
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                link = factory.get(ids[a], ids[b])
                if link:
                    links[f"{ids[a]},{ids[b]}"] = {
                        "bandwidth": link.bandwidth, "latency": link.latency,
                        "failure_rate": link.failure_rate}
        d["links"] = links
    return d


def instance_from_dict(d: dict) -> Instance:
    tasks = {int(t): Task(tid=int(t), workload=v["workload"],
                          resource=v.get("resource", 1.0), mem=v.get("mem", 1.0))
             for t, v in d["tasks"].items()}
    edges = [(int(i), int(j)) for i, j in d["edges"]]
    data = {}
    for key, v in d["data"].items():
        i, j = key.split(",")
        data[(int(i), int(j))] = float(v)
    wf = Workflow(name=d["workflow_name"], tasks=tasks, edges=edges, data=data)
    nodes = {}
    for k, n in d["nodes"].items():
        nodes[int(k)] = Node(
            nid=int(k), compute=n["compute"], cpu_cap=n["cpu_cap"], mem_cap=n["mem_cap"],
            load=n["load"], load_var=n["load_var"], failure_rate=n["failure_rate"],
            cost_coef=n["cost_coef"], layer=n["layer"], fault_domain=n["fault_domain"])
    continuum = Continuum(nodes=nodes, links={})
    return Instance(iid=d["iid"], workflow=wf, continuum=continuum,
                    deadline=d["deadline"], recovery_window=d["recovery_window"],
                    meta=d.get("meta", {}))


def load_instance(path: str | Path, cfg: Config | None = None) -> Instance:
    inst = instance_from_dict(load_json(path))
    cfg = cfg or load_config()
    factory = get_factory(inst, cfg)
    compute_structural(inst.workflow, inst.continuum, cfg, factory)
    return inst


# ---------------------------------------------------------------------------
# CLI: generate a batch of synthetic instances
# ---------------------------------------------------------------------------
def generate_instances(cfg: Config, out: str | Path, *, scale: str = "small",
                       n: int = 20, load: str = "medium", seed: int | None = None,
                       scenarios: list[str] | None = None, iid_prefix: str = "",
                       overwrite: bool = True) -> list[Path]:
    """Generate a batch; protocol preparation uses a separate seed/ID namespace."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base_seed = seed if seed is not None else cfg.get("seeds.global", 2026)
    n_tasks = cfg["scales"]["workflow_tasks"][scale]
    n_nodes = cfg["scales"]["resource_nodes"][scale]
    load_level = cfg["scales"]["load_levels"][load]
    wf_types = cfg["workflow_types"]
    scenarios = list(scenarios) if scenarios else ["tight", "medium", "loose"]
    unknown = [s for s in scenarios if s not in cfg["sla"]["scenarios"]]
    if unknown:
        raise ValueError(f"unknown SLA scenarios: {unknown}")
    paths = []
    per_combo = max(1, n // (len(wf_types) * len(scenarios)) or 1)
    for wf_type, scenario in itertools.product(wf_types, scenarios):
        for rep in range(per_combo):
            instance_seed = base_seed + len(paths)
            raw_wf = synthetic.generate_raw_workflow(wf_type, n_tasks, instance_seed)
            machines = synthetic.generate_raw_machines(n_nodes, load_level,
                                                        instance_seed + 10_000)
            iid = f"{wf_type}_{scale}_{load}_{scenario}_r{rep}"
            if iid_prefix:
                iid = f"{iid_prefix}_{iid}"
            path = out / f"{iid}.json"
            if path.exists() and not overwrite:
                raise ValueError(f"Refusing to overwrite frozen instance: {path}")
            inst = build_instance(raw_wf, machines, cfg, iid, scenario,
                                  meta={"scale": scale, "load": load,
                                        "seed": instance_seed})
            save_json(instance_to_dict(inst, materialize_links=(n_nodes <= 60), cfg=cfg),
                      path)
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment instances")
    parser.add_argument("--out", default="data/instances")
    parser.add_argument("--n", type=int, default=20, help="instances per combo cap")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scale", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--load", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="SLA scenarios to build; default: tight medium loose "
                             "(extra tiers like tight2/tight3 must be requested "
                             "explicitly and go into a dedicated instance set)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = generate_instances(cfg, args.out, scale=args.scale, n=args.n,
                               load=args.load, seed=args.seed, scenarios=args.scenarios)
    print(f"generated {len(paths)} instances into {args.out}")


if __name__ == "__main__":
    main()
