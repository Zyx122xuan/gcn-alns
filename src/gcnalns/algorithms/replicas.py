"""Helpers for active-replica placement under fault-domain isolation."""
from __future__ import annotations

from ..config import Config
from ..core.model import Instance, Solution


def footprint(instance: Instance, sol: Solution) -> tuple[dict[int, float], dict[int, float]]:
    used_cpu = {k: 0.0 for k in instance.continuum.nodes}
    used_mem = {k: 0.0 for k in instance.continuum.nodes}
    for i, repl in sol.placement.items():
        for k in repl:
            used_cpu[k] += instance.workflow.tasks[i].resource
            used_mem[k] += instance.workflow.tasks[i].mem
    return used_cpu, used_mem


def add_active_replicas(instance: Instance, sol: Solution, tasks: set[int],
                        m: int, cfg: Config) -> None:
    """Promote ``tasks`` to active backup with up to ``m`` total replicas.

    Extra replicas are placed on nodes in *distinct* fault domains, preferring
    reliable, high-capacity nodes, respecting remaining capacity. Tasks that
    cannot satisfy fault-domain isolation stay passive.
    """
    used_cpu, used_mem = footprint(instance, sol)
    nodes = instance.continuum.nodes

    for i in tasks:
        task = instance.workflow.tasks[i]
        primary = sol.placement.get(i, [])
        if not primary:
            continue
        domains = {nodes[k].fault_domain for k in primary}
        chosen = list(primary)
        while len(chosen) < m:
            # least-loaded reliable node in a distinct fault domain (spread replicas)
            cands = [k for k in nodes
                     if k not in chosen
                     and nodes[k].fault_domain not in domains
                     and used_cpu[k] + task.resource <= nodes[k].res_cpu + 1e-9
                     and used_mem[k] + task.mem <= nodes[k].res_mem + 1e-9]
            if not cands:
                break
            # prefer fast yet uncongested nodes so the backup is a real
            # alternative (close finish time to the primary), tie-break low failure
            k = max(cands, key=lambda k: (nodes[k].compute / (1.0 + used_cpu[k]),
                                          -nodes[k].failure_rate))
            chosen.append(k)
            domains.add(nodes[k].fault_domain)
            used_cpu[k] += task.resource
            used_mem[k] += task.mem
        if len(chosen) >= 2:
            sol.placement[i] = chosen
            sol.mode[i] = True
        else:
            sol.mode[i] = False  # could not isolate -> remain passive
