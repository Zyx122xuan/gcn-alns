"""Controllable single-point fault set Omega(S) (Modeling doc sec.3, Spec sec.10).

    Omega(S) = Omega^task(S) U Omega^node(S) U Omega^link(S)

Only objects actually occupied by the solution are included:
* task-instance faults: one per deployed (task, node) replica
* node faults: one per node hosting >= 1 instance
* link faults: one per cross-node dependency transfer under the *effective*
  placement (eff_node of producer != eff_node of consumer)
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import Instance, Solution
from .schedule_eval import ScheduleResult


@dataclass(frozen=True)
class Fault:
    kind: str                 # 'task' | 'node' | 'link' | 'domain'
    i: int = -1               # task id (task/link producer)
    j: int = -1               # consumer task (link)
    k: int = -1               # node id (task/node/link producer node)
    l: int = -1               # consumer node (link)
    domain: str = ""          # fault-domain label (kind == 'domain': whole domain fails)

    def label(self) -> str:
        if self.kind == "task":
            return f"task(t{self.i}@n{self.k})"
        if self.kind == "node":
            return f"node(n{self.k})"
        if self.kind == "domain":
            return f"domain({self.domain})"
        return f"link(t{self.i}->t{self.j}, n{self.k}->n{self.l})"


def build_fault_set(instance: Instance, solution: Solution,
                    sched: ScheduleResult) -> list[Fault]:
    faults: list[Fault] = []

    # task-instance faults (one per replica)
    for i, repl in solution.placement.items():
        for k in repl:
            faults.append(Fault(kind="task", i=i, k=k))

    # node faults (nodes hosting >= 1 instance)
    hosting: set[int] = set()
    for repl in solution.placement.values():
        hosting.update(repl)
    for k in sorted(hosting):
        faults.append(Fault(kind="node", k=k))

    # link faults (cross-node effective dependency transfers)
    for (i, j) in instance.workflow.edges:
        ki = sched.eff_node.get(i)
        kj = sched.eff_node.get(j)
        if ki is None or kj is None or ki == kj:
            continue
        faults.append(Fault(kind="link", i=i, j=j, k=ki, l=kj))

    return faults


def build_domain_faults(instance: Instance, solution: Solution) -> list[Fault]:
    """Domain-level (correlated) faults: one per fault domain hosting >= 1 instance.

    A domain fault fails ALL nodes in that domain simultaneously (shared root
    cause: rack/power/switch/AZ). NOT part of the default single-point Omega
    (build_fault_set) -- used only by the fault-domain-isolation robustness study
    so existing single-point verification stays byte-identical.
    """
    hosting_domains: set[str] = set()
    for repl in solution.placement.values():
        for k in repl:
            hosting_domains.add(instance.continuum.nodes[k].fault_domain)
    return [Fault(kind="domain", domain=d) for d in sorted(hosting_domains)]
