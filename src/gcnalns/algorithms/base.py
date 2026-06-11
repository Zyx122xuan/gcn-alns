"""Scheduler base class. Every algorithm produces a :class:`Solution`.

All schedulers share the same downstream recovery verifier; they only differ
in how they build the placement / fault-tolerance mode / replica positions.
"""
from __future__ import annotations

import abc

from ..config import Config
from ..core.model import Instance, Solution


class Scheduler(abc.ABC):
    name: str = "base"

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.runtime: float = 0.0          # scheduling wall-clock (s)
        self.convergence_iter: int = 0     # for iterative methods
        self.prune_ratio: float = 0.0      # fraction of pruned candidate actions

    @abc.abstractmethod
    def schedule(self, instance: Instance) -> Solution:
        """Return a scheduling solution for ``instance``."""
        raise NotImplementedError


def topo_node_order(instance: Instance, solution: Solution) -> dict[int, list[int]]:
    """Derive each node's execution order O as the topological order restricted
    to tasks placed on that node (a valid linear extension of the DAG)."""
    from ..core.dag import topological_order

    order = topological_order(instance.workflow)
    node_order: dict[int, list[int]] = {k: [] for k in instance.continuum.nodes}
    for i in order:
        for k in solution.replicas(i):
            node_order[k].append(i)
    return node_order
