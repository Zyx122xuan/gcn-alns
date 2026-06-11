"""Core data model for RLC-HFTSP.

Defines the immutable problem inputs (Task, Node, Link, Workflow, Continuum,
Instance) and the mutable scheduling solution (Solution). All symbols follow
the modeling document:

    Workflow  G_A = (T, A)
    Continuum G_P = (N, L)
    Instance  I  = (G_A, G_P, D, Delta_SLA)
    Solution  chi = (X, Y, m, O, Pi)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Problem inputs
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A DAG task t_i (Modeling doc sec.2)."""

    tid: int
    workload: float            # w_i, computation workload (MI)
    resource: float = 1.0      # sigma_i, resource demand (cores)
    mem: float = 1.0           # memory demand (GB)
    # Structural attributes (filled by dag analysis, independent of placement)
    criticality: float = 0.0   # h_i in [0, 1]
    ranku: float = 0.0         # upward rank
    slack: float = 0.0         # slack time
    est: float = 0.0           # earliest start time
    eft: float = 0.0           # earliest finish time
    lft: float = 0.0           # latest finish time
    indeg: int = 0
    outdeg: int = 0
    level: int = 0             # topological level


@dataclass
class Node:
    """A physical computing-continuum node n_k (Modeling doc sec.1)."""

    nid: int
    compute: float             # c_k / f_k, compute capability (MI/s)
    cpu_cap: float             # R_k^cap (cores)
    mem_cap: float             # memory capacity (GB)
    load: float                # u_k current utilization [0, 1]
    load_var: float            # load variance / volatility
    failure_rate: float        # lambda_k (s^-1)
    cost_coef: float           # rho_k unit compute cost ($/s)
    layer: str                 # 'cloud' | 'edge' | 'device'
    fault_domain: str          # fd_k
    queue_wait: float = 0.0    # estimated queueing delay (s)
    degree: int = 0            # topological degree in physical network

    @property
    def res_cpu(self) -> float:
        """Remaining CPU capacity = cap * (1 - load)."""
        return self.cpu_cap * (1.0 - self.load)

    @property
    def res_mem(self) -> float:
        return self.mem_cap * (1.0 - self.load)


@dataclass
class Link:
    """A physical link l_kl between two nodes (Modeling doc sec.1)."""

    src: int
    dst: int
    bandwidth: float           # B_kl (MB/s)
    latency: float             # L_kl (s)
    failure_rate: float        # mu_kl (s^-1)
    cost_coef: float = 0.0     # rho_kl unit data transfer cost ($/MB)


@dataclass
class Workflow:
    """DAG workflow G_A = (T, A)."""

    name: str
    tasks: dict[int, Task]
    edges: list[tuple[int, int]]                 # (i, j) dependency edges
    data: dict[tuple[int, int], float]           # d_ij communication volume (MB)

    def successors(self, tid: int) -> list[int]:
        return [j for (i, j) in self.edges if i == tid]

    def predecessors(self, tid: int) -> list[int]:
        return [i for (i, j) in self.edges if j == tid]

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


@dataclass
class Continuum:
    """Physical network G_P = (N, L)."""

    nodes: dict[int, Node]
    links: dict[tuple[int, int], Link]           # keyed by (k, l)

    def link(self, k: int, l: int) -> Optional[Link]:
        if k == l:
            return None
        return self.links.get((k, l)) or self.links.get((l, k))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


@dataclass
class Instance:
    """A complete experiment instance I = (G_A, G_P, D, Delta_SLA).

    ``meta`` carries reproducibility / scenario fields (workflow name, scale,
    load level, SLA scenario, seeds), per Spec sec.20-21.
    """

    iid: str
    workflow: Workflow
    continuum: Continuum
    deadline: float                              # D / D_g
    recovery_window: float                       # Delta_SLA (R^max)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------


@dataclass
class Solution:
    """A scheduling solution chi = (X, Y, m, O, Pi).

    placement: task -> list of node ids hosting its instances (replicas).
               len == 1 for passive recompute, == m_i for active backup.
    mode:      task -> bool, True = active backup (y_i = 1), False = passive.
    order:     node -> ordered list of task ids (execution order O on that node).
               If None, derived from DAG topological order during evaluation.
    """

    placement: dict[int, list[int]] = field(default_factory=dict)
    mode: dict[int, bool] = field(default_factory=dict)
    order: dict[int, list[int]] = field(default_factory=dict)
    # Cached evaluation outputs (populated by schedule_eval / recovery)
    meta: dict = field(default_factory=dict)

    def replicas(self, tid: int) -> list[int]:
        return self.placement.get(tid, [])

    def num_replicas(self, tid: int) -> int:
        return len(self.placement.get(tid, []))

    def is_active(self, tid: int) -> bool:
        return bool(self.mode.get(tid, False))

    def clone(self) -> "Solution":
        return Solution(
            placement={t: list(v) for t, v in self.placement.items()},
            mode=dict(self.mode),
            order={n: list(v) for n, v in self.order.items()},
            meta=dict(self.meta),
        )
