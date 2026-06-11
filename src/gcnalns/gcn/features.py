"""Feature extraction for the GCN scoring model (Spec sec.16).

Three feature groups, all derived from an :class:`Instance` and its
:class:`LinkFactory` (deployment-independent, so a single feature tensor can be
reused to score every candidate task->node action):

* **task features**   f_i : structural DAG attributes (workload, resource, mem,
  criticality, ranku, slack, est/eft/lft, in/out degree, level).
* **node features**   g_k : physical-node attributes (compute, capacity, load,
  volatility, failure rate, cost, queue) + layer one-hot, plus a separate
  fault-domain index fed to an embedding table in the model.
* **interaction**     z_ik: task-node coupling (execution time t_ik, failure
  probability P_fail, passive recovery risk, deployment cost, communication
  affinity, relative compute, and resource/deadline constraint-satisfaction
  bits).

The physical graph used by the GCN connects every node to its ``knn_k`` nearest
neighbours by per-MB communication delay (or to all nodes when the pool is
small); it is deterministic given the instance + config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..core.model import Instance
from ..data.links import LinkFactory

LAYERS = ("cloud", "edge", "device")

TASK_FEATURE_NAMES = (
    "workload", "resource", "mem", "criticality", "ranku", "slack",
    "est", "eft", "lft", "indeg", "outdeg", "level",
)
NODE_FEATURE_NAMES = (
    "compute", "cpu_cap", "mem_cap", "load", "load_var", "failure_rate",
    "cost_coef", "queue_wait", "res_cpu", "is_cloud", "is_edge", "is_device",
)
INTERACTION_FEATURE_NAMES = (
    "t_ik", "p_fail", "rec_risk", "cost", "comm", "compute_ratio",
    "cpu_fit", "mem_fit", "deadline_fit",
)

TASK_DIM = len(TASK_FEATURE_NAMES)
NODE_DIM = len(NODE_FEATURE_NAMES)
INTER_DIM = len(INTERACTION_FEATURE_NAMES)


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=float)
    return (a - lo) / (hi - lo)


def domain_index_map(cfg: Config) -> dict[str, int]:
    """Stable global index for each fault domain string ``"<layer>-<j>"``.

    Indices are consistent across instances so the embedding table is shared.
    """
    groups = cfg["fault_domain"]["groups"]
    mapping: dict[str, int] = {}
    offset = 0
    for layer in LAYERS:
        g = int(groups[layer])
        for j in range(g):
            mapping[f"{layer}-{j}"] = offset + j
        offset += g
    return mapping


def num_domains(cfg: Config) -> int:
    groups = cfg["fault_domain"]["groups"]
    return int(sum(int(groups[l]) for l in LAYERS))


@dataclass
class InstanceFeatures:
    iid: str
    task_ids: list[int]                 # row order of task tensor
    node_ids: list[int]                 # row order of node tensor
    task_feat: np.ndarray               # [T, TASK_DIM]
    node_feat: np.ndarray               # [N, NODE_DIM]
    node_domain: np.ndarray             # [N] long, global domain index
    edge_index: np.ndarray              # [2, E] physical graph (undirected dup)
    inter: np.ndarray                   # [T, N, INTER_DIM]
    task_pos: dict[int, int]            # tid -> row
    node_pos: dict[int, int]            # nid -> row

    @property
    def num_tasks(self) -> int:
        return len(self.task_ids)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


def _node_link_stats(instance: Instance, factory: LinkFactory,
                     node_ids: list[int], cfg: Config,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-node mean latency and mean inverse-bandwidth over sampled neighbours."""
    n = len(node_ids)
    sample = int(cfg.get("gcn.comm_neighbor_sample", 64))
    rng = np.random.default_rng(int(cfg.get("seeds.global", 2026)))
    mean_lat = np.zeros(n, dtype=float)
    mean_inv_bw = np.zeros(n, dtype=float)
    for r, k in enumerate(node_ids):
        if n <= 1:
            continue
        others = [l for l in node_ids if l != k]
        if len(others) > sample:
            pick = rng.choice(len(others), size=sample, replace=False)
            others = [others[p] for p in pick]
        lats, inv_bws = [], []
        for l in others:
            link = factory.get(int(k), int(l))
            if link is None:
                continue
            lats.append(link.latency)
            inv_bws.append(1.0 / max(link.bandwidth, 1e-9))
        mean_lat[r] = float(np.mean(lats)) if lats else 0.0
        mean_inv_bw[r] = float(np.mean(inv_bws)) if inv_bws else 0.0
    return mean_lat, mean_inv_bw


def _knn_edges(instance: Instance, factory: LinkFactory, node_ids: list[int],
               cfg: Config) -> np.ndarray:
    """Undirected kNN physical graph by unit-data comm delay (lower = nearer)."""
    n = len(node_ids)
    k = int(cfg.get("gcn.knn_k", 8))
    edges: set[tuple[int, int]] = set()
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    if n <= k + 1:
        for a in range(n):
            for b in range(a + 1, n):
                edges.add((a, b))
    else:
        for a in range(n):
            ka = node_ids[a]
            delays = []
            for b in range(n):
                if b == a:
                    continue
                delays.append((factory.comm_delay(int(ka), int(node_ids[b]), 1.0), b))
            delays.sort(key=lambda x: x[0])
            for _, b in delays[:k]:
                edges.add((min(a, b), max(a, b)))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    src, dst = [], []
    for a, b in edges:
        src += [a, b]
        dst += [b, a]
    return np.array([src, dst], dtype=np.int64)


def build_features(instance: Instance, cfg: Config,
                   factory: LinkFactory) -> InstanceFeatures:
    """Build task / node / interaction feature tensors for ``instance``."""
    wf = instance.workflow
    nodes = instance.continuum.nodes
    task_ids = list(wf.tasks.keys())
    node_ids = list(nodes.keys())
    T, N = len(task_ids), len(node_ids)
    task_pos = {t: r for r, t in enumerate(task_ids)}
    node_pos = {k: r for r, k in enumerate(node_ids)}

    # ---- task features ----------------------------------------------------
    workload = np.array([wf.tasks[t].workload for t in task_ids], dtype=float)
    resource = np.array([wf.tasks[t].resource for t in task_ids], dtype=float)
    mem = np.array([wf.tasks[t].mem for t in task_ids], dtype=float)
    crit = np.array([wf.tasks[t].criticality for t in task_ids], dtype=float)
    ranku = np.array([wf.tasks[t].ranku for t in task_ids], dtype=float)
    slack = np.array([wf.tasks[t].slack for t in task_ids], dtype=float)
    est = np.array([wf.tasks[t].est for t in task_ids], dtype=float)
    eft = np.array([wf.tasks[t].eft for t in task_ids], dtype=float)
    lft = np.array([wf.tasks[t].lft for t in task_ids], dtype=float)
    indeg = np.array([wf.tasks[t].indeg for t in task_ids], dtype=float)
    outdeg = np.array([wf.tasks[t].outdeg for t in task_ids], dtype=float)
    level = np.array([wf.tasks[t].level for t in task_ids], dtype=float)

    task_feat = np.stack([
        _minmax(workload), _minmax(resource), _minmax(mem), crit,
        _minmax(ranku), _minmax(slack), _minmax(est), _minmax(eft),
        _minmax(lft), _minmax(indeg), _minmax(outdeg), _minmax(level),
    ], axis=1)

    # ---- node features ----------------------------------------------------
    compute = np.array([nodes[k].compute for k in node_ids], dtype=float)
    cpu_cap = np.array([nodes[k].cpu_cap for k in node_ids], dtype=float)
    mem_cap = np.array([nodes[k].mem_cap for k in node_ids], dtype=float)
    load = np.array([nodes[k].load for k in node_ids], dtype=float)
    load_var = np.array([nodes[k].load_var for k in node_ids], dtype=float)
    frate = np.array([nodes[k].failure_rate for k in node_ids], dtype=float)
    cost_coef = np.array([nodes[k].cost_coef for k in node_ids], dtype=float)
    queue = np.array([nodes[k].queue_wait for k in node_ids], dtype=float)
    res_cpu = np.array([nodes[k].res_cpu for k in node_ids], dtype=float)
    is_cloud = np.array([1.0 if nodes[k].layer == "cloud" else 0.0 for k in node_ids])
    is_edge = np.array([1.0 if nodes[k].layer == "edge" else 0.0 for k in node_ids])
    is_device = np.array([1.0 if nodes[k].layer == "device" else 0.0 for k in node_ids])

    log_frate = np.log10(np.clip(frate, 1e-12, None))
    node_feat = np.stack([
        _minmax(compute), _minmax(cpu_cap), _minmax(mem_cap), load, load_var,
        _minmax(log_frate), _minmax(cost_coef), _minmax(queue), _minmax(res_cpu),
        is_cloud, is_edge, is_device,
    ], axis=1)

    dmap = domain_index_map(cfg)
    node_domain = np.array(
        [dmap.get(nodes[k].fault_domain, 0) for k in node_ids], dtype=np.int64)

    # ---- interaction features z_ik ---------------------------------------
    rcv = cfg["recovery"]
    det = float(rcv["delta_detect"])
    exec_mat = workload[:, None] / np.clip(compute[None, :], 1e-9, None)   # [T,N]
    pfail = 1.0 - np.exp(-frate[None, :] * exec_mat)
    recrisk = np.clip(det + queue[None, :] + exec_mat - slack[:, None], 0.0, None)
    cost_mat = cost_coef[None, :] * exec_mat

    mean_lat, mean_inv_bw = _node_link_stats(instance, factory, node_ids, cfg)
    in_data = np.zeros(T, dtype=float)
    n_edges_i = np.zeros(T, dtype=float)
    for (i, j), v in wf.data.items():
        if i in task_pos:
            in_data[task_pos[i]] += v
            n_edges_i[task_pos[i]] += 1.0
        if j in task_pos:
            in_data[task_pos[j]] += v
            n_edges_i[task_pos[j]] += 1.0
    comm_mat = (in_data[:, None] * mean_inv_bw[None, :]
                + n_edges_i[:, None] * mean_lat[None, :])

    avg_c = float(np.mean(compute)) if N else 1.0
    compute_ratio = np.broadcast_to(compute[None, :] / max(avg_c, 1e-9), (T, N))

    cpu_fit = (resource[:, None] <= res_cpu[None, :] + 1e-9).astype(float)
    res_mem = np.array([nodes[k].res_mem for k in node_ids], dtype=float)
    mem_fit = (mem[:, None] <= res_mem[None, :] + 1e-9).astype(float)
    deadline_fit = ((eft[:, None] + exec_mat) <= instance.deadline + 1e-9).astype(float)

    inter = np.stack([
        _minmax(exec_mat), pfail, _minmax(recrisk), _minmax(cost_mat),
        _minmax(comm_mat), np.asarray(compute_ratio, dtype=float),
        cpu_fit, mem_fit, deadline_fit,
    ], axis=2)   # [T, N, INTER_DIM]

    edge_index = _knn_edges(instance, factory, node_ids, cfg)

    return InstanceFeatures(
        iid=instance.iid, task_ids=task_ids, node_ids=node_ids,
        task_feat=task_feat.astype(np.float32),
        node_feat=node_feat.astype(np.float32),
        node_domain=node_domain,
        edge_index=edge_index,
        inter=inter.astype(np.float32),
        task_pos=task_pos, node_pos=node_pos,
    )
