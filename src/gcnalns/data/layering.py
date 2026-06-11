"""Map raw machines to computing-continuum Nodes (Spec sec.3, sec.4, sec.5).

Implements:
* normalized capacity score cap_k
* node quality score q_k and cloud/edge/device split (top20/mid50/bottom30),
  with the volatility / capacity correction rules
* effective compute f_k = f_std * (0.5 + 2.5*cap_k)
* layer cost coefficient and base failure rate, with load/var adjusted
  lambda_k = lambda_layer * (1 + g1*load + g2*var)
* fault domain FD_k = (layer, hash(machine_id) % G_layer)
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..config import Config
from ..core.model import Node


def _hash_mod(s: str, m: int) -> int:
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return h % max(1, m)


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def build_nodes(machines: list[dict], cfg: Config) -> dict[int, Node]:
    """Convert raw machine dicts into a dict of physical Nodes."""
    lay = cfg["layering"]
    cap_w = lay["cap_weights"]
    q_w = lay["q_weights"]
    f_std = float(cfg["f_std"])

    cpu_cap = np.array([m["cpu_cap"] for m in machines], dtype=float)
    mem_cap = np.array([m["mem_cap"] for m in machines], dtype=float)
    cpu_util = np.array([m["cpu_util"] for m in machines], dtype=float)
    mem_util = np.array([m["mem_util"] for m in machines], dtype=float)
    cpu_var = np.array([m["cpu_var"] for m in machines], dtype=float)
    mem_var = np.array([m["mem_var"] for m in machines], dtype=float)

    cap_k = cap_w["cpu"] * _minmax(cpu_cap) + cap_w["mem"] * _minmax(mem_cap)
    load_k = 0.5 * cpu_util + 0.5 * mem_util
    var_k = 0.5 * cpu_var + 0.5 * mem_var
    var_norm = _minmax(var_k)

    q_k = q_w["cap"] * cap_k + q_w["load"] * (1.0 - load_k) + q_w["var"] * (1.0 - var_norm)

    n = len(machines)
    order = np.argsort(-q_k)  # descending quality
    cloud_n = int(round(lay["split"]["cloud"] * n))
    edge_n = int(round(lay["split"]["edge"] * n))
    layer_of = {}
    for rank, idx in enumerate(order):
        if rank < cloud_n:
            layer_of[idx] = "cloud"
        elif rank < cloud_n + edge_n:
            layer_of[idx] = "edge"
        else:
            layer_of[idx] = "device"

    # Correction rules: very volatile high-cap nodes capped at edge;
    # bottom-20% capacity forced to device.
    var_top20 = np.quantile(var_norm, 0.80)
    cap_bottom20 = np.quantile(cap_k, 0.20)
    for idx in range(n):
        if layer_of[idx] == "cloud" and var_norm[idx] >= var_top20:
            layer_of[idx] = "edge"
        if cap_k[idx] <= cap_bottom20:
            layer_of[idx] = "device"

    layers_cfg = lay["layers"]
    g1 = lay["failure_load_gamma1"]
    g2 = lay["failure_var_gamma2"]
    groups = cfg["fault_domain"]["groups"]

    nodes: dict[int, Node] = {}
    for idx, m in enumerate(machines):
        layer = layer_of[idx]
        f_k = f_std * (lay["fk_base"] + lay["fk_scale"] * float(cap_k[idx]))
        base_lambda = layers_cfg[layer]["base_failure_rate"]
        # Use trace-derived failure rate if provided, else layer default.
        if m.get("failure_rate") is not None:
            lam = float(m["failure_rate"])
        else:
            lam = base_lambda * (1.0 + g1 * float(load_k[idx]) + g2 * float(var_norm[idx]))
        fd = f"{layer}-{_hash_mod(m['machine_id'], groups[layer])}"
        nodes[idx] = Node(
            nid=idx,
            compute=float(f_k),
            cpu_cap=float(m["cpu_cap"]),
            mem_cap=float(m["mem_cap"]),
            load=float(load_k[idx]),
            load_var=float(var_norm[idx]),
            failure_rate=lam,
            cost_coef=float(layers_cfg[layer]["cost_coef"]),
            layer=layer,
            fault_domain=fd,
        )
    return nodes
