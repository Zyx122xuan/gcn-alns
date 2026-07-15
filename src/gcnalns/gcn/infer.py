"""GCN inference: load a trained model and expose a ``gcn_score(i, k)`` hook.

The hook returns a value in [0, 1] (sigmoid of the learned logit), higher =
more preferred. It is consumed by ``InsertCost`` as ``- w6 * GCNScore`` so the
GCN biases the search toward node assignments resembling the high-quality
demonstrations, while *never* overriding the hard constraints / recovery
verification done elsewhere. Resource / constraint-violating actions are masked
to the lowest preference (score 0) so the GCN never recommends them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..config import Config, load_config
from ..core.model import Instance
from ..data.links import LinkFactory
from .features import build_features
from .model import GCNScorer


class GCNScorerRunner:
    """Wraps a trained model; builds per-instance score tables on demand."""

    def __init__(self, model: GCNScorer, cfg: Config, has_mode_head: bool = False):
        self.model = model
        self.cfg = cfg
        self.has_mode_head = has_mode_head
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path, cfg: Config | None = None) -> "GCNScorerRunner":
        cfg = cfg or load_config()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = GCNScorer(cfg, int(ckpt["num_domains"]))
        # strict=False so old checkpoints without the FT-mode head still load
        # (the mode head then stays random and is disabled via has_mode_head).
        model.load_state_dict(ckpt["state_dict"], strict=False)
        has_mode = bool(ckpt.get("has_mode_head",
                                 ckpt.get("meta", {}).get("has_mode_head", False)))
        return cls(model, cfg, has_mode_head=has_mode)

    @torch.no_grad()
    def score_table(self, instance: Instance, factory: LinkFactory) -> np.ndarray:
        """Return a [T, N] preference table (sigmoid scores) aligned to instance ids.

        Rows/cols indexed by the instance's task/node id order; masked
        (infeasible) actions are set to 0.
        """
        feats = build_features(instance, self.cfg, factory)
        node_feat = torch.tensor(feats.node_feat, dtype=torch.float32)
        node_domain = torch.tensor(feats.node_domain, dtype=torch.long)
        edge_index = torch.tensor(feats.edge_index, dtype=torch.long)
        task_feat = torch.tensor(feats.task_feat, dtype=torch.float32)
        inter = torch.tensor(feats.inter, dtype=torch.float32)

        h_node = self.model.encode_nodes(node_feat, node_domain, edge_index)
        h_task = self.model.encode_tasks(task_feat)

        T, N = feats.num_tasks, feats.num_nodes
        t_idx = torch.arange(T).repeat_interleave(N)
        k_idx = torch.arange(N).repeat(T)
        logits = self.model.score_pairs(h_task, h_node, t_idx, k_idx,
                                        inter.view(T * N, -1))
        scores = torch.sigmoid(logits).view(T, N).numpy()

        # mask infeasible candidates (cpu/mem/deadline bits) to lowest preference
        from .features import INTERACTION_FEATURE_NAMES
        bidx = [INTERACTION_FEATURE_NAMES.index(b)
                for b in ("cpu_fit", "mem_fit", "deadline_fit")]
        feas = feats.inter[:, :, bidx].min(axis=2) > 0.5
        scores = np.where(feas, scores, 0.0)
        return scores.astype(np.float32), feats.task_pos, feats.node_pos

    def make_hook(self, instance: Instance, factory: LinkFactory) -> Callable[[int, int], float]:
        """Return ``gcn_score(i, k)`` for use as an ALNS / Repairer callback."""
        table, task_pos, node_pos = self.score_table(instance, factory)

        def gcn_score(i: int, k: int) -> float:
            ti = task_pos.get(i)
            kk = node_pos.get(k)
            if ti is None or kk is None:
                return 0.0
            return float(table[ti, kk])

        return gcn_score

    @torch.no_grad()
    def _tables(self, instance: Instance, factory: LinkFactory):
        """Compute the placement-score and (optional) FT-mode tables in one
        forward pass. Returns ``(scores[T,N], modes[T,N]|None, task_pos, node_pos)``.
        Infeasible actions are masked to 0 in the score table."""
        feats = build_features(instance, self.cfg, factory)
        node_feat = torch.tensor(feats.node_feat, dtype=torch.float32)
        node_domain = torch.tensor(feats.node_domain, dtype=torch.long)
        edge_index = torch.tensor(feats.edge_index, dtype=torch.long)
        task_feat = torch.tensor(feats.task_feat, dtype=torch.float32)
        inter = torch.tensor(feats.inter, dtype=torch.float32)

        h_node = self.model.encode_nodes(node_feat, node_domain, edge_index)
        h_task = self.model.encode_tasks(task_feat)

        T, N = feats.num_tasks, feats.num_nodes
        t_idx = torch.arange(T).repeat_interleave(N)
        k_idx = torch.arange(N).repeat(T)
        inter_flat = inter.view(T * N, -1)

        s_logits = self.model.score_pairs(h_task, h_node, t_idx, k_idx, inter_flat)
        scores = torch.sigmoid(s_logits).view(T, N).numpy()

        from .features import INTERACTION_FEATURE_NAMES
        bidx = [INTERACTION_FEATURE_NAMES.index(b)
                for b in ("cpu_fit", "mem_fit", "deadline_fit")]
        feas = feats.inter[:, :, bidx].min(axis=2) > 0.5
        scores = np.where(feas, scores, 0.0).astype(np.float32)

        modes = None
        if self.has_mode_head:
            m_logits = self.model.mode_pairs(h_task, h_node, t_idx, k_idx, inter_flat)
            modes = torch.sigmoid(m_logits).view(T, N).numpy().astype(np.float32)
        return scores, modes, feats.task_pos, feats.node_pos

    def make_hooks(self, instance: Instance, factory: LinkFactory):
        """Return ``(gcn_score, gcn_mode)`` callbacks. ``gcn_mode`` is ``None``
        when the loaded model has no trained FT-mode head."""
        scores, modes, task_pos, node_pos = self._tables(instance, factory)

        def gcn_score(i: int, k: int) -> float:
            ti = task_pos.get(i)
            kk = node_pos.get(k)
            if ti is None or kk is None:
                return 0.0
            return float(scores[ti, kk])

        if modes is None:
            return gcn_score, None

        def gcn_mode(i: int, k: int) -> float:
            ti = task_pos.get(i)
            kk = node_pos.get(k)
            if ti is None or kk is None:
                return 0.0
            return float(modes[ti, kk])

        return gcn_score, gcn_mode
