"""GCN scoring model (Spec sec.16, 5.5.7).

Architecture::

    h_k = GCN(physical graph; node_feat || domain_embedding)   # 2 layers, ReLU
    h_i = MLP(task_feat)                                        # task encoder
    Score(i,k) = MLP([h_i, h_k, h_i (.) h_k, z_ik])            # bilinear-style head

The graph convolution is the standard symmetric-normalised propagation
``H' = D^-1/2 (A + I) D^-1/2 H W`` over the physical-node graph built from
``edge_index``. It is implemented with a dense adjacency (node pools are <= a
few hundred, so this is cheap) which keeps training and inference deterministic
and independent of the exact torch-geometric build. ``torch-geometric`` is an
optional environment dependency (its datasets/utilities may be used elsewhere)
but the convolution here is self-contained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .features import INTER_DIM, NODE_DIM, TASK_DIM

# Kept for backward-compatible introspection; the model always uses the
# self-contained dense GCN convolution below for train/infer consistency.
_HAS_PYG = False


def _normalized_adj(edge_index: torch.Tensor, n: int,
                    device: torch.device) -> torch.Tensor:
    """Dense D^-1/2 (A+I) D^-1/2 from an undirected ``edge_index`` [2, E]."""
    adj = torch.eye(n, device=device)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        adj[src, dst] = 1.0
    deg = adj.sum(dim=1)
    dinv = torch.pow(deg.clamp(min=1.0), -0.5)
    return dinv.unsqueeze(1) * adj * dinv.unsqueeze(0)


class _DenseGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return adj_norm @ self.lin(x)


class GCNScorer(nn.Module):
    """Joint task/node encoder + task-node scoring head."""

    def __init__(self, cfg: Config, num_domains: int):
        super().__init__()
        g = cfg["gcn"]
        h = int(g["hidden_dim"])
        emb = int(g["emb_dim"])
        self.dropout = float(g["dropout"])
        self.num_layers = int(g["num_layers"])

        self.domain_emb = nn.Embedding(num_domains, emb)
        node_in = NODE_DIM + emb

        self.convs = nn.ModuleList()
        self.convs.append(_DenseGCNConv(node_in, h))
        for _ in range(self.num_layers - 1):
            self.convs.append(_DenseGCNConv(h, h))

        self.task_mlp = nn.Sequential(
            nn.Linear(TASK_DIM, h), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(h, h),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * h + INTER_DIM, h), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(h, 1),
        )
        # Second head (Part C): predicts P(active backup) for a (task, node)
        # placement, so the GCN can also drive the fault-tolerance mode decision.
        self.mode_mlp = nn.Sequential(
            nn.Linear(3 * h + INTER_DIM, h), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(h, 1),
        )
        # Second head (Part C): predicts P(active backup) for a (task, node)
        # pair, so the learned guidance can drive the fault-tolerance mode - not
        # just placement - which is what actually moves the feasibility needle.
        self.mode_mlp = nn.Sequential(
            nn.Linear(3 * h + INTER_DIM, h), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(h, 1),
        )

    # -- encoders -----------------------------------------------------------
    def encode_nodes(self, node_feat: torch.Tensor, node_domain: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        emb = self.domain_emb(node_domain)
        x = torch.cat([node_feat, emb], dim=1)
        adj = _normalized_adj(edge_index, node_feat.size(0), node_feat.device)
        for li, conv in enumerate(self.convs):
            x = conv(x, adj)
            if li < len(self.convs) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

    def encode_tasks(self, task_feat: torch.Tensor) -> torch.Tensor:
        return self.task_mlp(task_feat)

    # -- scoring ------------------------------------------------------------
    def score_pairs(self, h_task: torch.Tensor, h_node: torch.Tensor,
                    task_idx: torch.Tensor, node_idx: torch.Tensor,
                    inter: torch.Tensor) -> torch.Tensor:
        """Score a batch of (task, node) pairs. Returns raw logits [P]."""
        hi = h_task[task_idx]
        hk = h_node[node_idx]
        feat = torch.cat([hi, hk, hi * hk, inter], dim=1)
        return self.score_mlp(feat).squeeze(-1)

    def mode_pairs(self, h_task: torch.Tensor, h_node: torch.Tensor,
                   task_idx: torch.Tensor, node_idx: torch.Tensor,
                   inter: torch.Tensor) -> torch.Tensor:
        """FT-mode logits [P] for a batch of (task, node) pairs (P(active))."""
        hi = h_task[task_idx]
        hk = h_node[node_idx]
        feat = torch.cat([hi, hk, hi * hk, inter], dim=1)
        return self.mode_mlp(feat).squeeze(-1)

    def forward(self, node_feat, node_domain, edge_index, task_feat,
                task_idx, node_idx, inter) -> torch.Tensor:
        h_node = self.encode_nodes(node_feat, node_domain, edge_index)
        h_task = self.encode_tasks(task_feat)
        return self.score_pairs(h_task, h_node, task_idx, node_idx, inter)
