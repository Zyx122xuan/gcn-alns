"""Train the GCN scoring model (Spec sec.17, 5.5.7).

Loss = pointwise BCE on (chosen vs. feasible-unchosen) candidates
       + ``rank_loss_weight`` * pairwise BPR ranking loss,
with L2 regularisation via Adam ``weight_decay``. Negative candidates are
sampled (``neg_per_pos`` per positive). Early stopping on validation loss.

CLI::

    python -m gcnalns.gcn.train --data data/gcn --out results/gcn_model.pt
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, load_config
from .dataset import InstanceSample
from .model import GCNScorer


def _to_tensors(s: InstanceSample, device):
    f = s.feats
    return {
        "node_feat": torch.tensor(f.node_feat, dtype=torch.float32, device=device),
        "node_domain": torch.tensor(f.node_domain, dtype=torch.long, device=device),
        "edge_index": torch.tensor(f.edge_index, dtype=torch.long, device=device),
        "task_feat": torch.tensor(f.task_feat, dtype=torch.float32, device=device),
        "inter": torch.tensor(f.inter, dtype=torch.float32, device=device),
        "pos": s.pos,
        "feas_mask": s.feas_mask,
    }


def _positives_by_task(pos: np.ndarray, T: int) -> dict[int, set]:
    d: dict[int, set] = {}
    for tp, kp in pos:
        d.setdefault(int(tp), set()).add(int(kp))
    return d


def _sample_pairs(s, neg_per_pos: int, batch_size: int, rng: np.random.Generator):
    """Return (task_idx, pos_node, neg_node[neg]) numpy arrays for one instance."""
    pos = s["pos"]
    feas = s["feas_mask"]
    if len(pos) == 0:
        return None
    by_task = _positives_by_task(pos, feas.shape[0])
    t_list, p_list, n_list = [], [], []
    for tp, kp in pos:
        tp, kp = int(tp), int(kp)
        feasible = np.where(feas[tp] > 0)[0]
        pool = np.array([k for k in feasible if k not in by_task[tp]], dtype=np.int64)
        if pool.size == 0:
            continue
        replace = pool.size < neg_per_pos
        negs = rng.choice(pool, size=neg_per_pos, replace=replace)
        t_list.append(tp)
        p_list.append(kp)
        n_list.append(negs)
    if not t_list:
        return None
    t_arr = np.array(t_list, dtype=np.int64)
    p_arr = np.array(p_list, dtype=np.int64)
    n_arr = np.stack(n_list, axis=0)               # [P, neg]
    if t_arr.size > batch_size:                    # subsample to batch budget
        sel = rng.choice(t_arr.size, size=batch_size, replace=False)
        t_arr, p_arr, n_arr = t_arr[sel], p_arr[sel], n_arr[sel]
    return t_arr, p_arr, n_arr


def _instance_loss(model, tens, t_arr, p_arr, n_arr, rank_w, device):
    h_node = model.encode_nodes(tens["node_feat"], tens["node_domain"], tens["edge_index"])
    h_task = model.encode_tasks(tens["task_feat"])
    P, neg = n_arr.shape
    t_t = torch.tensor(t_arr, dtype=torch.long, device=device)
    p_t = torch.tensor(p_arr, dtype=torch.long, device=device)
    inter = tens["inter"]

    s_pos = model.score_pairs(h_task, h_node, t_t, p_t, inter[t_t, p_t])
    t_rep = t_t.unsqueeze(1).expand(P, neg).reshape(-1)
    n_t = torch.tensor(n_arr.reshape(-1), dtype=torch.long, device=device)
    s_neg = model.score_pairs(h_task, h_node, t_rep, n_t, inter[t_rep, n_t])

    bce = (F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))
           + F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg)))
    rank = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg.view(P, neg)).mean()
    return bce + rank_w * rank


def train(data: dict, cfg: Config, device: str = "cpu", verbose: bool = True):
    g = cfg["gcn"]
    dev = torch.device(device)
    torch.manual_seed(int(cfg.get("seeds.gcn_training", 2026)))
    rng = np.random.default_rng(int(cfg.get("seeds.gcn_training", 2026)))

    samples = data["samples"]
    split = data["split"]
    num_domains = data["num_domains"]
    tens = {iid: _to_tensors(s, dev) for iid, s in samples.items()}

    model = GCNScorer(cfg, num_domains).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(g["learning_rate"]),
                           weight_decay=float(g["weight_decay"]))
    neg = int(g["neg_per_pos"])
    bs = int(g["batch_size"])
    rank_w = float(g["rank_loss_weight"])
    max_epochs = int(g["max_epochs"])
    patience = int(g["early_stop_patience"])

    def epoch_pass(iids, train_mode: bool):
        model.train(train_mode)
        total, count = 0.0, 0
        for iid in iids:
            pairs = _sample_pairs(tens[iid], neg, bs, rng)
            if pairs is None:
                continue
            t_arr, p_arr, n_arr = pairs
            if train_mode:
                opt.zero_grad()
            with torch.set_grad_enabled(train_mode):
                loss = _instance_loss(model, tens[iid], t_arr, p_arr, n_arr, rank_w, dev)
                if train_mode:
                    loss.backward()
                    opt.step()
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    best_val = float("inf")
    best_state = None
    wait = 0
    history = []
    for ep in range(max_epochs):
        tr = epoch_pass(split["train"], True)
        va = epoch_pass(split["val"], False) if split["val"] else tr
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if verbose:
            print(f"  epoch {ep:3d}  train={tr:.4f}  val={va:.4f}")
        if va < best_val - 1e-5:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep} (best val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val": best_val,
                   "num_domains": num_domains}


def save_model(model: GCNScorer, meta: dict, cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "num_domains": meta["num_domains"],
        "gcn_cfg": cfg["gcn"],
        "meta": {k: v for k, v in meta.items() if k != "history"},
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GCN scoring model")
    ap.add_argument("--data", default="data/gcn")
    ap.add_argument("--out", default="results/gcn_model.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(Path(args.data) / "dataset.pkl", "rb") as fh:
        data = pickle.load(fh)
    if not data["samples"]:
        raise SystemExit("dataset is empty (no feasible demonstrations found)")

    print(f"Training GCN on {len(data['split']['train'])} train instances ...")
    t0 = time.perf_counter()
    model, meta = train(data, cfg, device=args.device)
    save_model(model, meta, cfg, args.out)
    print(f"Saved model -> {args.out}  (best val={meta['best_val']:.4f}, "
          f"{time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
