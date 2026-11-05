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
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, load_config
from ..utils.provenance import code_version, content_hash, file_hash, write_manifest
from .dataset import InstanceSample
from .model import GCNScorer


def _to_tensors(s: InstanceSample, device):
    f = s.feats
    pos_w = getattr(s, "pos_w", None)
    if pos_w is None:
        pos_w = np.ones(len(s.pos), dtype=np.float32)
    pos_mode = getattr(s, "pos_mode", None)
    if pos_mode is None:
        pos_mode = np.zeros(len(s.pos), dtype=np.float32)
    return {
        "node_feat": torch.tensor(f.node_feat, dtype=torch.float32, device=device),
        "node_domain": torch.tensor(f.node_domain, dtype=torch.long, device=device),
        "edge_index": torch.tensor(f.edge_index, dtype=torch.long, device=device),
        "task_feat": torch.tensor(f.task_feat, dtype=torch.float32, device=device),
        "inter": torch.tensor(f.inter, dtype=torch.float32, device=device),
        "pos": s.pos,
        "pos_w": np.asarray(pos_w, dtype=np.float32),
        "pos_mode": np.asarray(pos_mode, dtype=np.float32),
        "feas_mask": s.feas_mask,
    }


def _positives_by_task(pos: np.ndarray, T: int) -> dict[int, set]:
    d: dict[int, set] = {}
    for tp, kp in pos:
        d.setdefault(int(tp), set()).add(int(kp))
    return d


def _sample_pairs(s, neg_per_pos: int, batch_size: int, rng: np.random.Generator):
    """Return (task_idx, pos_node, neg_node[neg], pos_weight, pos_mode)."""
    pos = s["pos"]
    feas = s["feas_mask"]
    pos_w = s["pos_w"]
    pos_mode = s["pos_mode"]
    if len(pos) == 0:
        return None
    by_task = _positives_by_task(pos, feas.shape[0])
    t_list, p_list, n_list, w_list, m_list = [], [], [], [], []
    for j, (tp, kp) in enumerate(pos):
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
        w_list.append(float(pos_w[j]) if j < len(pos_w) else 1.0)
        m_list.append(float(pos_mode[j]) if j < len(pos_mode) else 0.0)
    if not t_list:
        return None
    t_arr = np.array(t_list, dtype=np.int64)
    p_arr = np.array(p_list, dtype=np.int64)
    n_arr = np.stack(n_list, axis=0)               # [P, neg]
    w_arr = np.array(w_list, dtype=np.float32)
    m_arr = np.array(m_list, dtype=np.float32)
    if t_arr.size > batch_size:                    # subsample to batch budget
        sel = rng.choice(t_arr.size, size=batch_size, replace=False)
        t_arr, p_arr, n_arr = t_arr[sel], p_arr[sel], n_arr[sel]
        w_arr, m_arr = w_arr[sel], m_arr[sel]
    return t_arr, p_arr, n_arr, w_arr, m_arr


def _instance_loss(model, tens, t_arr, p_arr, n_arr, w_arr, m_arr,
                   rank_w, mode_w, device, mode_pw=1.0):
    h_node = model.encode_nodes(tens["node_feat"], tens["node_domain"], tens["edge_index"])
    h_task = model.encode_tasks(tens["task_feat"])
    P, neg = n_arr.shape
    t_t = torch.tensor(t_arr, dtype=torch.long, device=device)
    p_t = torch.tensor(p_arr, dtype=torch.long, device=device)
    inter = tens["inter"]

    # normalise importance weights to mean 1 so the loss scale is stable
    w_t = torch.tensor(w_arr, dtype=torch.float32, device=device)
    w_t = w_t / (w_t.mean() + 1e-9) if w_t.numel() else w_t

    s_pos = model.score_pairs(h_task, h_node, t_t, p_t, inter[t_t, p_t])
    t_rep = t_t.unsqueeze(1).expand(P, neg).reshape(-1)
    n_t = torch.tensor(n_arr.reshape(-1), dtype=torch.long, device=device)
    s_neg = model.score_pairs(h_task, h_node, t_rep, n_t, inter[t_rep, n_t])

    bce_pos = F.binary_cross_entropy_with_logits(
        s_pos, torch.ones_like(s_pos), weight=w_t)
    bce_neg = F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg))
    rank = -(w_t * F.logsigmoid(s_pos.unsqueeze(1) - s_neg.view(P, neg)).mean(dim=1)).mean()

    loss = bce_pos + bce_neg + rank_w * rank
    # FT-mode head (Part C): supervise P(active) on the chosen positive pairs.
    # active positives are rare (~2.5%), so up-weight them via pos_weight to
    # stop the head collapsing to always-passive.
    if mode_w > 0.0:
        m_logit = model.mode_pairs(h_task, h_node, t_t, p_t, inter[t_t, p_t])
        m_target = torch.tensor(m_arr, dtype=torch.float32, device=device)
        pw = torch.tensor(float(mode_pw), dtype=torch.float32, device=device)
        loss = loss + mode_w * F.binary_cross_entropy_with_logits(
            m_logit, m_target, weight=w_t, pos_weight=pw)
    return loss


@torch.no_grad()
def evaluate_ranking(model, tens: dict, iids: list[str]) -> dict[str, float]:
    """Test-set ranking quality: top-1 hit rate + pairwise AUC over feasible cands."""
    model.eval()
    top1_hit, top1_tot = 0, 0
    auc_corr, auc_tot = 0.0, 0.0
    for iid in iids:
        s = tens[iid]
        pos = s["pos"]
        feas = s["feas_mask"]
        if len(pos) == 0:
            continue
        h_node = model.encode_nodes(s["node_feat"], s["node_domain"], s["edge_index"])
        h_task = model.encode_tasks(s["task_feat"])
        inter = s["inter"]
        by_task = _positives_by_task(pos, feas.shape[0])
        for tp, kset in by_task.items():
            feasible = np.where(feas[tp] > 0)[0]
            if feasible.size == 0:
                continue
            t_idx = torch.full((feasible.size,), tp, dtype=torch.long, device=h_node.device)
            k_idx = torch.tensor(feasible, dtype=torch.long, device=h_node.device)
            logits = model.score_pairs(h_task, h_node, t_idx, k_idx, inter[t_idx, k_idx])
            scores = logits.detach().cpu().numpy()
            order = feasible[np.argsort(-scores)]
            top1_tot += 1
            if int(order[0]) in kset:
                top1_hit += 1
            pos_mask = np.array([int(k) in kset for k in feasible])
            ps, ns = scores[pos_mask], scores[~pos_mask]
            if ps.size and ns.size:
                for p in ps:
                    auc_corr += float(np.sum(p > ns) + 0.5 * np.sum(p == ns))
                    auc_tot += ns.size
    return {
        "top1_hit_rate": top1_hit / max(top1_tot, 1),
        "pairwise_auc": auc_corr / max(auc_tot, 1.0),
        "n_tasks_scored": top1_tot,
    }


@torch.no_grad()
def evaluate_mode(model, tens: dict, iids: list[str]) -> dict[str, float]:
    """FT-mode head quality on chosen positives: AUC(active vs passive),
    accuracy@0.5, and the active-label prevalence (so we can tell whether the
    head learned anything beyond the majority-passive prior)."""
    model.eval()
    scores, labels = [], []
    for iid in iids:
        s = tens[iid]
        pos = s["pos"]
        if len(pos) == 0:
            continue
        m = s["pos_mode"]
        h_node = model.encode_nodes(s["node_feat"], s["node_domain"], s["edge_index"])
        h_task = model.encode_tasks(s["task_feat"])
        inter = s["inter"]
        t_idx = torch.tensor(pos[:, 0], dtype=torch.long, device=h_node.device)
        k_idx = torch.tensor(pos[:, 1], dtype=torch.long, device=h_node.device)
        logit = model.mode_pairs(h_task, h_node, t_idx, k_idx, inter[t_idx, k_idx])
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        scores.append(prob)
        labels.append(np.asarray(m, dtype=np.float32))
    if not scores:
        return {"mode_auc": float("nan"), "mode_acc": float("nan"),
                "n_active": 0, "n_pos": 0}
    sc = np.concatenate(scores)
    lb = np.concatenate(labels)
    pos_s, neg_s = sc[lb > 0.5], sc[lb <= 0.5]
    if pos_s.size and neg_s.size:
        corr = sum(float(np.sum(p > neg_s) + 0.5 * np.sum(p == neg_s)) for p in pos_s)
        auc = corr / (pos_s.size * neg_s.size)
    else:
        auc = float("nan")
    acc = float(np.mean((sc > 0.5) == (lb > 0.5)))
    return {"mode_auc": auc, "mode_acc": acc,
            "n_active": int((lb > 0.5).sum()), "n_pos": int(lb.size)}


def train(data: dict, cfg: Config, device: str = "cpu", verbose: bool = True):
    g = cfg["gcn"]
    dev = torch.device(device)
    torch.manual_seed(int(cfg.get("seeds.gcn_training", 2026)))
    rng = np.random.default_rng(int(cfg.get("seeds.gcn_training", 2026)))

    samples = data["samples"]
    split = data["split"]
    assigned = [iid for ids in split.values() for iid in ids]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Training split leakage: an iid appears in multiple splits")
    if set(assigned) != set(samples):
        raise ValueError("Training split assignments do not match dataset samples")
    if not split.get("train"):
        raise ValueError("No feasible training demonstrations; increase the teacher budget")
    provenance = data.get("provenance")
    if provenance and provenance.get("protocol_sha256"):
        if split.get("test"):
            raise ValueError("Frozen solver test instances must not enter GCN training")
        if not split.get("val"):
            raise ValueError("No validation demonstrations; do not borrow from the test set")
        for role in ("train", "val"):
            allowed = {r["iid"] for r in provenance["source_splits"][role]}
            if not set(split[role]).issubset(allowed):
                raise ValueError(f"Samples outside frozen {role} split")
    num_domains = data["num_domains"]
    tens = {iid: _to_tensors(s, dev) for iid, s in samples.items()}

    model = GCNScorer(cfg, num_domains).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(g["learning_rate"]),
                           weight_decay=float(g["weight_decay"]))
    neg = int(g["neg_per_pos"])
    bs = int(g["batch_size"])
    rank_w = float(g["rank_loss_weight"])
    mode_w = float(g.get("mode_loss_weight", 0.5))
    max_epochs = int(g["max_epochs"])
    patience = int(g["early_stop_patience"])

    # FT-mode class imbalance: pos_weight for the mode-BCE. If mode_pos_weight<=0
    # (default), auto-compute from the train-split active/passive ratio (capped),
    # so the head is not dominated by the ~97.5% passive majority.
    cfg_mpw = float(g.get("mode_pos_weight", 0.0))
    if cfg_mpw > 0:
        mode_pw = cfg_mpw
    else:
        n_act = n_pas = 0
        for iid in split["train"]:
            pm = tens[iid]["pos_mode"]
            n_act += int((pm > 0.5).sum())
            n_pas += int((pm <= 0.5).sum())
        mode_pw = min(float(n_pas) / max(n_act, 1), 50.0) if n_act > 0 else 1.0
    if verbose:
        print(f"  mode-head pos_weight={mode_pw:.2f} "
              f"(active positives up-weighted for class imbalance)")

    def epoch_pass(iids, train_mode: bool):
        model.train(train_mode)
        total, count = 0.0, 0
        for iid in iids:
            pairs = _sample_pairs(tens[iid], neg, bs, rng)
            if pairs is None:
                continue
            t_arr, p_arr, n_arr, w_arr, m_arr = pairs
            if train_mode:
                opt.zero_grad()
            with torch.set_grad_enabled(train_mode):
                loss = _instance_loss(model, tens[iid], t_arr, p_arr, n_arr,
                                      w_arr, m_arr, rank_w, mode_w, dev,
                                      mode_pw=mode_pw)
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

    # Report test-set ranking quality so we can confirm the model learned a
    # useful signal (vs. ~random when AUC ~ 0.5 / top-1 ~ chance).
    rank_metrics = {}
    mode_metrics = {}
    for split_name in ("val", "test"):
        ids = split.get(split_name) or []
        if ids:
            rank_metrics[split_name] = evaluate_ranking(model, tens, ids)
            if verbose:
                m = rank_metrics[split_name]
                print(f"  [{split_name}] top1_hit_rate={m['top1_hit_rate']:.3f} "
                      f"pairwise_auc={m['pairwise_auc']:.3f} "
                      f"(tasks={m['n_tasks_scored']})")
            if mode_w > 0.0:
                mode_metrics[split_name] = evaluate_mode(model, tens, ids)
                if verbose:
                    mm = mode_metrics[split_name]
                    print(f"  [{split_name}] mode_auc={mm['mode_auc']:.3f} "
                          f"mode_acc={mm['mode_acc']:.3f} "
                          f"(active={mm['n_active']}/{mm['n_pos']})")

    return model, {"history": history, "best_val": best_val,
                   "num_domains": num_domains, "ranking": rank_metrics,
                   "mode": mode_metrics, "has_mode_head": mode_w > 0.0,
                   "mode_pos_weight": mode_pw,
                   "provenance": {
                       "dataset": provenance,
                       "sample_splits": split,
                       "config": copy.deepcopy(cfg.data),
                       "config_sha256": content_hash(cfg.data),
                       "code": code_version(),
                   }}


def save_model(model: GCNScorer, meta: dict, cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "num_domains": meta["num_domains"],
        "has_mode_head": bool(meta.get("has_mode_head", False)),
        "gcn_cfg": cfg["gcn"],
        "training_provenance": meta.get("provenance"),
        "meta": {k: v for k, v in meta.items() if k != "history"},
    }, path)
    write_manifest({"schema_version": 1, "checkpoint_sha256": file_hash(path),
                    "training": meta.get("provenance")},
                   path.with_name(path.name + ".provenance.json"))


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
    meta["provenance"]["dataset_sha256"] = file_hash(Path(args.data) / "dataset.pkl")
    save_model(model, meta, cfg, args.out)
    print(f"Saved model -> {args.out}  (best val={meta['best_val']:.4f}, "
          f"{time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
