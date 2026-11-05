"""Rebuild the new-protocol report strictly from completed, versioned runs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from gcnalns.eval.aggregate import aggregate
from gcnalns.utils.provenance import write_manifest


def holm(p_values):
    indexed = sorted(enumerate(p_values), key=lambda item: 1.0 if item[1] is None else item[1])
    adjusted, previous = [None] * len(p_values), 0.0
    for rank, (i, p) in enumerate(indexed):
        value = min(1.0, (len(p_values) - rank) * (1.0 if p is None else p))
        previous = max(previous, value)
        adjusted[i] = None if p is None else previous
    return adjusted


def paired(left, right, metric, relative=False):
    """Pair exact iid/seed, then use instance means as independent units."""
    keys = ["iid", "rep"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate paired run key")
    a = left[keys + [metric] + ([] if metric == "feasible" else ["feasible"])].copy()
    b = right[keys + [metric] + ([] if metric == "feasible" else ["feasible"])].copy()
    data = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if metric != "feasible":
        data = data[data.feasible_a.astype(bool) & data.feasible_b.astype(bool)]
    xa = pd.to_numeric(data[f"{metric}_a"], errors="coerce").astype(float)
    xb = pd.to_numeric(data[f"{metric}_b"], errors="coerce").astype(float)
    mask = np.isfinite(xa) & np.isfinite(xb)
    if metric == "first_feasible_iter":
        mask &= (xa >= 0) & (xb >= 0)
    data = data.loc[mask].copy()
    diff = xa[mask] - xb[mask]
    if relative:
        diff = 100.0 * diff / xb[mask].abs().clip(lower=1e-12)
    elif metric == "feasible":
        diff = 100.0 * diff
    values = diff.groupby(data.iid).mean().to_numpy(dtype=float)
    result = {"n_instances": len(values), "n_paired_runs": len(data),
              "mean_difference": float(values.mean()) if len(values) else None,
              "ci95": None, "p_t": None, "p_wilcoxon": None}
    if len(values) < 2:
        return result
    rng = np.random.default_rng(2026)
    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    result["ci95"] = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    if np.allclose(values, 0, rtol=0, atol=1e-12):
        result.update(p_t=1.0, p_wilcoxon=1.0)
    else:
        p = float(stats.ttest_1samp(values, 0).pvalue)
        result["p_t"] = p if math.isfinite(p) else None
        try:
            result["p_wilcoxon"] = float(stats.wilcoxon(values).pvalue)
        except ValueError:
            pass
    return result


def summary(records):
    feasible = records[records.feasible.astype(bool)]
    result = {"n_runs": len(records), "n_instances": records.iid.nunique(),
              "feasible_macro": float(records.groupby("iid").feasible.mean().mean())}
    for field in ("cost_total", "safety_margin", "first_feasible_iter"):
        values = pd.to_numeric(feasible[field], errors="coerce")
        values = values[np.isfinite(values)]
        if field == "first_feasible_iter":
            values = values[values >= 0]
        result[field] = float(values.mean()) if len(values) else None
    result["runtime"] = float(records.runtime.mean())
    return result


def table(headers, rows):
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4g}" if math.isfinite(value) else "—"
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]])


def build_report(root, plots=True):
    root = Path(root)
    plan = json.loads((root / "pipeline_plan.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "model_selection.json").read_text(encoding="utf-8"))
    raw = root / "test/raw_json"
    groups = {p.stem: pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
              for p in raw.glob("*.json")}
    expected = plan["expected_test_records"]
    main_tags = [f"main_{s}" for s in ("small", "medium", "large")]
    for scale, n in expected["main"].items():
        if f"main_{scale}" not in groups or len(groups[f"main_{scale}"]) != n:
            raise ValueError(f"Incomplete main comparison: {scale}")
    for prefix, key in (("ablation_", "ablation"), ("budget_", "budget"), ("model_ab_", "model_ab")):
        actual = sum(len(rows) for name, rows in groups.items() if name.startswith(prefix))
        if actual != expected[key]:
            raise ValueError(f"Incomplete {key}: {actual} != {expected[key]}")
    all_rows = pd.concat(groups.values(), ignore_index=True)
    if not ((all_rows.evaluation_split == "test").all()
            and (all_rows.protocol_sha256 == plan["protocol_sha256"]).all()
            and (all_rows.code_sha256 == plan["code"]["sha256"]).all()):
        raise ValueError("Mixed experiment provenance in report inputs")
    main = pd.concat([groups[t] for t in main_tags], ignore_index=True)
    main_summary = {f"{scale}/{algorithm}": summary(rows)
                    for (scale, algorithm), rows in main.groupby(["scale", "algorithm"])}
    main_pairs = {}
    pair_rows = []
    for scale in ("small", "medium", "large"):
        a = main[(main.scale == scale) & (main.algorithm == "GCN-ALNS")]
        b = main[(main.scale == scale) & (main.algorithm == "ALNS")]
        for field, relative in (("feasible", False), ("cost_total", True),
                                ("safety_margin", False), ("first_feasible_iter", False)):
            value = paired(a, b, field, relative)
            main_pairs[f"{scale}/{field}"] = value
            pair_rows.append([scale, field, value["n_instances"], value["mean_difference"],
                              value["ci95"], value["p_t"], value["p_wilcoxon"]])
    full = groups["ablation_Full"]
    variant_tags = sorted(t for t in groups if t.startswith("ablation_") and t != "ablation_Full")
    ablation = {}
    for field, relative in (("feasible", False), ("cost_total", True),
                            ("safety_margin", False), ("first_feasible_iter", False)):
        values = [paired(full, groups[t], field, relative) for t in variant_tags]
        adjusted = holm([v["p_t"] for v in values])
        for tag, value, p_adj in zip(variant_tags, values, adjusted):
            value["p_t_holm"] = p_adj
            ablation.setdefault(tag, {})[field] = value
    families = {name: {str(algo): summary(rows) for algo, rows in frame.groupby("algorithm")}
                for name, frame in groups.items() if name.startswith(("budget_", "model_ab_"))}
    output = {"protocol_sha256": plan["protocol_sha256"], "model_selection": selection,
              "run_counts": {name: len(frame) for name, frame in groups.items()},
              "main": main_summary, "main_paired": main_pairs,
              "ablation_paired": ablation, "other_studies": families}
    write_manifest(output, root / "summary.json")
    aggregate(raw, root / "test/csv")
    if plots:
        from gcnalns.viz.plots import make_all
        make_all(root / "test/csv", root / "test/figures")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ab_tags = ["model_ab_alns", "model_ab_a2_off", "model_ab_a2_on"]
        ax.bar(["ALNS", "GCN A2-off", "GCN A2-on"],
               [summary(groups[t])["feasible_macro"] for t in ab_tags])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Model / solver")
        ax.set_ylabel("SLA-feasible ratio (instance macro mean)")
        ax.set_title("Model A/B on held-out small instances")
        count = groups["model_ab_alns"].iid.nunique()
        repeats = plan["model_ab"]["repeats"]
        fig.text(0.02, 0.01, f"Source: model_ab_* test records; {count} instances x {repeats} seeds", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(root / "test/figures/fig_model_ab.png", dpi=150)
        plt.close(fig)
    rows = [[key, x["n_instances"], x["n_runs"], 100*x["feasible_macro"],
             x["cost_total"], x["safety_margin"], x["first_feasible_iter"], x["runtime"]]
            for key, x in main_summary.items()]
    abl_rows = [[tag.removeprefix("ablation_"), values["feasible"]["n_instances"],
                 values["feasible"]["mean_difference"], values["feasible"]["p_t_holm"],
                 values["cost_total"]["mean_difference"], values["cost_total"]["p_t_holm"],
                 values["safety_margin"]["mean_difference"], values["safety_margin"]["p_t_holm"]]
                for tag, values in ablation.items()]
    family_rows = [[tag, algo, x["n_instances"], 100*x["feasible_macro"],
                    x["cost_total"], x["first_feasible_iter"]]
                   for tag, algos in families.items() for algo, x in algos.items()]
    report = ["# GCN-ALNS 独立评测重跑报告", "",
              f"协议：`{plan['protocol_sha256']}`。代码内容：`{plan['code']['sha256']}`。",
              f"本报告共 {len(all_rows)} 条新测试记录。teacher、训练与模型选择只使用 train/val；历史实验不并入。",
              f"验证集选出的模型为 **{selection['selected']}**，依据：{selection['reason']}。",
              "", "## 主对比", "",
              "可行率先按实例聚合随机重复再做宏平均；成本、安全余量、首达可行迭代为成功运行的描述均值，不同方法成功集合可能不同。runtime 为完整求解时间（秒），首达可行迭代不等同于实测首达可行时间。",
              "", table(["规模/算法", "实例数", "运行数", "可行率%", "可行成本", "安全余量秒", "首达可行迭代", "runtime秒"], rows),
              "", "## GCN-ALNS 与 ALNS 的配对比较", "",
              "差值为 GCN-ALNS − ALNS。可行率单位为百分点，成本为相对 ALNS 的百分比，安全余量为秒。成本、安全余量及迭代仅取相同实例/随机重复下双方均成功的运行；再按实例汇总。95% CI 为按实例 bootstrap（2000 次、seed=2026）；p 为配对实例差值检验。",
              "", table(["规模", "指标", "配对实例数", "均值差", "95% CI", "t检验p", "Wilcoxon p"], pair_rows),
              "", "## 消融", "",
              f"差值为 Full − 去组件变体。成本为相对变体百分比。各指标分别对 {len(variant_tags)} 个 Full-vs-variant 比较做 Holm 校正；全部字段与配对样本数见 summary.json。未定义的统计量显示为 —。",
              "", table(["变体", "实例数", "可行率差pp", "Holm p", "成本差%", "Holm p", "余量差秒", "Holm p"], abl_rows),
              "", "## 预算与模型 A/B", "",
              table(["tag", "算法", "实例数", "可行率%", "成功成本均值", "首达可行迭代"], family_rows),
              "", "## 图表与可复核材料", "",
              "图表位于 [test/figures](test/figures/)，逐次指标位于 [test/csv/runs.csv](test/csv/runs.csv)。配置/代码/模型/实例指纹位于 test/manifests，每次部署解位于 test/solutions。",
              "训练与验证结果、固定的模型选择规则及运行预算见 pipeline_plan.json、model_selection.json 和 logs/teacher_build_status.json。",
              ""]
    (root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"REPORT COMPLETE: {root / 'REPORT.md'}")
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    build_report(args.root, plots=not args.no_plots)
