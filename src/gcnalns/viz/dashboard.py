"""Interactive browser for the GCN-ALNS experiment runs.

Reads ``results/csv/runs.csv`` (one row per run) and links the same filters
the report uses: study, scale, SLA scenario, workflow, and algorithm.
Dimensional metrics can be restricted to SLA-feasible runs so low-feasibility
algorithms are not compared on a handful of easy instances.

Launch from the repository root::

    streamlit run src/gcnalns/viz/dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = ROOT / "results" / "csv" / "runs.csv"

STUDIES = ["主对比", "难档", "消融", "迭代预算", "模型 A/B"]
STUDY_BLURB = {
    "主对比": "main_small / main_medium / main_large，14 个算法。",
    "难档": "main_hard_*，SLA 只有 tight2 / tight3。",
    "消融": "14 个 GCN-ALNS 变体。差值以 Full 为基准。",
    "迭代预算": "small 规模，ALNS 与 GCN-ALNS，迭代上限 50–1000。",
    "模型 A/B": "small 规模，A2-off、A2-on 与 ALNS。",
}
SCALE_ORDER = ["small", "medium", "large"]
SLA_ORDER = ["tight", "tight2", "tight3", "medium", "loose"]
WORKFLOW_ORDER = ["Montage", "CyberShake", "Epigenomics", "LIGO", "SIPHT"]
ALGO_ORDER = [
    "HEFT", "PEFT", "No-FT", "Reactive-Only", "Full-Proactive",
    "Critical-Path Backup", "Criticality-Based Hybrid", "Reliability-Aware",
    "RECR-D", "HFWPF", "GA-Placement", "RA-MOMA",
    "ALNS", "GCN-ALNS",
    "GCN-ALNS (A2-off)", "GCN-ALNS (A2-on)",
]
ABLATION_ORDER = [
    "Full", "w/o GCN", "w/o Recovery Penalty", "w/o Fault-Domain",
    "w/o Deployment Risk", "w/o Dynamic Resched",
    "w/o Hybrid (all-passive)", "w/o Hybrid (all-active)",
    "w/o Warm Start", "w/o GCN Mode", "w/o GCN Destroy",
    "w/o Feasibility-First", "w/o Restart", "w/o Init-Diversify",
]
FULL_VARIANT = "GCN-ALNS [Full]"

# Quality / cost figures in the report are means over feasible runs only.
# Rates stay on every run: the feasible rate of the feasible subset is 1.
FEASIBLE_ONLY_METRICS = {
    "cost_total", "cost_per_task", "cost_compute", "cost_comm",
    "cost_reserve", "cost_recovery", "c_base", "c_worst", "r_max",
    "avg_recovery", "safety_margin", "first_feasible_iter",
}
COST_PARTS = [
    ("cost_compute", "计算", "#4C78A8"),
    ("cost_comm", "通信", "#F58518"),
    ("cost_reserve", "预留", "#54A24B"),
    ("cost_recovery", "恢复", "#E45756"),
]
METRICS = {
    "feasible": ("可行率", True, ".3f"),
    "cost_total": ("总成本", False, ".1f"),
    "cost_per_task": ("单任务成本", False, ".2f"),
    "safety_margin": ("安全余量", True, ".2f"),
    "r_max": ("最坏恢复 R_max", False, ".2f"),
    "first_feasible_iter": ("首达可行迭代", False, ".1f"),
    "runtime": ("求解时间 (s)", False, ".1f"),
    "active_backup_ratio": ("主动备份比例", True, ".3f"),
    "c_base": ("无故障 makespan", False, ".1f"),
    "c_worst": ("最坏 makespan", False, ".1f"),
    "avg_recovery": ("平均恢复时延", False, ".2f"),
    "prune_ratio": ("剪枝率", True, ".3f"),
}
HEATMAP_AXES = {"规模": "scale", "SLA": "sla_scenario", "工作流": "workflow_type"}
AXIS_ORDER = {"scale": SCALE_ORDER, "sla_scenario": SLA_ORDER, "workflow_type": WORKFLOW_ORDER}
FONT = dict(family="Microsoft YaHei, Segoe UI, sans-serif", size=13)
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#2CA02C",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def assign_study(tag: str) -> str:
    tag = str(tag)
    if tag.startswith("main_hard"):
        return "难档"
    if tag.startswith("main"):
        return "主对比"
    if tag.startswith("ablation"):
        return "消融"
    if tag.startswith("budget"):
        return "迭代预算"
    if tag == "model_ab" or tag.startswith("model_ab"):
        return "模型 A/B"
    return "其他"


def short_algo(name: str) -> str:
    prefix = "GCN-ALNS ["
    if name.startswith(prefix) and name.endswith("]"):
        return name[len(prefix):-1]
    return name


def order_names(names, preferred) -> list:
    names = list(dict.fromkeys(names))
    known = set(names)
    ordered = [n for n in preferred if n in known]
    return ordered + [n for n in names if n not in ordered]


def order_algorithms(names, study: str) -> list:
    names = list(dict.fromkeys(names))
    if study == "消融":
        by_short = {short_algo(n): n for n in names}
        ordered = [by_short[s] for s in ABLATION_ORDER if s in by_short]
        return ordered + [n for n in names if n not in ordered]
    return order_names(names, ALGO_ORDER)


def display_name(name: str, study: str) -> str:
    return short_algo(name) if study == "消融" else name


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add study / per-task cost and coerce the columns the charts read."""
    out = df.copy()
    out["study"] = out["tag"].map(assign_study)
    out["feasible"] = out["feasible"].astype(str).str.lower().isin(("true", "1", "1.0"))
    numeric = [
        "cost_total", "cost_compute", "cost_comm", "cost_reserve", "cost_recovery",
        "c_base", "c_worst", "r_max", "avg_recovery", "safety_margin",
        "first_feasible_iter", "runtime", "active_backup_ratio", "prune_ratio",
        "num_tasks", "sla_violation", "gcn_infer_time",
    ]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
    out["cost_per_task"] = out["cost_total"] / out["num_tasks"].where(out["num_tasks"] > 0)
    out.loc[out["first_feasible_iter"] < 0, "first_feasible_iter"] = np.nan
    return out


def load_runs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return prepare(pd.read_csv(path))


def slice_for(df: pd.DataFrame, metric: str, feasible_only: bool) -> pd.DataFrame:
    if feasible_only and metric in FEASIBLE_ONLY_METRICS:
        return df[df["feasible"]]
    return df


def group_stats(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean, sample std, and count per group. Empty groups are omitted."""
    if df.empty or not metrics:
        return pd.DataFrame(columns=[*group_cols])
    work = df.copy()
    for metric in metrics:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    grouped = work.groupby(group_cols, dropna=False)[metrics]
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    count = grouped.count()
    mean.columns = [f"{c}_mean" for c in mean.columns]
    std.columns = [f"{c}_std" for c in std.columns]
    count.columns = [f"{c}_count" for c in count.columns]
    return mean.join(std).join(count).reset_index()


def color_for(names: list[str], study: str) -> dict[str, str]:
    fixed = {"GCN-ALNS": "#E45756", "ALNS": "#4C78A8", FULL_VARIANT: "#E45756"}
    if study == "消融":
        fixed = {FULL_VARIANT: "#E45756"}
    colors = {}
    cursor = 0
    for name in names:
        if name in fixed:
            colors[name] = fixed[name]
            continue
        if study != "消融" and name.startswith("GCN-ALNS"):
            colors[name] = "#E45756"
        elif study != "消融" and name.startswith("ALNS"):
            colors[name] = "#4C78A8"
        else:
            colors[name] = PALETTE[cursor % len(PALETTE)]
            cursor += 1
    return colors


def _layout(fig: go.Figure, height: int, *, legend: bool = True) -> go.Figure:
    fig.update_layout(
        font=FONT,
        height=height,
        margin=dict(l=10, r=16, t=36, b=10),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0) if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    return fig


def fig_feasible(stats: pd.DataFrame, study: str) -> go.Figure | None:
    if stats.empty or "feasible_mean" not in stats.columns:
        return None
    d = stats.dropna(subset=["feasible_mean"]).sort_values("feasible_mean", ascending=True)
    if d.empty:
        return None
    labels = [display_name(a, study) for a in d["algorithm"]]
    colors = color_for(d["algorithm"].tolist(), study)
    fig = go.Figure(go.Bar(
        y=labels,
        x=d["feasible_mean"],
        orientation="h",
        marker_color=[colors[a] for a in d["algorithm"]],
        error_x=dict(
            type="data",
            array=d["feasible_std"].fillna(0),
            thickness=1.2,
            width=3,
            color="#666",
        ),
        customdata=np.stack([
            d["feasible_std"].fillna(0),
            d["feasible_count"].fillna(0),
        ], axis=1),
        hovertemplate="%{y}<br>可行率 %{x:.3f}<br>std %{customdata[0]:.3f}<br>n %{customdata[1]:.0f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.05], title="可行率")
    return _layout(fig, max(280, 28 * len(d) + 80), legend=False)


def fig_cost_stack(df: pd.DataFrame, study: str, per_task: bool) -> go.Figure | None:
    """Stacked cost. ``df`` is already restricted to the requested run scope."""
    parts = [c for c, _, _ in COST_PARTS]
    if df.empty:
        return None
    view = df.copy()
    if per_task:
        for col in parts:
            view[col] = view[col] / view["num_tasks"].where(view["num_tasks"] > 0)
    stats = group_stats(view, ["algorithm"], parts)
    if stats.empty:
        return None
    stats["total"] = stats[[f"{c}_mean" for c in parts]].sum(axis=1)
    stats = stats.sort_values("total", ascending=True)
    labels = [display_name(a, study) for a in stats["algorithm"]]
    fig = go.Figure()
    for col, label, color in COST_PARTS:
        fig.add_bar(
            name=label,
            y=labels,
            x=stats[f"{col}_mean"].fillna(0),
            orientation="h",
            marker_color=color,
            customdata=stats[f"{col}_count"].fillna(0),
            hovertemplate=f"%{{y}}<br>{label} %{{x:.2f}}<br>n %{{customdata:.0f}}<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    unit = "单任务成本" if per_task else "总成本"
    fig.update_xaxes(title=unit)
    return _layout(fig, max(280, 28 * len(stats) + 120))


def fig_heatmap(df: pd.DataFrame, metric: str, axis: str, study: str,
                feasible_only: bool) -> go.Figure | None:
    work = slice_for(df, metric, feasible_only)
    stats = group_stats(work, ["algorithm", axis], [metric])
    value = f"{metric}_mean"
    if stats.empty or value not in stats.columns:
        return None
    stats = stats.dropna(subset=[value])
    if stats.empty:
        return None
    algos = order_algorithms(stats["algorithm"].unique(), study)
    cols = order_names(stats[axis].unique(), AXIS_ORDER.get(axis, []))
    pivot = stats.pivot(index="algorithm", columns=axis, values=value)
    pivot = pivot.reindex(index=algos, columns=cols)
    counts = stats.pivot(index="algorithm", columns=axis, values=f"{metric}_count")
    counts = counts.reindex(index=algos, columns=cols)
    label, higher, fmt = METRICS[metric]
    scale = "Blues" if higher else "YlOrRd_r"
    text = pivot.map(lambda v: "" if pd.isna(v) else format(v, fmt))
    hover = counts.map(lambda v: "" if pd.isna(v) else f"n {int(v)}")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[display_name(a, study) for a in pivot.index],
        text=text.values,
        texttemplate="%{text}",
        customdata=hover.values,
        hovertemplate="%{y} · %{x}<br>" + label + " %{z}<br>%{customdata}<extra></extra>",
        colorscale=scale,
        colorbar=dict(title=label),
    ))
    fig.update_layout(title=label)
    return _layout(fig, max(320, 26 * len(pivot.index) + 100), legend=False)


def fig_scatter(df: pd.DataFrame, x_metric: str, y_metric: str, study: str,
                feasible_only: bool) -> tuple[go.Figure | None, int]:
    work = slice_for(df, y_metric, feasible_only)
    work = slice_for(work, x_metric, feasible_only)
    work = work.dropna(subset=[x_metric, y_metric])
    total = len(work)
    if work.empty:
        return None, 0
    shown = work
    if total > 6000:
        shown = work.sample(6000, random_state=0)
    colors = color_for(order_algorithms(shown["algorithm"].unique(), study), study)
    fig = go.Figure()
    for algo, part in shown.groupby("algorithm", sort=False):
        fig.add_trace(go.Scattergl(
            x=part[x_metric],
            y=part[y_metric],
            mode="markers",
            name=display_name(algo, study),
            marker=dict(size=7, color=colors.get(algo, "#4C78A8"), opacity=0.75),
            customdata=np.stack([
                part["iid"].astype(str),
                part["workflow_type"].astype(str),
                part["scale"].astype(str),
                part["sla_scenario"].astype(str),
                part["rep"].astype(str),
                part["feasible"].map(lambda v: "可行" if v else "不可行"),
            ], axis=1),
            hovertemplate=(
                "%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]} · %{customdata[3]}"
                " · rep %{customdata[4]} · %{customdata[5]}<br>"
                + METRICS[x_metric][0] + " %{x:.3g}<br>"
                + METRICS[y_metric][0] + " %{y:.3g}<extra>"
                + display_name(algo, study) + "</extra>"
            ),
        ))
    fig.update_xaxes(title=METRICS[x_metric][0])
    fig.update_yaxes(title=METRICS[y_metric][0])
    return _layout(fig, 480), total


def fig_ablation_delta(df: pd.DataFrame) -> go.Figure | None:
    if FULL_VARIANT not in set(df["algorithm"]):
        return None
    rate = group_stats(df, ["algorithm"], ["feasible"])
    cost = group_stats(df[df["feasible"]], ["algorithm"], ["cost_per_task"])
    base_rate = rate.loc[rate["algorithm"] == FULL_VARIANT, "feasible_mean"]
    base_cost = cost.loc[cost["algorithm"] == FULL_VARIANT, "cost_per_task_mean"]
    if base_rate.empty or base_cost.empty:
        return None
    merged = rate.merge(cost[["algorithm", "cost_per_task_mean"]], on="algorithm", how="left")
    merged["d_rate"] = (merged["feasible_mean"] - float(base_rate.iloc[0])) * 100
    merged["d_cost"] = (merged["cost_per_task_mean"] / float(base_cost.iloc[0]) - 1) * 100
    merged["label"] = merged["algorithm"].map(short_algo)
    merged = merged.sort_values("d_rate", ascending=True)
    fig = go.Figure()
    fig.add_bar(
        name="可行率差 (百分点)",
        y=merged["label"],
        x=merged["d_rate"],
        orientation="h",
        marker_color=["#E45756" if a == "Full" else "#4C78A8" for a in merged["label"]],
        hovertemplate="%{y}<br>可行率差 %{x:.2f} pp<extra></extra>",
    )
    fig.add_bar(
        name="单任务成本差 (%)",
        y=merged["label"],
        x=merged["d_cost"],
        orientation="h",
        marker_color="#F58518",
        hovertemplate="%{y}<br>成本差 %{x:.2f}%<extra></extra>",
    )
    fig.update_layout(barmode="group", title="相对 Full 的差值（成本只在可行解上）")
    fig.update_xaxes(title="相对 Full", zeroline=True, zerolinecolor="#999")
    return _layout(fig, max(360, 32 * len(merged) + 80))


def fig_budget(df: pd.DataFrame) -> go.Figure | None:
    work = df.copy()
    work["budget"] = pd.to_numeric(work["tag"].astype(str).str.replace("budget_", "", regex=False),
                                   errors="coerce")
    work = work.dropna(subset=["budget"])
    if work.empty:
        return None
    rate = group_stats(work, ["budget", "algorithm"], ["feasible"])
    cost = group_stats(work[work["feasible"]], ["budget", "algorithm"], ["cost_total"])
    stats = rate.merge(
        cost[["budget", "algorithm", "cost_total_mean"]],
        on=["budget", "algorithm"], how="left",
    ).sort_values("budget")
    colors = color_for(stats["algorithm"].unique().tolist(), "迭代预算")
    fig = go.Figure()
    for algo, part in stats.groupby("algorithm", sort=False):
        fig.add_trace(go.Scatter(
            x=part["budget"], y=part["feasible_mean"],
            mode="lines+markers", name=f"{algo} 可行率",
            line=dict(color=colors.get(algo, "#4C78A8")),
            customdata=part["feasible_count"],
            hovertemplate=f"{algo}<br>预算 %{{x}}<br>可行率 %{{y:.3f}}<br>n %{{customdata}}<extra></extra>",
        ))
    fig.update_xaxes(title="迭代预算", type="log")
    fig.update_yaxes(title="可行率", range=[0, 1.05])
    return _layout(fig, 380)


def fig_budget_cost(df: pd.DataFrame) -> go.Figure | None:
    work = df[df["feasible"]].copy()
    work["budget"] = pd.to_numeric(work["tag"].astype(str).str.replace("budget_", "", regex=False),
                                   errors="coerce")
    stats = group_stats(work.dropna(subset=["budget"]), ["budget", "algorithm"], ["cost_total"])
    if stats.empty:
        return None
    stats = stats.sort_values("budget")
    colors = color_for(stats["algorithm"].unique().tolist(), "迭代预算")
    fig = go.Figure()
    for algo, part in stats.groupby("algorithm", sort=False):
        fig.add_trace(go.Scatter(
            x=part["budget"], y=part["cost_total_mean"],
            mode="lines+markers", name=algo,
            line=dict(color=colors.get(algo, "#4C78A8")),
            customdata=part["cost_total_count"],
            hovertemplate=f"{algo}<br>预算 %{{x}}<br>可行成本 %{{y:.1f}}<br>n %{{customdata}}<extra></extra>",
        ))
    fig.update_xaxes(title="迭代预算", type="log")
    fig.update_yaxes(title="总成本（可行解）")
    return _layout(fig, 380)


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(np.nan, index=frame.index)


def summary_table(df: pd.DataFrame, study: str, feasible_only: bool) -> pd.DataFrame:
    rate = group_stats(df, ["algorithm"], ["feasible", "active_backup_ratio"])
    quality_src = df[df["feasible"]] if feasible_only else df
    quality = group_stats(quality_src, ["algorithm"], [
        "cost_total", "cost_per_task", "safety_margin", "first_feasible_iter",
    ])
    feas_n = df[df["feasible"]].groupby("algorithm").size().rename("n_feasible")
    table = rate.merge(quality, on="algorithm", how="left") if not quality.empty else rate
    table = table.merge(feas_n, left_on="algorithm", right_index=True, how="left")
    table["算法"] = table["algorithm"].map(lambda a: display_name(a, study))
    scope = "可行解" if feasible_only else "全部 run"
    out = pd.DataFrame({
        "算法": table["算法"],
        "n": _series(table, "feasible_count").round().astype("Int64"),
        "可行率": _series(table, "feasible_mean"),
        "可行率 std": _series(table, "feasible_std"),
        "可行 n": _series(table, "n_feasible").fillna(0).round().astype("Int64"),
        f"总成本（{scope}）": _series(table, "cost_total_mean"),
        f"单任务成本（{scope}）": _series(table, "cost_per_task_mean"),
        f"安全余量（{scope}）": _series(table, "safety_margin_mean"),
        f"首达可行迭代（{scope}）": _series(table, "first_feasible_iter_mean"),
        "主动备份比例": _series(table, "active_backup_ratio_mean"),
    })
    order = {a: i for i, a in enumerate(order_algorithms(df["algorithm"].unique(), study))}
    out["_ord"] = table["algorithm"].map(order)
    return out.sort_values(["_ord"]).drop(columns="_ord")


def _pair_metrics(df: pd.DataFrame) -> dict | None:
    if not {"ALNS", "GCN-ALNS"}.issubset(set(df["algorithm"])):
        return None
    out = {}
    for name in ("ALNS", "GCN-ALNS"):
        part = df[df["algorithm"] == name]
        feas = part[part["feasible"]]
        out[name] = {
            "rate": float(part["feasible"].mean()) if len(part) else np.nan,
            "n": int(len(part)),
            "cost_per_task": float(feas["cost_per_task"].mean()) if len(feas) else np.nan,
            "margin": float(feas["safety_margin"].mean()) if len(feas) else np.nan,
            "ttf": float(feas["first_feasible_iter"].mean()) if len(feas) else np.nan,
        }
    return out


def _synced_multiselect(label: str, options: list, key: str, signature) -> list:
    options = list(options)
    sig_key = key + "__sig"
    if st.session_state.get(sig_key) != signature:
        st.session_state[sig_key] = signature
        st.session_state[key] = options
    else:
        chosen = st.session_state.get(key, options)
        st.session_state[key] = [item for item in chosen if item in options]
    return st.multiselect(label, options, key=key)


def _show(fig: go.Figure | None, empty: str) -> None:
    if fig is None:
        st.info(empty)
        return
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="GCN-ALNS 实验浏览器", layout="wide")
    st.title("GCN-ALNS 实验浏览器")
    st.caption(
        "每条记录是一次求解。可行率用全部 run；打开「仅可行解」后，"
        "成本、makespan、恢复时延和安全余量只在 SLA 可行的 run 上取均值。"
    )

    csv_path = DEFAULT_CSV
    try:
        mtime = csv_path.stat().st_mtime
    except OSError:
        st.error(f"找不到结果表：{csv_path}")
        return
    runs = _cached_load(str(csv_path), mtime)

    with st.sidebar:
        st.header("筛选")
        study = st.selectbox("实验组", STUDIES, format_func=lambda s: s)
        st.caption(STUDY_BLURB[study])
        pool = runs[runs["study"] == study]
        if pool.empty:
            st.warning("这个实验组没有 run。")
            return
        scales = order_names(pool["scale"].dropna().unique(), SCALE_ORDER)
        slas = order_names(pool["sla_scenario"].dropna().unique(), SLA_ORDER)
        flows = order_names(pool["workflow_type"].dropna().unique(), WORKFLOW_ORDER)
        algos = order_algorithms(pool["algorithm"].dropna().unique(), study)
        picked_scales = _synced_multiselect("规模", scales, "scales", (study, tuple(scales)))
        picked_slas = _synced_multiselect("SLA", slas, "slas", (study, tuple(slas)))
        picked_flows = _synced_multiselect("工作流", flows, "flows", (study, tuple(flows)))
        picked_algos = _synced_multiselect("算法", algos, "algos", (study, tuple(algos)))
        feasible_only = st.toggle("成本 / 恢复 / makespan 仅统计可行解", value=True)
        st.caption(f"数据 {csv_path.name} · 该组 {len(pool):,} 条 run")

    chosen = pool[
        pool["scale"].isin(picked_scales)
        & pool["sla_scenario"].isin(picked_slas)
        & pool["workflow_type"].isin(picked_flows)
        & pool["algorithm"].isin(picked_algos)
    ]
    if chosen.empty:
        st.info("当前筛选没有 run。把规模、SLA、工作流或算法重新勾上即可。")
        return

    n_feas = int(chosen["feasible"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("run 数", f"{len(chosen):,}")
    c2.metric("可行率", f"{chosen['feasible'].mean():.3f}")
    c3.metric("可行 run", f"{n_feas:,}")
    c4.metric("算法数", f"{chosen['algorithm'].nunique()}")

    pair = _pair_metrics(chosen)
    if pair is not None:
        gcn, alns = pair["GCN-ALNS"], pair["ALNS"]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("可行率 GCN − ALNS", f"{(gcn['rate'] - alns['rate']) * 100:+.1f} pp")
        p2.metric("单任务成本 GCN − ALNS", _delta_num(gcn["cost_per_task"], alns["cost_per_task"], "{:+.3f}"))
        p3.metric("安全余量 GCN − ALNS", _delta_num(gcn["margin"], alns["margin"], "{:+.2f}"))
        p4.metric("首达可行迭代 GCN − ALNS", _delta_num(gcn["ttf"], alns["ttf"], "{:+.1f}"))
        st.caption("后三项只在各自的可行解上比较。可行率用两边的全部 run。")

    rate_stats = group_stats(chosen, ["algorithm"], ["feasible"])
    left, right = st.columns(2)
    with left:
        st.subheader("可行率")
        _show(fig_feasible(rate_stats, study), "没有可画的可行率。")
    with right:
        st.subheader("成本构成")
        multi_scale = chosen["scale"].nunique() > 1
        if multi_scale:
            per_task = True
            st.caption("选了多个规模，成本已除以任务数，避免 small 和 large 混进同一个均值。")
        else:
            per_task = st.checkbox("按任务归一化", value=False)
        cost_src = chosen[chosen["feasible"]] if feasible_only else chosen
        scope = "可行解" if feasible_only else "全部 run"
        st.caption(f"四项之和是总成本（不含已删除的 C_backup）。当前口径：{scope}。")
        _show(fig_cost_stack(cost_src, study, per_task), "这个口径下没有成本可画。")

    st.subheader("热力图")
    h1, h2 = st.columns([2, 1])
    metric = h1.selectbox(
        "指标", list(METRICS),
        format_func=lambda key: METRICS[key][0],
    )
    axis_label = h2.selectbox("列", list(HEATMAP_AXES), index=0)
    axis = HEATMAP_AXES[axis_label]
    mixes_scale = (
        chosen["scale"].nunique() > 1
        and axis != "scale"
        and metric in FEASIBLE_ONLY_METRICS
    )
    if mixes_scale:
        st.caption("列不是规模，而筛选里有多个规模。单元格是跨规模均值，绝对成本、时延和 makespan 会被混在一起。改看「单任务成本」，或把列换成规模。")
    _show(
        fig_heatmap(chosen, metric, axis, study, feasible_only),
        "这个指标在当前筛选下没有数值。",
    )

    st.subheader("单次 run")
    s1, s2 = st.columns(2)
    x_metric = s1.selectbox(
        "横轴",
        ["cost_total", "cost_per_task", "runtime", "c_base"],
        format_func=lambda key: METRICS[key][0],
    )
    y_metric = s2.selectbox(
        "纵轴",
        ["safety_margin", "r_max", "c_worst", "first_feasible_iter", "cost_per_task"],
        format_func=lambda key: METRICS[key][0],
    )
    scatter, scatter_n = fig_scatter(chosen, x_metric, y_metric, study, feasible_only)
    if scatter_n > 6000:
        st.caption(f"共 {scatter_n:,} 个点，图上随机显示 6,000 个。悬停仍是单次 run。")
    _show(scatter, "这个口径下没有点可画。")

    if study == "消融":
        st.subheader("相对 Full")
        _show(fig_ablation_delta(chosen), "筛选里没有 Full，无法算差值。")
    elif study == "迭代预算":
        st.subheader("预算曲线")
        b1, b2 = st.columns(2)
        with b1:
            _show(fig_budget(chosen), "没有预算曲线。")
        with b2:
            _show(fig_budget_cost(chosen), "没有可行解成本。")
    elif study == "模型 A/B":
        st.subheader("A2 开与关")
        _show_model_ab(chosen)

    st.subheader("汇总表")
    table = summary_table(chosen, study, feasible_only)
    number_cfg = {
        col: st.column_config.NumberColumn(format="%.3f" if "可行率" in col or "备份" in col else "%.2f")
        for col in table.columns
        if col not in {"算法", "n", "可行 n"}
    }
    st.dataframe(table, use_container_width=True, hide_index=True, column_config=number_cfg)
    st.download_button(
        "下载当前汇总 CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"dashboard_{study}.csv",
        mime="text/csv",
    )


def _delta_num(left: float, right: float, fmt: str) -> str:
    if np.isnan(left) or np.isnan(right):
        return "—"
    return fmt.format(left - right)


def _show_model_ab(df: pd.DataFrame) -> None:
    labels = ["GCN-ALNS (A2-off)", "GCN-ALNS (A2-on)", "ALNS"]
    present = [name for name in labels if name in set(df["algorithm"])]
    if len(present) < 2:
        st.info("筛选里少于两个模型标签，直接看上面的图即可。")
        return
    rows = []
    for name in present:
        part = df[df["algorithm"] == name]
        feas = part[part["feasible"]]
        rows.append({
            "算法": name,
            "可行率": part["feasible"].mean(),
            "单任务成本（可行）": feas["cost_per_task"].mean(),
            "安全余量（可行）": feas["safety_margin"].mean(),
            "主动备份比例": part["active_backup_ratio"].mean(),
            "n": len(part),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def _cached_load(path_str: str, mtime: float) -> pd.DataFrame:
    del mtime  # part of the cache key so a rewritten CSV is reloaded
    return load_runs(Path(path_str))


if __name__ == "__main__":
    main()
