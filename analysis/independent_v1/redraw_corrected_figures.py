"""Redraw audited main-study figures without changing the frozen experiment.

python analysis/independent_v1/redraw_corrected_figures.py
Outputs PNG, SVG, plotted CSV values, and provenance under figures_corrected/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/independent_v1"
OUT = SOURCE / "test/figures_corrected"
SCALES = ("small", "medium", "large")
ALGORITHMS = ("HEFT", "PEFT", "No-FT", "Reactive-Only", "Full-Proactive",
              "Critical-Path Backup", "Criticality-Based Hybrid", "Reliability-Aware",
              "RECR-D", "HFWPF", "GA-Placement", "RA-MOMA", "ALNS", "GCN-ALNS")
SHORT = dict(zip(ALGORITHMS, ("HEFT", "PEFT", "No-FT", "Reactive-only", "Full-proactive",
                             "Critical-path backup", "Criticality hybrid", "Reliability-aware",
                             "RECR-D adaptation", "HFWPF adaptation", "GA-Placement",
                             "RA-MOMA adaptation", "ALNS", "GCN-ALNS")))
SEARCH = ("ALNS", "GCN-ALNS")
COMPONENTS = ("cost_compute", "cost_comm", "cost_reserve", "cost_recovery")
COLORS = ("#597387", "#0072B2", "#E69F00", "#009E73")
SOURCE_NOTE = "Source: independent_v1/main_* only; 90 held-out instances (45 small / 30 medium / 15 large)."
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "savefig.facecolor": "white"})
FIGURES = {}
INPUT_HASHES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    INPUT_HASHES[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else np.nan
    if len(values) < 2 or np.allclose(values, values[0], rtol=0, atol=1e-12):
        return mean, np.nan, np.nan, len(values)
    draws = np.random.default_rng(2026).choice(values, (2000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return mean, float(low), float(high), len(values)


def stratified_interval(frame, field):
    """Fix the 45/30/15 instance allocation while bootstrapping each scale."""
    rng = np.random.default_rng(2026)
    draws = np.zeros(2000)
    for scale in SCALES:
        values = frame.loc[frame.scale == scale, field].to_numpy(float)
        assert np.isfinite(values).all()
        draws += rng.choice(values, (2000, len(values)), replace=True).mean(axis=1) * len(values) / len(frame)
    mean = float(frame[field].mean())
    if np.allclose(frame[field], frame[field].iloc[0], rtol=0, atol=1e-12):
        return mean, np.nan, np.nan, len(frame)
    low, high = np.quantile(draws, [.025, .975])
    return mean, float(low), float(high), len(frame)


def finish(fig, name, caption, correction):
    fig.text(.5, .012, "\n".join(textwrap.wrap(caption, 155)), ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT / (name + ".png"), dpi=300)
    fig.savefig(OUT / (name + ".svg"))
    plt.close(fig)
    FIGURES[name] = {"caption": caption, "correction": correction, "formats": ["png", "svg"]}


def color_for(algorithm):
    return "#0072B2" if algorithm == "GCN-ALNS" else "#5F6368" if algorithm == "ALNS" else "#B9C5CD"


def one_bar(instance, field, name, title, xlabel, percent=True):
    rows = []
    for algorithm in ALGORITHMS:
        sub = instance[instance.algorithm == algorithm]
        mean, low, high, n = stratified_interval(sub, field)
        scale = 100 if percent else 1
        rows.append({"algorithm": algorithm, "metric": field, "mean": mean * scale,
                     "ci95_low": low * scale, "ci95_high": high * scale, "n_instances": n})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    fig.subplots_adjust(left=.22, right=.91, top=.90, bottom=.16)
    yy = np.arange(len(data))
    ax.barh(yy, data["mean"], color=[color_for(a) for a in ALGORITHMS], height=.65)
    for y, r in data.iterrows():
        if np.isfinite(r.ci95_low):
            ax.errorbar(r["mean"], y, xerr=[[max(0, r["mean"] - r.ci95_low)], [max(0, r.ci95_high - r["mean"])]],
                        fmt="none", ecolor="0.15", capsize=3, linewidth=1)
        endpoint = max(r["mean"], r.ci95_high) if np.isfinite(r.ci95_high) else r["mean"]
        ax.text(endpoint + 1.2, y, f"{r['mean']:.2f}", va="center", fontsize=9)
    ax.set_yticks(yy, [SHORT[a] for a in ALGORITHMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 108 if percent else max(data["mean"].max() * 1.18, .01))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Main-study algorithm / adapted configuration")
    ax.set_title(title, pad=16)
    ax.grid(axis="x", color="0.91")
    ax.set_axisbelow(True)
    data.to_csv(OUT / (name + ".csv"), index=False)
    caption = SOURCE_NOTE + " Seeds averaged within instance, then all 90 instances equally weighted. "
    caption += "95% bootstrap intervals preserve scale allocation; intervals omitted for constant samples. "
    caption += "All runs included; no filtering on feasibility."
    finish(fig, name, caption, "Instance macro mean replaces unequal seed/run weighting; fixed 45/30/15 scale allocation.")


def panels(instance, field, name, title, xlabel, algorithms=ALGORITHMS, percent=False, limits=None):
    rows = []
    compact = len(algorithms) <= 2
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1 if compact else 7.0), sharey=True)
    fig.subplots_adjust(left=.17, right=.97, top=.80 if compact else .86,
                        bottom=.30 if compact else .17, wspace=.18)
    yy = np.arange(len(algorithms))
    for ax, scale in zip(axes, SCALES):
        sub = instance[instance.scale == scale]
        series = []
        for algorithm in algorithms:
            values = sub.loc[sub.algorithm == algorithm, field]
            mean, low, high, n = interval(values)
            factor = 100 if percent else 1
            mean, low, high = mean * factor, low * factor, high * factor
            rows.append({"algorithm": algorithm, "scale": scale, "metric": field, "mean": mean,
                         "ci95_low": low, "ci95_high": high, "n_instances": n})
            series.append((mean, low, high, n))
        upper = max((high if np.isfinite(high) else mean for mean, low, high, n in series if n), default=1)
        span = limits[1] if limits else max(upper * 1.22, .01)
        for y, (algorithm, (mean, low, high, n)) in enumerate(zip(algorithms, series)):
            if not n:
                ax.text(span * .02, y, "N/F", va="center", fontsize=8, color="0.4")
                continue
            ax.barh(y, mean, color=color_for(algorithm), height=.36 if compact else .62)
            if np.isfinite(low):
                ax.errorbar(mean, y, xerr=[[max(0, mean-low)], [max(0, high-mean)]], fmt="none",
                            ecolor="0.15", capsize=3, linewidth=.8)
            endpoint = max(mean, high) if np.isfinite(high) else mean
            label = f"{mean:.2f}" if percent else f"n={n}"
            ax.text(endpoint + span * .015, y, label, va="center", fontsize=8)
        ax.set_xlim(limits if limits else (0, span))
        ax.set_yticks(yy, [SHORT[a] for a in algorithms])
        ax.set_title(scale.capitalize() + f" ({int(sub.iid.nunique())} instances)")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="0.91")
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    if compact:
        axes[0].set_ylim(len(algorithms) - .3, -.7)
    axes[0].set_ylabel("Main-study algorithm / adapted configuration")
    fig.suptitle(title, y=.96, fontsize=13)
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)
    conditional = field.startswith("feas_")
    caption = SOURCE_NOTE + " Seeds averaged within instance, then instances equally weighted within each scale. "
    caption += "95% instance-bootstrap intervals; omitted for n<2 or constant samples. "
    caption += ("Feasible runs only; n is the number of instances with feasible output; N/F denotes no feasible output, not a zero metric value. "
                if conditional else "All runs included. ")
    if field == "feas_convergence_iter":
        caption += "Recorded last feasible-register update index (zero-based); this is not a convergence guarantee."
    if field == "gcn_infer_time":
        caption += "Only the model-using solver is shown; training time excluded."
    finish(fig, name, caption, "Main tags only; scale-separated instance means; explicit conditioning and valid sample counts.")


def cost_breakdown(instance):
    name = "fig_cost_breakdown"
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 7.5), sharey=True)
    fig.subplots_adjust(left=.17, right=.97, top=.83, bottom=.17, wspace=.16)
    yy = np.arange(len(ALGORITHMS))
    for ax, scale in zip(axes, SCALES):
        sub = instance[instance.scale == scale]
        totals = []
        for algorithm in ALGORITHMS:
            group = sub[sub.algorithm == algorithm]
            means = group[["feas_" + c for c in COMPONENTS]].mean()
            total = float(group.feas_cost_total.mean())
            n = int(group.feas_cost_total.notna().sum())
            values = means.to_numpy(float)
            if n:
                assert np.isclose(values.sum(), total, rtol=1e-10, atol=1e-8)
            row = {"algorithm": algorithm, "scale": scale, "n_successful_instances": n,
                   "cost_total": total, **dict(zip(COMPONENTS, values))}
            rows.append(row)
            totals.append((values, total, n))
        xmax = max(t for v, t, n in totals if n) * 1.23
        for y, (values, total, n) in enumerate(totals):
            if not n:
                ax.text(.02 * xmax, y, "N/F", va="center", fontsize=8, color="0.4")
                continue
            left = 0
            for value, color in zip(values, COLORS):
                ax.barh(y, value, left=left, color=color, height=.62)
                left += value
            ax.text(total + xmax * .015, y, f"n={n}", va="center", fontsize=8)
        ax.set_yticks(yy, [SHORT[a] for a in ALGORITHMS])
        ax.set_xlim(0, xmax)
        ax.set_title(scale.capitalize())
        ax.set_xlabel("Conditional mean cost (model units)")
        ax.grid(axis="x", color="0.91")
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Main-study algorithm / adapted configuration")
    fig.suptitle("Deployment cost components by scale (feasible outputs only)", y=.97, fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, ["Compute (all copies)", "Communication", "Passive reserve", "Recovery-delay proxy"],
               loc="upper center", bbox_to_anchor=(.57, .925), ncol=4, frameon=False, fontsize=10)
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)
    finish(fig, name, SOURCE_NOTE + " Feasible seed means per instance, then successful instances equally weighted; n is successful instances. "
           "Each stack sums to reported total cost. Backup is already inside compute and is not stacked again. N/F: no feasible output. Successful subsets differ across algorithms.",
           "Remove duplicated backup charge; separate problem scales and feasible-only instance-mean components.")


def runtime_plot(instance):
    name = "fig_scalability_runtime"
    fig, ax = plt.subplots(figsize=(11.7, 7.7))
    fig.subplots_adjust(left=.11, right=.97, top=.91, bottom=.36)
    tasks = [50, 150, 500]
    palette = plt.get_cmap("tab20").colors
    rows = []
    for index, algorithm in enumerate(ALGORITHMS):
        means = []
        for scale in SCALES:
            mean = float(instance.loc[(instance.algorithm == algorithm) & (instance.scale == scale), "runtime"].mean())
            assert mean > 0
            means.append(mean)
            rows.append({"algorithm": algorithm, "scale": scale, "num_tasks": tasks[SCALES.index(scale)], "runtime_mean_s": mean})
        focus = algorithm in SEARCH
        color = "#0072B2" if algorithm == "GCN-ALNS" else "#D55E00" if algorithm == "ALNS" else palette[index]
        ax.plot(tasks, means, label=SHORT[algorithm], color=color, marker="o" if focus else ("s", "^", "v", "D")[index % 4],
                linewidth=2.5 if focus else 1.1, markersize=6 if focus else 4,
                linestyle="-" if focus else ("--", "-.", ":")[index % 3], zorder=4 if focus else 2)
    ax.set_yscale("log")
    ax.set_xticks(tasks)
    ax.set_xlabel("Tasks / resource nodes per instance (count; linear axis)")
    ax.set_ylabel("Full solver runtime (seconds; logarithmic scale)")
    ax.set_title("Main-study runtime versus problem size")
    ax.grid(axis="y", color="0.9")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=4, frameon=False, fontsize=8.5)
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)
    finish(fig, name, SOURCE_NOTE + " Seed means per instance, then all instances at each scale equally weighted. "
           "ALNS limits: 1000 / 1000 / 800 iterations; stage worker counts: 8 / 8 / 12. Training excluded. "
           "Solver families use their registered search budgets; curves are not matched-wall-clock comparisons.",
           "Filter main_* before aggregation; remove budget/model-diagnostic runs; retain all 14 main algorithms.")


def scenario_plot(main):
    name = "fig_sla_by_scenario"
    cells = [(scale, scenario) for scale in SCALES for scenario in ("tight", "medium", "loose")]
    values = np.empty((len(ALGORITHMS), len(cells)))
    rows = []
    for y, algorithm in enumerate(ALGORITHMS):
        for x, (scale, scenario) in enumerate(cells):
            data = main[(main.algorithm == algorithm) & (main.scale == scale) & (main.sla_scenario == scenario)]
            per_instance = data.groupby("iid").sla_violation.mean()
            rate = float(100 * per_instance.mean())
            values[y, x] = rate
            rows.append({"algorithm": algorithm, "scale": scale, "scenario": scenario,
                         "n_instances": len(per_instance), "violation_percent": rate})
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    fig.subplots_adjust(left=.21, right=.91, top=.88, bottom=.19)
    image = ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(9), [s.capitalize() + "\n" + q for s, q in cells])
    ax.set_yticks(range(14), [SHORT[a] for a in ALGORITHMS])
    for y in range(14):
        for x in range(9):
            ax.text(x, y, f"{values[y,x]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if values[y,x] >= 55 else "black")
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", linewidth=2)
    ax.set_xlabel("Problem scale and SLA scenario (categorical)")
    ax.set_ylabel("Main-study algorithm / adapted configuration")
    ax.set_title("Recorded SLA violation rate by scale and SLA scenario")
    cb = fig.colorbar(image, ax=ax, fraction=.04, pad=.025)
    cb.set_label("Violation rate (instance macro mean, %)")
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False)
    finish(fig, name, SOURCE_NOTE + " Each cell first averages seeds within instance, then instances. "
           "Each scenario contains 15 / 10 / 5 instances at small / medium / large scale. The recorded violation flag equals 1 minus feasibility.",
           "Do not pool scales with unequal repeats within SLA groups; show all nine scale-by-scenario cells.")


def main():
    plan = read(SOURCE / "pipeline_plan.json")
    selection = read(SOURCE / "model_selection.json")
    report = read(SOURCE / "summary.json")
    frames = []
    for scale in SCALES:
        frame = pd.DataFrame(read(SOURCE / "test/raw_json" / ("main_" + scale + ".json")))
        assert len(frame) == plan["expected_test_records"]["main"][scale]
        assert set(frame.tag) == {"main_" + scale}
        assert set(frame.algorithm) == set(ALGORITHMS)
        assert (frame.evaluation_split == "test").all()
        assert (frame.protocol_sha256 == plan["protocol_sha256"]).all()
        assert (frame.code_sha256 == plan["code"]["sha256"]).all()
        assert (frame.loc[frame.algorithm == "GCN-ALNS", "model_sha256"] == selection["model_sha256"]).all()
        assert not frame.duplicated(["algorithm", "iid", "rep"]).any()
        assert np.allclose(frame[list(COMPONENTS)].sum(axis=1), frame.cost_total, rtol=1e-10, atol=1e-8)
        for algorithm, rows in frame.groupby("algorithm"):
            assert np.isclose(rows.groupby("iid").feasible.mean().mean(), report["main"][scale + "/" + algorithm]["feasible_macro"])
        frames.append(frame)
    main_rows = pd.concat(frames, ignore_index=True)
    assert len(main_rows) == 4200 and main_rows.iid.nunique() == 90
    assert (main_rows.sla_violation == 1 - main_rows.feasible.astype(int)).all()
    records = []
    for (algorithm, scale, iid), rows in main_rows.groupby(["algorithm", "scale", "iid"]):
        feasible = rows[rows.feasible].copy()
        row = {"algorithm": algorithm, "scale": scale, "iid": iid, "n_runs": len(rows), "n_feasible_runs": len(feasible)}
        for field in ("feasible", "sla_violation", "active_backup_ratio", "prune_ratio", "gcn_infer_time", "runtime"):
            row[field] = float(rows[field].mean())
        for field in (*COMPONENTS, "cost_total", "safety_margin", "convergence_iter"):
            values = feasible[field].replace([np.inf, -np.inf], np.nan)
            assert values.notna().all()
            row["feas_" + field] = float(values.mean()) if len(values) else np.nan
        row["feas_cost_per_task"] = float((feasible.cost_total / feasible.num_tasks).mean()) if len(feasible) else np.nan
        records.append(row)
    instance = pd.DataFrame(records)
    assert len(instance) == 1260 and (instance.groupby("algorithm").iid.nunique() == 90).all()
    OUT.mkdir(parents=True, exist_ok=True)
    instance.to_csv(OUT / "instance_plot_data.csv", index=False)
    cost_breakdown(instance)
    panels(instance, "feas_cost_per_task", "fig_cost_per_task", "Cost per task by scale (feasible outputs only)", "Conditional mean cost per task (model units)")
    runtime_plot(instance)
    one_bar(instance, "feasible", "fig_feasible_ratio", "SLA-feasible rate: equal weight per held-out instance", "SLA-feasible rate (instance macro mean, %)")
    panels(instance, "feasible", "fig_feasible_ratio_by_scale", "SLA-feasible rate by problem scale", "SLA-feasible rate (instance macro mean, %)", percent=True, limits=(0, 112))
    one_bar(instance, "sla_violation", "fig_sla_violation", "Recorded SLA violation rate: equal weight per instance", "Violation rate (instance macro mean, %)")
    one_bar(instance, "active_backup_ratio", "fig_backup_ratio", "Active-task fraction over all returned deployments", "Active-task fraction (instance macro mean, %)")
    panels(instance, "prune_ratio", "fig_prune_ratio", "Candidate pruning fraction for the two ALNS solvers", "Pruned candidate fraction (instance macro mean, %)", algorithms=SEARCH, percent=True, limits=(0, 112))
    panels(instance, "gcn_infer_time", "fig_gcn_infer", "One-time learned-score inference time by scale", "Mean inference time (seconds)", algorithms=("GCN-ALNS",))
    panels(instance, "feas_convergence_iter", "fig_convergence", "Last recorded feasible-register update (feasible outputs only)", "Mean recorded update index (iterations; zero-based)", algorithms=SEARCH)
    panels(instance, "feas_safety_margin", "fig_safety_margin", "Recovery safety margin by scale (feasible outputs only)", "Conditional mean recovery safety margin (seconds)")
    scenario_plot(main_rows)
    frozen = dict(plan["code"]["files"])
    frozen["scripts/rerun_independent_v1.py"] = plan["driver_sha256"]
    frozen["tools/report_independent_v1.py"] = plan["report_sha256"]
    assert all(sha(ROOT / p) == h for p, h in frozen.items())
    assert all(sha(ROOT / p) == h for p, h in INPUT_HASHES.items())
    manifest = {"schema_version": 1, "date": "2026-11-05", "main_records": 4200,
                "instances": 90, "algorithms": 14, "protocol_sha256": plan["protocol_sha256"],
                "experiment_code_sha256": plan["code"]["sha256"], "selected_model_sha256": selection["model_sha256"],
                "source_files_sha256": INPUT_HASHES, "render_script_sha256": sha(Path(__file__)),
                "protected_experiment_files_verified": len(frozen), "figures": FIGURES,
                "aggregation": "Average seeds within iid, then equally weight instances. Conditional metrics use instances having feasible runs; subsets differ by algorithm.",
                "intervals": "2000 instance bootstrap samples, seed 2026; all-scale intervals stratified at fixed 45/30/15 allocation; omit for n<2 or constant values."}
    notes = """# 修正后的主实验图表

日期：2026-11-05。12张英文图，每张提供300 dpi PNG及可编辑SVG。
只使用main_small/main_medium/main_large的4200条记录、90个实例、14个算法标签。
不包含预算、消融或模型诊断。原图与原始实验记录保留。

## 口径

- 随机重复先在实例内平均，再对实例等权平均。总体实例权重为45/90、30/90、15/90。
- 成本和余量仅使用可行运行：先取实例内可行运行均值，再平均有可行输出的实例。
  各算法成功子集不同，柱高不能替代配对成本效应；图上n为成功实例数。
- N/F表示没有可行输出，不是成本为零。n<2或样本恒定时不画置信区间。
- 区间为2000次实例bootstrap；总体区间在固定45/30/15分层配置内重采样。
  图上区间不是种子标准差，也不能代替配对显著性检验。
- compute已包含所有副本；backup是说明字段，不再单独叠加。所有4200条记录及绘图堆叠均核对四项和等于总成本。
- 各规模运行上限不同（ALNS为1000/1000/800），不同算法保留自身注册预算。
- 主要图适合IEEE双栏通栏展示；过多算法时不建议直接缩成单栏。

## 图表

"""
    for name, info in FIGURES.items():
        notes += f"- [{name}]({name}.png)：{info['correction']} 对应[{name}.svg]({name}.svg)。\n"
    notes += """
## 使用

批量预览见preview_contact_sheet.png。逐点数值见每张图同名CSV，实例级绘图数据见instance_plot_data.csv，指纹见figure_manifest.json。
复现命令：`python analysis/independent_v1/redraw_corrected_figures.py`。
fig_convergence已准确标注为“最后一次记录的可行方案寄存器更新时间”，不代表收敛保证或首达可行时间。
这些图不修复两组all-active/all-passive消融的策略约束问题；本目录没有用它们推导纯策略优越性。
"""
    (OUT / "README.md").write_text(notes, encoding="utf-8")
    thumbs = []
    for name in FIGURES:
        with Image.open(OUT / (name + ".png")) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((650, 430))
            tile = Image.new("RGB", (680, 470), "white")
            tile.paste(thumb, ((680-thumb.width)//2, 28+(430-thumb.height)//2))
            ImageDraw.Draw(tile).text((12, 8), name, fill="black")
            thumbs.append(tile)
    sheet = Image.new("RGB", (680*3, 470*4), "#dddddd")
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % 3)*680, (index//3)*470))
    sheet.save(OUT / "preview_contact_sheet.png")
    manifest["output_files_sha256"] = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "figure_manifest.json"}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    archive = OUT.parent / "figures_corrected.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
        bundle.write(Path(__file__), "redraw_corrected_figures.py")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print("REDRAW COMPLETE:", len(FIGURES), "figures; PNG + SVG + CSV + provenance")
    for algorithm in SEARCH:
        print(algorithm, "overall macro feasibility %", instance.loc[instance.algorithm == algorithm, "feasible"].mean() * 100)
    print("Source hashes unchanged; protected experiment files verified:", len(frozen))
    print("Output:", OUT)
    print("Archive:", archive)


if __name__ == "__main__":
    main()
