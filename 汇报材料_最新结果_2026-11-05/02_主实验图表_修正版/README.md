# 修正后的主实验图表

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

- [fig_cost_breakdown](fig_cost_breakdown.png)：Remove duplicated backup charge; separate problem scales and feasible-only instance-mean components. 对应[fig_cost_breakdown.svg](fig_cost_breakdown.svg)。
- [fig_cost_per_task](fig_cost_per_task.png)：Main tags only; scale-separated instance means; explicit conditioning and valid sample counts. 对应[fig_cost_per_task.svg](fig_cost_per_task.svg)。
- [fig_scalability_runtime](fig_scalability_runtime.png)：Filter main_* before aggregation; remove budget/model-diagnostic runs; retain all 14 main algorithms. 对应[fig_scalability_runtime.svg](fig_scalability_runtime.svg)。
- [fig_feasible_ratio](fig_feasible_ratio.png)：Instance macro mean replaces unequal seed/run weighting; fixed 45/30/15 scale allocation. 对应[fig_feasible_ratio.svg](fig_feasible_ratio.svg)。
- [fig_feasible_ratio_by_scale](fig_feasible_ratio_by_scale.png)：Main tags only; scale-separated instance means; explicit conditioning and valid sample counts. 对应[fig_feasible_ratio_by_scale.svg](fig_feasible_ratio_by_scale.svg)。
- [fig_sla_violation](fig_sla_violation.png)：Instance macro mean replaces unequal seed/run weighting; fixed 45/30/15 scale allocation. 对应[fig_sla_violation.svg](fig_sla_violation.svg)。
- [fig_backup_ratio](fig_backup_ratio.png)：Instance macro mean replaces unequal seed/run weighting; fixed 45/30/15 scale allocation. 对应[fig_backup_ratio.svg](fig_backup_ratio.svg)。
- [fig_prune_ratio](fig_prune_ratio.png)：Main tags only; scale-separated instance means; explicit conditioning and valid sample counts. 对应[fig_prune_ratio.svg](fig_prune_ratio.svg)。
- [fig_gcn_infer](fig_gcn_infer.png)：Main tags only; scale-separated instance means; explicit conditioning and valid sample counts. 对应[fig_gcn_infer.svg](fig_gcn_infer.svg)。
- [fig_convergence](fig_convergence.png)：Main tags only; scale-separated instance means; explicit conditioning and valid sample counts. 对应[fig_convergence.svg](fig_convergence.svg)。
- [fig_safety_margin](fig_safety_margin.png)：Main tags only; scale-separated instance means; explicit conditioning and valid sample counts. 对应[fig_safety_margin.svg](fig_safety_margin.svg)。
- [fig_sla_by_scenario](fig_sla_by_scenario.png)：Do not pool scales with unequal repeats within SLA groups; show all nine scale-by-scenario cells. 对应[fig_sla_by_scenario.svg](fig_sla_by_scenario.svg)。

## 使用

批量预览见preview_contact_sheet.png。逐点数值见每张图同名CSV，实例级绘图数据见instance_plot_data.csv，指纹见figure_manifest.json。
复现命令：`python analysis/independent_v1/redraw_corrected_figures.py`。
fig_convergence已准确标注为“最后一次记录的可行方案寄存器更新时间”，不代表收敛保证或首达可行时间。
这些图不修复两组all-active/all-passive消融的策略约束问题；本目录没有用它们推导纯策略优越性。
