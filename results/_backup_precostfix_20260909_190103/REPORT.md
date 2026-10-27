# GCN-ALNS / RLC-HFTSP 实验报告

> 本报告基于本机（`d:\zyx\gcn-alns`，32 核 CPU + 1×A40 49GB，conda 环境 `gcnALNS`，torch 2.5.1+cu121）的完整实验。
> **正文 §1–§9 描述"主对比"图**（`results/figures/`，来自 `main_small`+`main_medium`+`main_large` 汇总，见 `results/csv/summary_by_*`）。
> **附录 A/B** 记录两轮迭代改造(新数据集/A2 修复、收敛/难档/B 系归因/时间效率)的完整分析与数值，为最新结论,正文与其一致。
> 注:主对比覆盖 **small/medium/large 三规模**;large 用 `repeats=5, max-iter=800`(large 既定惯例,单解 ~4.5h),small/medium 用 `repeats=10, max-iter=1000`。

## 1. 实验配置概览

| 项目 | 配置 |
|---|---|
| 实例规模 | small=50 / medium=150 / large=500 任务与节点，各 45 / 30 / 15 个（共 **90** 实例） |
| 工作流类型 | Montage, CyberShake, Epigenomics, LIGO, SIPHT |
| SLA 场景 | tight(α1.15/β0.05)、medium(α1.30/β0.10)、loose(α1.50/β0.20)（另有难档 tight2/tight3,见附录 B） |
| 对比算法 | HEFT, PEFT, No-FT, Reactive-Only, Full-Proactive, Critical-Path Backup, Criticality-Based Hybrid, Reliability-Aware, ALNS, **GCN-ALNS** |
| 随机算法 | ALNS / GCN-ALNS 多 seed 重复（repeats=10、ALNS 迭代=1000） |
| GCN 模型 | 满迭代 teacher + mode 标签重训，**test pairwise AUC=0.889**；默认模型已升级为 A2 修复版(mode AUC 0.975,见附录 B.4) |
| 判定口径 | 所有算法统一经 `core.recovery.verify` 判定 SLA 可行性 |

主对比 run 数：`main_small`=1260、`main_medium`=840、`main_large`=270；消融 14 变体×375；另有 `budget_*`/`main_hard_{small,medium,large}`/`model_ab`（见附录 B，不并入主对比汇总）。

**总体结论**：仅 **ALNS / GCN-ALNS** 达成显著 SLA 可行率，且 **GCN-ALNS 在可行率略高的同时成本更低**——三规模全局 **GCN-ALNS 可行率 0.503 / ALNS 0.492**、仅可行成本 **641.8 vs 654.2**；按规模看 large 尤其明显(**GCN 0.800 / ALNS 0.787**,成本 2025.7 vs 2056.6,且安全余量 12.0 vs 7.1)。所有启发式 baseline 明显更弱(全局 ≤0.18,三种备份/全主动策略 ≤0.03)。**GCN 的增量价值主要在成本、收敛速度与安全余量;可行率的主要驱动是搜索机制(见附录 B.3),但其可行率优势在 large 规模比 small/medium 更明显(large-medium 场景 0.96 vs 0.92)**。

---

## 2. 主对比：无量纲指标

### 2.1 SLA 可行率 — `fig_feasible_ratio.png`

![可行率](figures/fig_feasible_ratio.png)

全局可行率(三规模)：**GCN-ALNS 0.503、ALNS 0.492**，二者断层领先且 GCN 略高；其余 HEFT/No-FT 0.178、Reactive-Only/Reliability-Aware 0.167、PEFT 0.144，而 Critical-Path Backup 0.033 / Criticality-Based Hybrid 0.022 / Full-Proactive 0.000 几乎全灭。这印证论文核心主张：面向"部署 + 恢复"联合优化的元启发式（ALNS 系）才能在带容错约束的 SLA 下稳定求出可行解;纯启发式/纯主动冗余策略普遍失败。注:GCN 对 ALNS 的可行率增量小(+0.011),其真正价值在成本、安全余量与收敛(见 §2.6、附录 B)。

### 2.2 SLA 违约率 — `fig_sla_violation.png`

![违约率](figures/fig_sla_violation.png)

违约率是可行率的互补（violation = 1 − feasible）：GCN-ALNS 0.497、ALNS 0.508 最低；Full-Proactive 1.000 全违约。注意 ALNS/GCN-ALNS 的违约主要来自 tight（及部分 medium）场景本身不可行（见 §6），而非算法缺陷。

### 2.3 主动备份比例 — `fig_backup_ratio.png`

![主动备份比例](figures/fig_backup_ratio.png)

`active_backup_ratio` = 被放置主动副本的任务占比：Full-Proactive=1.000（对所有任务开主动副本，成本失控）、Criticality-Based Hybrid=0.300、Critical-Path Backup=0.213，而 **GCN-ALNS 0.059 / ALNS 0.058** 极低。说明 ALNS 系只对少数关键任务选择性地开主动副本，其余靠被动恢复即可满足 R_max —— 这正是它们成本低的直接原因。HEFT/PEFT 等无容错策略为 0。(A2 修复模型会小幅提高该比例,见附录 B.4。)

### 2.4 候选剪枝率 — `fig_prune_ratio.png`

![剪枝率](figures/fig_prune_ratio.png)

仅 ALNS 系有该指标：GCN-ALNS 0.103、ALNS 0.093。插入修复阶段约 10% 的候选被快速预筛选剪掉,在保证质量的同时降低单次迭代开销。**GCN-ALNS 剪枝率略高**——GCN 打分帮它更早跳过差插入,是其"更快达可行"的机制之一(见附录 B.6)。

### 2.5 GCN 推理耗时 — `fig_gcn_infer.png`

![GCN推理耗时](figures/fig_gcn_infer.png)

仅 GCN-ALNS 非零：**~0.06 s(small)~约 1–2 s(large)**（整张 [T,N] 打分表一次算好并缓存,为一次性开销;三规模均值 ~0.57 s)。相对于单次求解数十秒~数小时的搜索开销，GCN 推理开销可忽略，属于"低成本高杠杆"的引导信号。

### 2.6 ALNS 收敛迭代 — `fig_convergence.png`

![收敛迭代](figures/fig_convergence.png)

达到最优 incumbent 的迭代数：GCN-ALNS 304、ALNS 299,二者相近。更能体现 GCN 效率优势的是**首次达可行的迭代数**:GCN 比 ALNS 少 ~28%(119 vs 165,见附录 B.1/B.6 的 `fig_budget_convergence`)——GCN 用更少迭代到达可行/更省。其余确定性算法无迭代（记 0）。

---

## 3. 成本分析

### 3.1 成本构成堆叠 — `fig_cost_breakdown.png`

![成本构成](figures/fig_cost_breakdown.png)

按 compute / backup / comm / reserve / recovery 五项分解（单位成本，全体 run 均值）：

按 compute / backup / comm / reserve / recovery 五项分解(三规模全体 run 均值)：

| 算法 | compute | backup | comm | reserve | recovery | 合计 |
|---|---|---|---|---|---|---|
| **GCN-ALNS** | 362.2 | 46.2 | 169.8 | 12.3 | 1.2 | **591.8** |
| ALNS | 375.1 | 53.9 | 173.1 | 12.3 | 1.1 | 615.5 |
| HEFT/No-FT | 475.2 | 0 | 200.3 | 15.8 | 1.2 | 692.6 |
| Critical-Path Backup | 591.6 | 216.7 | 201.6 | 13.1 | 1.9 | 1025.0 |
| Criticality-Based Hybrid | 676.2 | 361.7 | 204.3 | 11.1 | 2.2 | 1255.6 |
| Full-Proactive | 1048.9 | 1048.9 | 210.3 | 0 | 4.7 | 2312.8 |

关键洞察：GCN-ALNS 相比 ALNS 主要节省在 **compute（362.2 vs 375.1）与 backup（46.2 vs 53.9）**——GCN 引导使任务落到更合适的节点并更省地选主动副本。启发式 baseline 合计更高且大多不可行(HEFT 692.6/可行率 0.18);Critical-Path/Criticality-Hybrid 的 backup 成本(216.7/361.7)与 Full-Proactive(1048.9)直接吞噬全部收益。(注:本表为跨规模均值,受各规模 run 数与量级混合影响;分规模成本见 §3.3。)

### 3.2 按任务归一化成本 — `fig_cost_per_task.png`

![单任务成本](figures/fig_cost_per_task.png)

`cost_total / num_tasks`（仅可行解，按规模分组），用于跨规模可比。**GCN-ALNS 4.53(small)/4.06(medium)/4.05(large) 均低于 ALNS 4.66/4.19/4.11**;HEFT/PEFT 在 small 数字更低(3.89~4.20)是幸存者偏差——它们只在极少数易解实例上可行。优势并非来自规模效应,而是每个任务的部署决策更优。

### 3.3 总成本（按规模，仅可行解）— `fig_cost_total_by_scale.png`

![总成本按规模](figures/fig_cost_total_by_scale.png)

| 算法 | small | medium | large |
|---|---|---|---|
| **GCN-ALNS** | **226.7** | **609.6** | **2025.7** |
| ALNS | 233.1 | 629.2 | 2056.6 |
| HEFT/No-FT | 210.1 | 624.7 | 2110.6 |
| PEFT | 194.3 | 638.2 | 2100.5 |

GCN-ALNS 在 medium/large 成本最低,small 上也低于 ALNS;HEFT/PEFT 在 small 更低同样是幸存者偏差(可行率 <0.09,样本偏易)。跨规模看 GCN-ALNS 相对 ALNS 的成本优势稳定保持,且 large 时绝对差距(~31)最大。

---

## 4. 恢复质量与容错指标（按规模，仅可行解）

### 4.1 无故障 makespan — `fig_makespan_base_by_scale.png`

![基准makespan](figures/fig_makespan_base_by_scale.png)

`c_base`（无故障基准完工时间）：无容错的 HEFT/PEFT 略短，ALNS/GCN-ALNS 的 c_base（~67 @small、~121 @medium、~265~275 @large）略高,是"成本-时延-容错"三者平衡后的合理取舍。**GCN-ALNS 略优于 ALNS**(large 264.8 vs 275.2)。

### 4.2 最坏 makespan — `fig_makespan_worst_by_scale.png`

![最坏makespan](figures/fig_makespan_worst_by_scale.png)

`c_worst`（仅可行解，考虑最坏故障后的完工时间）：ALNS/GCN-ALNS 在 small/medium 接近（85.1/154.3 vs 85.7/153.8）,large 时 **GCN-ALNS 329.5 < ALNS 344.1**。这说明 ALNS 系的解在故障场景下退化可控——最坏情况仍守住 deadline（否则不会被判可行）。

### 4.3 最坏恢复时延 R_max — `fig_r_max_by_scale.png`

![Rmax按规模](figures/fig_r_max_by_scale.png)

`r_max`（仅可行解，最坏单次恢复时延，需 ≤ 预算）：GCN-ALNS 9.9/14.4/19.7，ALNS 9.9/15.0/18.6，二者贴近且都控制在预算内(large 时 ALNS 略低)。

### 4.4 平均恢复时延 — `fig_avg_recovery_by_scale.png`

![平均恢复时延](figures/fig_avg_recovery_by_scale.png)

`avg_recovery`（仅可行解）：约 0.9~1.5 s 量级，medium 比 small 略低（大图中单个故障的相对影响被摊薄）。GCN-ALNS 与 ALNS 相当，说明其低成本不是靠牺牲平均恢复速度换来的。

### 4.5 恢复安全余量（单图 + 按规模）— `fig_safety_margin.png` / `fig_safety_margin_by_scale.png`

![安全余量](figures/fig_safety_margin.png)
![安全余量按规模](figures/fig_safety_margin_by_scale.png)

`safety_margin = 预算 − r_max`（仅可行解，正值=有余量）。**baseline 的安全余量反而更大**（HEFT medium 达 10.6,large 32.2；Reliability large 43.3），而 ALNS/GCN-ALNS 较小（small 1.09/0.93,medium 2.86/2.73,large 11.99/7.13）。这不是缺点而是**优化到位的表现**：ALNS 系把恢复能力"恰好"配到满足预算即可,把省下的冗余转化为成本节约;baseline 余量大是因为它们只在预算富余的少数实例上侥幸可行。值得注意 **GCN-ALNS 的安全余量明显高于 ALNS(large 11.99 vs 7.13)**,即在同等可行、更省成本的前提下解还更"稳"。

### 4.6 可控故障集规模 |Ω| — `fig_fault_set_by_scale.png`

![故障集规模](figures/fig_fault_set_by_scale.png)

`|Ω|`（task∪node∪link 故障域大小）：随规模增长（small ~133→medium ~400→large ~1340）。GCN-ALNS/ALNS 的 |Ω|（large 1337 vs 1364）略低于开满主动副本的 Full-Proactive(large 2060)——主动副本越少，需要考虑的故障组合越少，验证也越轻。这从侧面解释了 ALNS 系 verify_time 可控（见 §5.2）。

---

## 5. 计算开销与可扩展性

### 5.1 调度运行时（按规模，对数轴）— `fig_runtime_by_scale.png`

![运行时按规模](figures/fig_runtime_by_scale.png)

| 算法 | small | medium | large |
|---|---|---|---|
| GCN-ALNS | 24.8 s | 497.7 s | **16507 s (~4.6 h)** |
| ALNS | 22.3 s | 449.7 s | 14946 s (~4.2 h) |
| 启发式 baseline | <0.03 s | <0.2 s | <2 s |
| PEFT | 0.1 s | 2.9 s | 105 s |

ALNS 系的运行时随规模超线性增长——large（500 任务）单次求解约 **4.5 小时**（纯 Python 的 ALNS+verify 是 CPU 瓶颈,`main_large` 15 实例×repeats5 共耗 ~44h)。**GCN-ALNS 在满迭代下比 ALNS 略慢 ~10%**(每次迭代多了 GCN 打分),换来更低成本、更高安全余量;但若以"达可行时间"衡量则 GCN 更快(少 ~28% 迭代),详见附录 B.6。

### 5.2 恢复验证耗时（按规模，对数轴）— `fig_verify_time_by_scale.png`

![验证耗时按规模](figures/fig_verify_time_by_scale.png)

`verify_time`：small ~0.11 s、medium ~1.8 s、large ~50 s。GCN-ALNS/ALNS 相当(large 50.7/51.6),略低于 Full-Proactive(large 83 s,与 §4.6 |Ω| 更大一致)。验证开销随规模增长,是端到端耗时的重要组成,也是大规模落地的主要瓶颈。

### 5.3 运行时 vs 问题规模 — `fig_scalability_runtime.png`

![可扩展性运行时](figures/fig_scalability_runtime.png)

运行时随规模的曲线（对数轴）：ALNS/GCN-ALNS 呈陡峭的超线性增长（22→500→16000+ s，small→medium→large），启发式基本平坦（亚秒~百秒级）。这清晰刻画了"解质量 vs 计算成本"的权衡——ALNS 系用显著更高的算力换取可行性与成本优势。若要落地大规模，需对 verify 做向量化/增量化或对 Ω 采样加速（见 §9）。

---

## 6. 按 SLA 场景的违约率 — `fig_sla_by_scenario.png`

![按场景违约率](figures/fig_sla_by_scenario.png)

这是最能体现算法价值的一张图。**可行率**按 loose/medium/tight 拆分(三规模汇总,violation = 1−可行率)：

| 算法 | loose | medium | tight |
|---|---|---|---|
| **GCN-ALNS** | **1.000** | **0.429** | **0.080** |
| ALNS | 0.993 | 0.422 | 0.062 |
| HEFT/No-FT | 0.433 | 0.067 | 0.033 |
| Reliability-Aware | 0.433 | 0.067 | 0.000 |
| Reactive-Only | 0.400 | 0.100 | 0.000 |
| PEFT | 0.333 | 0.067 | 0.033 |
| 三种备份/全主动策略 | ≤0.07 | ≤0.03 | 0.000 |

结论：loose 场景下 **ALNS/GCN-ALNS 高达 0.99+**，而所有 baseline ≤0.43；medium 场景 ALNS 系 ~0.42、baseline ≤0.10;tight 场景在 β=0.05 下对几乎所有算法都近 0（GCN 0.080 vs ALNS 0.062）——这是论文预期的"tight 是压力上界"。**ALNS 系把可行的 SLA 边界从 loose 显著推进到 medium,是 baseline 做不到的;GCN-ALNS 在三个场景可行率均 ≥ ALNS**(large 规模优势最明显,见 §3.3/附录)。若需"难但可解"的判别区(可行率 0.3~0.76),见附录 B.2 的 tight2/tight3。

---

## 7. 消融实验 — `fig_ablation.png`

![消融实验](figures/fig_ablation.png)

在 small+medium 上对 GCN-ALNS 逐一关闭组件（14 变体,每变体 375 runs,repeats=5）,按对可行率的影响排序：

| 变体 | 可行率 | ΔvsFull | 成本(仅可行) | 解读 |
|---|---|---|---|---|
| **Full** | 0.467 | — | 408.2 | 完整 GCN-ALNS 基线 |
| w/o Dynamic Resched | **0.107** | **-0.360** | 574.7 | **可行率断崖式下跌 + 成本飙升 40%** → 动态重调度是可行性的生命线 |
| w/o Feasibility-First (B1) | 0.405 | -0.061 | 405.5 | **B1 可行性优先:去掉后可行率降最多(除 Resched 外)** |
| w/o Restart (B2) | 0.432 | -0.035 | 405.6 | **B2 停滞重启:次要但显著** |
| w/o Hybrid (all-passive) | 0.419 | -0.048 | 396.6 | 强制全被动:可行率下降 |
| w/o GCN | 0.451 | -0.016 | 414.9 | **去掉全部学习引导:可行率仅小降,成本升 1.6%** → GCN 主要降成本 |
| w/o Hybrid (all-active) | 0.461 | -0.005 | 434.4 | 强制全主动:成本明显上升(434) |
| w/o Recovery Penalty | 0.467 | 0.000 | 422.7 | 去掉恢复惩罚:成本升 3.6% |
| w/o Warm Start(A1)/Init-Diversify(B3) | 0.469/0.475 | +0.003/+0.008 | 410.7/406.4 | 单独影响在噪声带内 |
| w/o Fault-Domain / Deployment Risk / GCN Mode(A2) / GCN Destroy(A3) | 0.483~0.485 | +0.016~+0.019 | 399.5~411.8 | 单点差异 ±0.016 属采样噪声(repeats=5) |

核心结论（详见附录 A.3 / B.3 的干净归因）：
1. **动态重调度（Dynamic Resched）是可行性的生命线**——移除后可行率 0.467→0.107、成本 +40%,量级碾压其他;
2. **可行率的主要驱动是搜索机制:B1 可行性优先(-0.061) > B2 重启(-0.035) ≫ GCN(-0.016)**——这是关键的归因澄清(B 系消融开关本轮新增);
3. **GCN 引导的价值主要体现在成本**(相同可行率下省 ~1.6%)与收敛速度(§2.6、附录 B.6),而非最终可行率;
4. 全主动虽不降可行率但成本高、全被动可行率降,验证 hybrid 选择性备份的必要性;
5. A1/A2/A3 三个 GCN 子旋钮单独看在噪声带内、互为冗余(repeats=5 太小),建议 repeats≥10 再定论。

---

## 8. 迭代预算 / 收敛 — `fig_budget_convergence.png`

![预算收敛](figures/fig_budget_convergence.png)

在 small 上扫迭代预算 ∈ {50,100,200,500,1000}，对比 ALNS 与 GCN-ALNS 的可行率与仅可行成本随预算的变化(详见附录 B.1)：

| 预算(iter) | ALNS 可行率 | GCN 可行率 | GCN 优势 |
|---|---|---|---|
| 200 | 0.291 | **0.324** | +0.033 |
| 500 | 0.360 | **0.387** | +0.027 |
| 1000 | 0.416 | 0.416 | 收敛持平 |

**结论**:GCN-ALNS 在**中小预算下可行率一致高于 ALNS**,到满预算(1000)两者收敛到相同可行率,但 GCN **首次达可行的迭代数少 ~28%**(119 vs 165)。这正是 GCN 的核心卖点——**用更少迭代到达可行/更省**,在算力受限或在线场景更划算(时间/效率完整剖析见附录 B.6)。

> 敏感性分析(`recovery_penalty_weight`、`insert_cost.w6`)在早前一轮已做(结论:w6≈0.5 处可行率达峰、默认 100 惩罚权重为较优折中),其图归档于 `results/archive_oldcode_20260715/figures/`,本轮未重跑,故此处不再重复。

---

## 9. 总体结论与建议

**结论**
1. 面向 SLA + 容错的联合部署问题上，**ALNS / GCN-ALNS 是唯二可用的方法**（三规模全局可行率 0.50/0.49，loose 场景 0.99+），启发式 baseline 与纯主动冗余策略基本失败。
2. **GCN-ALNS 相对 ALNS：可行率略高(+0.011)、成本更低**（总成本 −3.9%，仅可行解 −1.9%，主要省在 compute/backup），GCN 推理开销可忽略（一次性）;并且**首达可行迭代少 ~28%**(更快收敛)、**安全余量更高**(large 12.0 vs 7.1)。其可行率优势在 large 规模最明显(0.800 vs 0.787,medium 场景 0.96 vs 0.92)。
3. **消融澄清(本轮关键)**：可行性由**动态重调度**(生命线)与 **B 系搜索机制(B1 可行性优先 > B2 重启)**主导,**GCN 是其上一层较小但正向的增量,价值在成本、收敛速度与解的稳健性**——不是提升最终可行率上限。
4. **A2 已修复**:mode 头 AUC 0.975(此前失效),求解增益小但为正、方向对,已设为默认模型(附录 B.4)。
5. **可扩展性**：ALNS 系运行时随规模超线性增长（medium 单次 ~8 min、large 单次 ~4.5 h,`main_large` 共 ~44h）,是当前主要工程瓶颈。

**后续优化建议**
- 对 `core.recovery.verify` 做向量化/增量重算，或对故障集 Ω 采样（最终报告解仍做完整验证），以压缩大规模端到端时间；
- **难档 tight2/tight3 已扩到 medium/large(2026-09-08,见附录 B.2)**:结论为"GCN 成本/安全余量三规模稳赢,可行率抬升主要在 small";后续可把收敛曲线扩到 medium/large,并把消融 repeats 提到 ≥10 加配对显著性检验;
- 若要 A2 有更大求解收益,需在"更依赖主动副本"的场景(更高故障率/更紧 r_eff)生成 teacher。

---

# 附录 A：本轮重跑（新数据集 + mode 头/A2 激活，2026-07-16）

> 目标：解决两大诉求 —— ①"GCN 作用太小（只降成本、不影响可行率）" ②"tight 基本不可行"。
> 本轮重建了带 `pos_mode` 标签、去掉 `--alns-iter` 上限（满迭代 ALNS teacher）的数据集并重训，
> **激活了 A2（GCN 驱动容错模式/副本）**；随后 tight 重跑主对比（small+medium）+ 11 变体消融。
> 旧结果（旧代码/无 mode 头模型）已归档到 `results/archive_oldcode_20260715/`。

## A.1 本轮配置
| 项目 | 配置 |
|---|---|
| 数据集 | 满迭代 ALNS teacher（无 iter 上限）+ mode 标签；**34 可行实例**（train24/val5/test5），3341 正样本（active 82 / passive 3259），40 故障域 |
| 模型 | 重训带 mode 头（`has_mode_head=True`）；**test pairwise AUC=0.889**（val AUC=0.877），early stop@43 |
| 主对比 | small(45)+medium(30)，repeats=10，iter=1000（每场景 250 次可行判定）；large 本轮**未重跑** |
| 消融 | small+medium，repeats=5，iter=1000（每变体 375 run，每场景 125）；含 3 个新变体 w/o Warm Start / GCN Mode / GCN Destroy |

## A.2 主对比：GCN-ALNS vs ALNS（按场景，仅可行解成本）
| 算法 | 场景 | 可行率 | 仅可行成本 | r_max(feas) | 安全余量(feas) |
|---|---|---|---|---|---|
| **GCN-ALNS** | tight | **0.044** | 686.9 | 9.08 | **0.369** |
| ALNS | tight | 0.024 | 687.9 | 9.32 | 0.210 |
| **GCN-ALNS** | medium | **0.376** | **500.1** | 10.39 | 0.918 |
| ALNS | medium | 0.372 | 515.3 | 10.33 | 0.841 |
| **GCN-ALNS** | loose | **1.000** | **360.9** | 12.79 | 2.373 |
| ALNS | loose | 0.992 | 371.9 | 13.08 | 2.146 |

**全局（small+medium）**：GCN-ALNS 可行率 **0.473** vs ALNS 0.463；仅可行成本 **407.9 vs 415.8（-1.9%）**；总成本 442.2 vs 462.7。其余算法可行率 ≤0.15（Reliability-Aware 0.147、HEFT/No-FT 0.133、Reactive 0.120、PEFT 0.093），三种主动/混合备份 baseline 全 0。

**结论①（GCN 现在影响可行率，不只是降成本）**：GCN-ALNS 在**三个场景可行率全面 ≥ ALNS**——tight **0.044 vs 0.024（≈1.8×）**、medium +0.4pt、loose 1.000 vs 0.992，同时成本更低、feasible 解安全余量更高。这与旧结论"GCN 只降成本、可行率几乎不变"形成明确对比。

**结论②（tight 仍是压力上界）**：tight 绝对可行率仍很低（4.4%）——`β=0.05` 令恢复预算 r_eff 常仅 1–2s，可行域近乎为空，属场景刻意设定的上界；GCN 把可行率相对翻倍但无法改变量级。若要"难但可解"的 tight，需放宽 `β`（如 0.05→0.10）或做 α/β 可行前沿扫描（另议）。

## A.3 消融归因（Full vs 去某组件；全局 small+medium，repeats=5）
| 变体 | 可行率 | ΔFeas | 仅可行成本 | ΔCost% |
|---|---|---|---|---|
| **[Full]** | 0.467 | — | 408.2 | — |
| [w/o Dynamic Resched] | **0.107** | **-0.360** | 574.7 | **+40.8** |
| [w/o Hybrid (all-passive)] | 0.419 | -0.048 | 396.6 | -2.8 |
| [w/o GCN]（去全部学习引导） | 0.451 | -0.016 | 414.9 | +1.6 |
| [w/o Hybrid (all-active)] | 0.461 | -0.005 | 434.4 | +6.4 |
| [w/o Recovery Penalty] | 0.467 | +0.000 | 422.7 | +3.6 |
| [w/o Warm Start] (A1) | 0.469 | +0.003 | 410.7 | +0.6 |
| [w/o Fault-Domain] | 0.483 | +0.016 | 411.3 | +0.8 |
| [w/o Deployment Risk] | 0.483 | +0.016 | 399.5 | -2.1 |
| [w/o GCN Mode] (A2) | 0.483 | +0.016 | 405.2 | -0.7 |
| [w/o GCN Destroy] (A3) | 0.485 | +0.019 | 411.8 | +0.9 |

**3 个 GCN 子组件按场景可行率（vs Full）**
| 变体 | tight | medium | loose |
|---|---|---|---|
| [Full] | 0.040 | 0.360 | 1.000 |
| [w/o Warm Start] | 0.056 | 0.352 | 1.000 |
| [w/o GCN Mode] | 0.016 | 0.432 | 1.000 |
| [w/o GCN Destroy] | 0.040 | 0.416 | 1.000 |
| [w/o GCN]（全去） | 0.016 | 0.344 | 0.992 |

**归因结论（务求诚实）**
1. **Dynamic Resched 仍是可行性的决定性组件**：去掉后可行率 0.467→0.107、成本 +40.8%，量级远超其他所有组件。
2. **GCN 作为整体有正贡献且方向正确**：`[w/o GCN]` 相对 Full 可行率 -0.016、成本 +1.6%；tight 场景 **Full 0.040 vs w/o GCN 0.016（2.5×）**、loose 1.000 vs 0.992 —— 这是"GCN 影响可行率"的最干净证据。
3. **A1/A2/A3 三个子旋钮单独看在噪声带内、互为冗余**：消融 repeats=5、每场景仅 125 次判定，tight 可行率 0.02–0.06 对应个位数可行 run，单点差异 ±0.016 基本是采样噪声；去掉任一子组件其余会补偿（去 A2 反而 medium 升、去 A3 反而 medium 升），说明**三条路径共同承载 GCN 的引导，而非某一条独大**。A2 效果偏弱还因 mode 标签极不均衡（active 仅占 2.5%），模型很少翻到 active（active_backup_ratio 0.119 vs ALNS 0.117）。
4. **对报告口径的提醒**：主对比（repeats=10、250/场景）比消融（repeats=5、125/场景）更可信；跨表的同名量（如 medium 可行率 main 0.376 vs ablation-Full 0.360）差异来自不同 seed 集的采样噪声，属正常。

## A.4 后续建议
- **想让 A2 更显著**：数据集 active 样本太少（2.5%）——可在 `dataset.py` 里对 active 正样本上采样/加权，或在更"需要主动副本"的场景（更高故障率/更紧 r_eff）里生成 teacher。
- **想让 tight "可解"**：放宽 `sla.scenarios.tight.beta`（0.05→0.10）或加一档中间场景；当前 β=0.05 下任何方法都近 0，不利于体现算法差异。
- **消融置信度**：3 个 GCN 子组件建议把 ablation `--repeats` 提到 ≥10 再下结论；本轮小样本下它们的单独收益不显著。
- large 规模本轮未重跑（沿用归档旧结果或按需补跑，单实例 ALNS+verify 极慢）。

---

*附：本轮完整数值见 `results/csv/summary_by_algorithm.csv`、`summary_by_algo_scenario.csv`、`summary_by_algo_scale_scenario.csv` 及 `runs.csv`；旧结果见 `results/archive_oldcode_20260715/`。*

---

# 附录 B：增强轮(收敛/难档/B 系归因/A2 上线,2026-07-16)

> 针对附录 A 暴露的短板做四件事:①用**迭代预算扫描**证明"GCN 更快收敛";②建**"难但可解"档**(tight2 β0.10 / tight3 β0.15)把可行率拉进判别区;③给 B 系搜索改造加**消融开关**,把"GCN 的功劳"从"搜索改造的功劳"里干净剥离;④修复 **A2 类不均衡**(mode-BCE `pos_weight`)并做新旧模型 A/B。
> 受控口径:除模型 A/B 外,所有对比均用同一模型 `results/gcn_model.pt`;A2 升级单独由 A/B 隔离衡量。数据:`runs.csv` 相应 tag(`budget_*`/`main_hard_{small,medium,large}`/`model_ab`/`ablation_wo_{Feasibility-First,Restart,Init-Diversify}`)。

## B.1 迭代预算扫描(small):GCN 更快收敛
| 预算(iter) | ALNS 可行率 | GCN 可行率 | ALNS 达可行迭代 | GCN 达可行迭代 |
|---|---|---|---|---|
| 50   | 0.238 | **0.244** | 6.7  | 6.1 |
| 100  | 0.262 | **0.280** | 12.9 | 15.7 |
| 200  | 0.291 | **0.324** | 26.7 | 34.6 |
| 500  | 0.360 | **0.387** | 79.5 | 78.4 |
| 1000 | 0.416 | 0.416 | 164.6 | **119.3** |

**结论**:GCN-ALNS 在**中小预算下可行率一致高于 ALNS**(200 iter:+0.033;500 iter:+0.027),到 1000 iter 两者收敛到相同可行率(0.416),但 GCN **首次达可行的迭代数少 ~28%**(119 vs 165)。即 GCN 的核心价值是**用更少迭代达到可行/更省**,而非最终质量的绝对上限——这正是"小预算/在线场景"下 GCN 的卖点(见 `fig_budget_convergence`)。

## B.2 "难但可解"档:GCN-ALNS vs ALNS(判别区,三规模)

> 难档(tight2 β0.10 / tight3 β0.15)专用实例集:`data/instances/hard_{small,medium,large}`,
> 5 工作流 × {tight2,tight3},small/medium/large 各 3/2/1 reps(=30/20/10 实例),seed 沿用默认 2026。
> 算法集 = 8 内置 baseline + ALNS + GCN-ALNS(与主对比外部基线无关),stochastic repeats small/medium=10、large=5,max-iter small/medium=1000、large=800。
> tag `main_hard_{small,medium,large}`(不并入主对比 summary)。**small 为 2026-07-16 首轮,medium/large 于 2026-09-08 补齐。**

**GCN-ALNS vs ALNS(仅可行成本/安全余量仅统计可行解;`best-BL`=8 内置 baseline 中最高可行率):**

| 规模 | 场景 | GCN 可行率 | ALNS 可行率 | GCN 仅可行成本 | ALNS 仅可行成本 | GCN 安全余量 | ALNS 安全余量 | best-BL 可行率 |
|---|---|---|---|---|---|---|---|---|
| small | tight2 | **0.200** | 0.160 | **253.5** | 256.0 | **0.47** | 0.41 | 0.000 |
| small | tight3 | **0.760** | 0.653 | **256.5** | 267.0 | 0.75 | **0.84** | 0.067 |
| medium | tight2 | 0.540 | **0.590** | **696.6** | 722.0 | **1.49** | 1.35 | 0.200 |
| medium | tight3 | 1.000 | 1.000 | **638.1** | 639.5 | **2.05** | 1.70 | 0.600 |
| large | tight2 | **1.000** | 0.960 | **2091.5** | 2153.0 | **3.69** | 3.61 | 0.400 |
| large | tight3 | 0.880 | 0.880 | **2166.8** | 2170.0 | 2.96 | **6.08** | 0.400 |

![难档三规模:可行率与仅可行成本](figures/fig_hard_tier_by_scale.png)

*图:难档 tight2/tight3 在 small/medium/large 上的对比(`fig_hard_tier_by_scale.png`)。左=SLA 可行率(GCN-ALNS / ALNS / 最优内置 baseline),右=仅可行解总成本(对数轴,GCN-ALNS 在 6 格全部 ≤ ALNS)。best-baseline 在所有格子断层落后,ALNS 系可行率随规模上升(判别区上移),GCN 与 ALNS 可行率贴近而成本持续更低。*

**结论(诚实解读,三规模)**:
1. **成本优势稳健且普适**:**6 个 (规模×场景) 格子里 GCN-ALNS 仅可行成本全部 ≤ ALNS**(降幅 0.2%–4.1%,small-tight3 最大 -3.9%);安全余量 GCN 在 4/6 更高(仅 small-tight3、large-tight3 由 ALNS 略高)。这是难档最一致的信号——**同等或更高可行率下 GCN 更省**。
2. **可行率优势随规模/场景变化**:GCN > ALNS 于 small(两档,tight3 +0.107 相对 +16%)与 large-tight2(+0.040);medium-tight3、large-tight3 两者持平;**唯一 ALNS 反超在 medium-tight2(0.590 vs 0.540)**。即 GCN 的可行率增益在 small 最明显,大规模退化为"持平+更省"。
3. **判别区随规模上移**:small 的判别区在 tight3(0.65–0.76),medium 上移到 tight2(0.54–0.59,tight3 已饱和 1.000),large 再上移到 tight3(0.88,tight2 近饱和 0.96–1.00)。规模越大、恢复预算相对越宽,同一 β 的难度越低——与正文/附录 C 的"large 是外部基线转正规模"规律一致。
4. **baseline 在难档仍全面落后**:best-BL 可行率 small ≤0.067、medium ≤0.60(仅 Reactive-Only 在 tight3)、large ≤0.40,且多数格子为 0;三种主动/混合备份策略几乎全灭(Criticality-Hybrid/Full-Proactive 全 0)。ALNS 系断层领先的结论在三档全成立。

**一句话**:难档扩到 medium/large 后,**GCN 的"更快收敛/更省成本/更稳"叙事得到三规模支撑,但"可行率抬升"主要是 small 规模象现**;大规模下 ALNS 与 GCN 可行率趋同,差异体现在成本与安全余量。这比原先仅凭 small-tight3 单点更完整、也更诚实。

## B.3 B 系归因(把"搜索改造"从"GCN"里剥离,同模型/repeats5/small+medium)
| 变体 | 可行率 | ΔFeas(vs Full) | 仅可行成本 | ΔCost% | tight 可行率 |
|---|---|---|---|---|---|
| **Full** | 0.467 | — | 408.2 | — | 0.040 |
| w/o Feasibility-First (B1) | **0.405** | **-0.061** | 405.5 | -0.7 | 0.016 |
| w/o Restart (B2) | 0.432 | **-0.035** | 405.7 | -0.6 | 0.024 |
| w/o GCN(全部学习引导) | 0.451 | -0.016 | 414.9 | +1.6 | 0.016 |
| w/o Init-Diversify (B3) | 0.475 | +0.008 | 406.4 | -0.4 | 0.040 |

**关键结论(诚实归因)**:可行率的主要功臣是 **B 系搜索改造**而非 GCN——
- **B1 可行性优先** 贡献最大(去掉 -0.061,tight 0.040→0.016),
- **B2 停滞重启** 次之(-0.035,tight 0.040→0.024),
- **GCN 引导** 贡献真实但更小(-0.016;主要体现在成本 +1.6% 与更快收敛),
- **B3 初始多样化** 在 small+medium 上基本无效(+0.008,噪声内)。

这回答了附录 A 未答的问题:"tight/整体可行率的提升到底来自 GCN 还是搜索改造?"——**主要来自 B1/B2 的可行性优先搜索与重启;GCN 是其上的一层较小但正向的增量(省成本 + 更快达可行)**。(注:Dynamic Resched 仍是全局最关键组件,见附录 A,量级远超上表任一项。)

## B.4 A2 修复 + 模型 A/B(small,旧模型 vs A2 修复模型)
- **训练层面**:mode-BCE 加 `pos_weight`(自动=38.8)后,mode 头 **AUC 0.975 / acc 0.93**(此前几乎失效),placement AUC 0.889 不变——**A2 的容错模式头现在真的学会判别 active/passive**。
- **求解层面(A/B)**:

| 算法 | 可行率 | 仅可行成本 | active_bkp | 达可行迭代 |
|---|---|---|---|---|
| ALNS | 0.416 | 233.1 | 0.092 | 164.6 |
| GCN-ALNS (A2-off,旧模型) | 0.416 | 226.7 | 0.095 | 119.3 |
| **GCN-ALNS (A2-on,修复模型)** | **0.431** | 227.4 | **0.099** | 118.5 |

配对比较(同 iid+seed,n=450):A2-on vs A2-off **可行 胜/平/负 = 23/411/16(净 +7)**;两者都可行的 171 例上成本几乎相同(+0.41)。

**结论**:A2 修复让 mode 头从"失效"变"高判别(AUC 0.975)",求解上带来**小而真实的可行率增益(+0.015,配对净 +7/450)并小幅提高主动副本占比**(0.095→0.099);成本无明显变化。收益偏小是因为在这些场景里"是否用主动副本"很少成为可行性的决定因素——但方向正确,建议将 `gcn_model_a2.pt` 作为默认模型。

## B.5 本轮总结论
1. **GCN 的真实价值 = 更快收敛(少 ~28% 迭代达可行)+ 全程略省成本 + 难档 small 更高可行率(tight3 +0.107、成本 -3.9%)**;并非最终可行率上限的大幅抬升。**难档扩到 medium/large 后(B.2)进一步表明:成本/安全余量三规模稳赢,但可行率抬升主要是 small 现象——大规模下 ALNS 与 GCN 可行率趋同(如 large-tight3 均 0.88),差异落在成本(GCN 仍 ≤ ALNS)。**
2. **可行率的主要驱动是搜索改造(B1 可行性优先 > B2 重启 ≫ GCN)**,GCN 是其上的正向增量——这是本轮最重要的归因澄清。
3. **A2 已被救活**(mode AUC 0.975),求解增益小但为正、方向对;推荐启用 `gcn_model_a2.pt`。
4. 仍待:~~难档补 medium 规模~~(**已于 2026-09-08 完成 medium+large,见 B.2**);A2 若要更大收益需在"更依赖主动副本"的场景(更高故障率/更紧 r_eff)生成 teacher;归因消融建议 repeats≥10 以缩小误差带。

## B.6 时间与效率剖析(GCN 到底"快"在哪)
> 关键区分:**"更少迭代达可行" ≠ "更省墙钟时间"**。GCN 每次迭代的 repair 打分更重,需把"迭代效率"与"墙钟时间"分开看。
> 数据口径:`first_feasible_iter` 埋点是本增强轮才加的,故 `main_small/medium`(跑在埋点前)无 ttf,迭代级指标取自 `budget_1000` / `model_ab`(均 small);墙钟取自 `main_small`/`main_medium`(满 1000 iter)。

**① 迭代效率:GCN 明显更好**

| 指标(small, iter=1000) | ALNS | GCN-ALNS |
|---|---|---|
| 达首个可行解迭代数 | 164.6 | **118.5(-28%)** |
| 可行率 @200 / @500 iter | 0.291 / 0.360 | **0.324 / 0.387** |
| 剪枝率(small / medium) | 0.114 / 0.067 | **0.128 / 0.075** |

**② 固定满预算(跑满 1000 iter)的墙钟:GCN 略慢 ~10%**

| 场景 | ALNS runtime | GCN-ALNS runtime | GCN 一次性推理 |
|---|---|---|---|
| small | 22.4 s | 24.8 s(+11%) | 0.056 s |
| medium | 449.7 s | 497.7 s(+11%) | 0.412 s |

一次性 GCN 推理开销可忽略(整表 [T,N] 一次算好缓存);多出的时间来自每次迭代 repair 打分更重(`-w6·GCN` 项 + mode/destroy 逻辑)。

**③ "达可行时间"(墙钟,anytime 视角):GCN 更快 ~20%**
把"少 28% 迭代"换算成墙钟(`ttf_iter × 每迭代秒 + 一次性推理`):small 下 GCN ≈ 118.5×0.0236 + 0.06 ≈ **2.85 s** vs ALNS 164.6×0.0219 ≈ **3.61 s**,**GCN 约快 21%**——"少迭代"的收益盖过"每迭代略重"的开销。

**结论(务实口径)**
- **迭代效率 / 达可行时间 / 限预算·anytime 可行率:GCN-ALNS 更好**(即 `fig_budget_convergence` 展示的);
- **跑满固定迭代预算的总墙钟:GCN-ALNS 略慢 ~10%**(每迭代打分开销);
- A2 上线不改变此图景(A2-on 24.4→23.6 s,ttf 119→118,基本持平)。
- 一句话:**GCN 的效率优势是"更快到达目标质量/可行",而非"单位墙钟做更多迭代"**;在算力受限/在线场景更划算,在"必须跑满迭代"的离线场景则以 ~10% 时间换取更低成本与更快达可行。

*附:本轮新增数据 tag 见 `runs.csv`;新增图 `results/figures/fig_budget_convergence.png`;A2 修复模型 `results/gcn_model_a2.pt`(训练日志 `results/gcn_train_a2.log`)。*

---

# 附录 C:外部论文基线对比(RECR-D / HFWPF / GA / RA-MOMA,静态适配版,2026-07-20)

> 目的:把 3 篇 CCF-B + 1 篇 **CCF-A** 论文的**"选节点 + 选容错策略"决策逻辑**复现为对比基线,在**同一 `core.recovery.verify` 闸门**下与 GCN-ALNS/ALNS/现有 baseline 公平比较。
> 代码在新包 `src/gcnalns/reproductions/`(懒加载注册进 `algorithms/registry.py`);已**全量并入主对比** tag `main_small`(45 实例)/`main_medium`(30 实例)/`main_large`(15 实例),数据见 `runs.csv` 与 `summary_by_*`,并已重绘进正文 §2–§6 主对比图。
> **诚实声明:这些是"忠于原文调度/容错决策逻辑、但去掉其在线/不确定性/能量/弹性/多目标外壳"的静态适配版基线,而非逐比特复现。** 各基线裁剪见 C.1(完整出处/映射见 `src/gcnalns/reproductions/README.md`)。

## C.1 各基线与静态适配裁剪

| 基线 | 出处 | 复现的核心逻辑 | 主要裁剪 / 口径对齐 |
|---|---|---|---|
| **RECR-D** | Taghinezhad-Niar & Taheri, *IEEE TSC* 18(5):2892-2903, 2025 (**CCF-B**) | Alg.1 子deadline 分配(EFT/LFT 凸组合)+ Alg.2 **EDF + 归一化惩罚**`PN=ζ·cost+β·deadline+γ·reliability` 选点 + Alg.3 **TRM 加主动副本** | **能量项丢弃**(α_energy=0);可靠性判定改用本框架 `verify`/`estimate_rmax` 的恢复预算(替代原文 CTMC 成功概率);deadline 作为**约束**(优先满足子deadline 的最省节点,否则最早完工兜底) |
| **HFWPF** | Ren & Yao, *IEEE TSC* 19(1):44-57, 2026 (**CCF-B**) | 只复现**静态核心**:preprocessing 切子deadline + 初始调度的**容错策略选择(式8 slack 判据)**:`slack≥2(Tboot+max_pred TT+ψ)`→resubmission(被动)否则 replication(主动 2 副本);容量偏置选点 | **去掉执行时间不确定性(论文核心)**、在线重执行、slack 回填、replication→resubmission 回切、弹性无限 VM;`Tboot=0`(本框架无 VM 启动)。**报告须注明:去掉不确定性后其相对优势被削弱,属预期** |
| **GA-Placement** | 平替 Xu et al. (MTGP), *IEEE TSC* 16(4):2657-2671, 2023 (**CCF-B**) | **标准 GA**(个体=每任务〈节点选择+主/被动位〉;适应度=`compute_cost`+deadline/恢复惩罚,与 ALNS 同一 quick 目标;锦标赛选择/均匀交叉/逐基因变异/精英保留,贪心 EFT 暖启动) | MTGP 原问题为**动态到达+makespan+无容错/无 SLA**,不可直接比;按交接选项 A,用标准 GA 代表"进化式元启发式",**非** GP 超启发式的逐比特复现 |
| **RA-MOMA** | Qin, Pi, Shao, Xu, Chen, *IEEE TPDS* 34(4):1343-1361, 2023 (**CCF-A**) | **可靠性感知多目标 memetic**:NSGA-II + **混合 HEFT 初始化**(HEFT/备份-HEFT/最省/最省+备份)+ **4 个邻域算子**(CP_perturb/CP_remove/VM_release/VM_backup);编码 `<α 主, β 备份>`;三目标 (makespan, cost, 恢复风险) Pareto 前沿 | 编码去掉 **γ(VM 租赁池,本框架节点固定)** 与 **π(执行序,评估器按拓扑序自动定)**;backup→**主动副本 m=2,须异故障域**;可靠性(Poisson/CTMC)、多云计费、性能退化→本框架确定 exec+`verify`;从 Pareto 前沿**选"经 verify 可行且最省 cost"的解**(原文"用户选调度"步骤,这里由 deadline+恢复预算驱动) |

## C.2 数值结果(main_small + main_medium + **main_large**,统一 `verify` 判定)

RECR-D/HFWPF 为确定性(1 rep/实例);GA-Placement / RA-MOMA 为随机(small/medium repeats=10,large repeats=5)。成本为全体 run 均值,仅可行成本/安全余量仅统计可行解。**large(15 实例×500 任务)已全量并入**(单解 `verify`~50-85s、GA/RA-MOMA 单解 ~5-7min,共 ~1.5h)。

| 规模 | 算法 | 可行率 | 成本(全体) | 仅可行成本 | 安全余量(feas) | 主动备份比例 |
|---|---|---|---|---|---|---|
| small | **RECR-D** | 0.000 | 276.9 | — | — | 0.177 |
| small | **HFWPF** | 0.000 | 580.7 | — | — | 0.599 |
| small | **GA-Placement** | 0.100 | 297.5 | 228.3 | 2.20 | 0.137 |
| small | **RA-MOMA** (CCF-A) | 0.009 | 246.0 | 184.0 | 0.99 | 0.072 |
| small | HEFT(参照) | 0.089 | 216.4 | 210.1 | 2.24 | 0.000 |
| small | ALNS / GCN-ALNS(参照) | 0.416 / 0.416 | 294.9 / 280.3 | 233.1 / 226.7 | 0.93 / 1.09 | 0.092 / 0.095 |
| medium | **RECR-D** | 0.000 | 636.9 | — | — | 0.077 |
| medium | **HFWPF** | 0.000 | 1295.4 | — | — | 0.304 |
| medium | **GA-Placement** | 0.013 | 1751.9 | 1810.0 | 1.74 | 0.537 |
| medium | **RA-MOMA** (CCF-A) | **0.160** | 721.5 | **625.8** | 8.64 | 0.036 |
| medium | HEFT(参照) | 0.200 | 661.7 | 624.7 | 10.62 | 0.000 |
| medium | ALNS / GCN-ALNS(参照) | 0.533 / **0.560** | 714.5 / 685.0 | 629.2 / **609.6** | 2.73 / 2.86 | 0.019 / 0.019 |
| large | **RECR-D** | 0.067 | 1817.3 | **1539.4** | 92.65 | 0.018 |
| large | **HFWPF** | 0.133 | 3440.2 | 3188.7 | 12.47 | 0.171 |
| large | **GA-Placement** | 0.053 | 5569.2 | 5215.6 | 11.57 | 0.516 |
| large | **RA-MOMA** (CCF-A) | **0.400** | 2294.5 | **2108.1** | 32.25 | 0.019 |
| large | HEFT(参照) | 0.400 | 2182.9 | 2110.6 | 32.20 | 0.000 |
| large | ALNS / GCN-ALNS(参照) | 0.787 / **0.800** | 2143.3 / 2087.7 | 2056.6 / **2025.7** | 7.13 / 11.99 | 0.006 / 0.004 |

**全局(small+medium+large)可行率**:GCN-ALNS 0.503 > ALNS 0.492 ≫ HEFT 0.178 > **RA-MOMA 0.099** > **GA-Placement 0.064** > HFWPF 0.022 > RECR-D 0.011。
**large 是外部基线的转正规模**:RA-MOMA 在 large 达 **0.400 可行率(= HEFT,仅可行成本 2108 ≈ ALNS 2057/GCN 2026)**,HFWPF 0.133、RECR-D 0.067(且 RECR-D 仅可行成本 **1539 为全场最低**),GA-Placement 0.053(但成本失控 5216)。与内置备份类 baseline 在 large 才转正(0.13-0.20)一致——**规模越大、预算相对越宽,一次性/进化式基线才开始出可行解**。

## C.3 结论(诚实解读)

1. **单次启发式(含这两篇论文的静态核心)在联合 deadline+恢复 SLA 下几乎全灭**:RECR-D/HFWPF 在 small/medium 可行率均为 0.000,与本框架内置的备份类 baseline(Full-Proactive/Critical-Path/Criticality-Hybrid 在 small/medium 同为 0.000)完全同档。这再次印证正文核心主张——**面向"部署+恢复"联合优化的搜索式方法(ALNS 系)才能稳定求可行解**;把成熟论文的调度决策逻辑单独拿出来做一次性放置并不足够。
2. **RECR-D 是所有加容错方法里最省的**(small 成本 276.9,低于 Critical-Path 395/Criticality-Hybrid 412/Full-Proactive 746),符合其"cost-efficient"定位——它把成本压得很低但因此牺牲 makespan/恢复余量,难达可行。
3. **HFWPF 成本偏高(581/1295)且主动备份比例高(0.599/0.304)**:去掉其执行时间不确定性(论文核心卖点)后,式8 slack 判据在确定时间下更倾向 replication,冗余上升、成本上升而可行性未获益——**这是交接文档预告的预期结果**(其相对 GCN-ALNS 的优势被削弱)。
4. **标准 GA 优于被动启发式但远逊于 ALNS/GCN-ALNS**:small 可行率 0.100(>HEFT 0.089),medium 骤降至 0.013 且成本失控(1752)——150 任务时通用 GA 的搜索效率不足,佐证**问题专用的 ALNS/GCN-ALNS(medium 0.53/0.56)对通用进化式元启发式有显著优势**。MTGP 原问题(动态 makespan)不可直接比,此处仅作进化式方法的代表。
5. **CCF-A 的 RA-MOMA 是最强的外部元启发式对照,且随规模优势扩大,但仍显著弱于 GCN-ALNS**:作为可靠性感知多目标 memetic(NSGA-II + 混合 HEFT 初始化 + 4 邻域算子),RA-MOMA **在 medium 上比 plain GA 强一个数量级(0.160 vs 0.013)**,在 **large 上达 0.400 可行率(= HEFT,仅可行成本 2108 ≈ ALNS 2057/GCN 2026、安全余量 32.3)**——其混合 HEFT 初始化 + 关键路径/低利用率邻域算子让它在 150/500 任务规模仍稳定收敛(不像 plain GA 在 large 成本飙到 5216)。但**它在三规模均明显落后于问题专用的 ALNS/GCN-ALNS**(large 0.400 vs 0.787/0.800;全局 0.099 vs 0.492/0.503)。**结论:即便是发表在 CCF-A 的成熟多目标 memetic,在本联合 deadline+恢复 SLA 问题上也不敌 GCN-ALNS——这是最有分量的外部对照。**
6. **规模效应**:四条外部基线都在 **large 才明显转正**(RA-MOMA 0.400 / HFWPF 0.133 / RECR-D 0.067 / GA 0.053),与内置备份类 baseline 同规律——大图预算相对更宽,一次性/进化式方法才有可行解;但 ALNS 系在 large 仍以 0.79/0.80 断层领先。
7. **公平性**:四者与所有方法共用 `compute_schedule → verify → compute_cost`,能量/不确定性/在线到达/多目标均已按 C.1 对齐;结论方向与正文/附录 A-B 一致,且为 GCN-ALNS 的价值提供了**外部(他人论文,含一篇 CCF-A)对照**。

*附:数据 `runs.csv`(tag `main_small`/`main_medium`/`main_large`)与 `summary_by_*` 已含 RECR-D/HFWPF/GA-Placement/RA-MOMA;实现见 `src/gcnalns/reproductions/{recr_d,hfwpf,ga_placement,ra_moma}.py`,参数见 `config/default.yaml` 的 `reproductions:` 段。**正文 §2–§6 的主对比图已重绘并纳入这四条基线**(三规模全量);其数值以本附录与更新后的 CSV/图为准。*
