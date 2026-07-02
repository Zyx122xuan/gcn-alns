# GCN-ALNS 实现进度与续作交接 (Handoff)

> 本文件用于跨 session 续作。记录已完成内容、当前状态、剩余工作、已知问题与恢复方式。
> 计划文件位于 `~/.cursor/plans/gcn-alns-experiment-plan_6de824b3.plan.md`（**不要编辑**）。

## 一句话状态
论文 RLC-HFTSP / GCN-ALNS 实验平台：**阶段 0–9 全部完成；已并行化改造并在新机（32 核 / 4×A40 / 128GB）上跑大规模实验**。
**主实验（90 实例三规模、全量迭代）已完成出图**；消融 / 敏感性 / 可扩展性正在后台运行（结束后自动重聚合+重绘）。
大规模主实验结论（见 `results/csv/summary_by_algorithm.csv`）：**仅 ALNS / GCN-ALNS 达成显著 SLA 可行率**
（GCN-ALNS 可行率 0.464、成本 541.8 ≤ ALNS 0.469 / 543.9；启发式 baseline 可行率 ≤0.18，Full-Proactive 0 且成本最高 2313），符合论文预期。

## 技术栈与环境
- **当前实验机**：工作区 `e:\user\zhangyuxuan\gcn-alns\gcn-alns`，CPU **32 核**、**4×A40**(46/49GB)、128GB RAM。运行需 `$env:PYTHONPATH="src"`。
- **torch `2.5.1+cu121`，CUDA 可用（4 卡）**——旧机的 "torch 2.4.1+cpu / CPU-only" 描述已过时；GCN 训练用 `--device cuda:0`。
- 已安装：numpy, pandas, pyyaml, networkx, matplotlib, scipy, tqdm, pytest。`torch-geometric` 仍**不被模型使用**（自包含稠密 GCN，见阶段6）。
- Windows PowerShell，shell 命令用 `$env:PYTHONPATH="src"; python ...`。
- 注意：matplotlib 首次 import 会构建字体缓存（之后正常；本机已建好，约数秒）。
- **真正瓶颈是纯 Python 的 ALNS+verify（CPU 密集）**；GPU 仅对 GCN 训练有意义。已对实验编排做多进程并行（见下）。

## 已完成（阶段 0–5）
- **阶段0 脚手架**：`requirements.txt`、`pyproject.toml`、`README.md`、`.gitignore`、`config/default.yaml`（所有默认参数与种子的唯一来源）、`src/gcnalns/config.py`、`utils/seeding.py`、`utils/io.py`。
- **阶段1 数据层**（`src/gcnalns/data/`）：
  - `synthetic.py`：5 类工作流 DAG + Alibaba 风格机器快照（同 parser schema）。
  - `pegasus_parser.py`、`alibaba_parser.py`：真实数据 parser（数据到位即切换，输出同 schema）。
  - `layering.py`：cloud/edge/device 分层（top20/mid50/bottom30 + 波动/容量修正）、故障域、f_k、λ_k。
  - `links.py`：`LinkFactory`（按层对表 + U(0.8,1.2) 扰动，种子 2026，确定性可重建）、`get_factory`、`build_continuum`。
  - `sla.py`：HEFT-NoFT 参考 → D=α·T_ref、R_max=β·T_ref。
  - `instance_builder.py`：构造实例 + JSON 序列化(`instance_to_dict`/`instance_from_dict`/`load_instance`) + CLI。
- **阶段2 核心原语**（`src/gcnalns/core/`）：`model.py`(Task/Node/Link/Workflow/Continuum/Instance/Solution)、`dag.py`(拓扑/ranku/EST/EFT/LFT/slack/结构关键性 h_i)、`schedule_eval.py`(无故障 list-scheduling、`compute_schedule`、容量/故障域检查)、`cost.py`(5 项成本分解)。
- **阶段3 恢复验证器（闸门，已锁定）**：`fault_set.py`(Ω=task∪node∪link)、`recovery.py`(`simulate_fault`/`verify`/`effective_budget`/`estimate_rmax` 快估)。`tests/test_recovery_verifier.py` **4 个 golden case 全过**。
- **阶段4 baseline**（`src/gcnalns/algorithms/`）：`heft.py`(HEFT/PEFT + 共享贪心 `_greedy_eft_place`)、`baselines.py`(NoFT/ReactiveOnly/FullProactive/CriticalPathBackup/CriticalityHybrid/ReliabilityAware)、`replicas.py`(故障域隔离副本放置，负载均衡+偏快节点)、`base.py`(Scheduler 接口)、`registry.py`(名称→类)。全部复用同一 verifier。
- **阶段5 ALNS（无 GCN）**：`algorithms/alns/`：`insertion.py`(`Repairer`+InsertCost+快速预筛选)、`operators.py`(5 destroy + 5 repair)、`solver.py`(ALNS 主循环：自适应 roulette 权重、SA 接受、快估驱动搜索 + 仅对“看起来可行且更省”的候选做精确 verify 来更新 feasible incumbent)。

## 验证记录（已通过）
- `python -m gcnalns.data.instance_builder ...` → 生成 15 个实例（`data/instances/`，5 类×3 SLA）。
- `pytest tests/test_recovery_verifier.py` → 4 passed。
- 全 baseline + PEFT 修复后跑通并经 verifier 输出合理指标。
- 阶段6：`gcn.dataset` 建出 4 个可行 loose 实例（train3/val1）；`gcn.train` 收敛、early stop、存 `results/gcn_model.pt`。
- 阶段7：`GCN-ALNS` 在 loose 实例上 feasible=True，gcn_infer≈2.7s/实例(首次含建表)。消融 `w/o GCN/RecPenalty/FaultDomain` 三路径均跑通。
- 阶段8：`run_main`(15 实例×10 算法, max_iter=50/repeats=2) 产出 `results/raw_json/main.json` + 5 个 CSV。结果：**仅 ALNS(可行率0.23)/GCN-ALNS(0.27) 达成 SLA 可行，GCN-ALNS 成本(276.7)≤ALNS(280)**，baseline 全 0；Full-Proactive 成本最高(748)且仍不可行。
- 阶段9：`viz.plots` 产出 18 张 PNG 到 `results/figures/`。

## 已完成（阶段 6–9）
### 阶段6 GCN 训练（`src/gcnalns/gcn/`）—— 已完成
- `features.py`：任务特征(12 维结构属性) / 节点特征(12 维含 layer one-hot) + 故障域全局索引(给 `nn.Embedding`) / 交互特征 z_ik(9 维: t_ik、P_fail、RecRisk、cost、comm、compute_ratio、cpu/mem/deadline 满足位)。物理图用按通信时延的 kNN(`gcn.knn_k`)。全部纯 numpy。
- `dataset.py`：跑 NoFT/Reactive/Full-Proactive/Criticality-Hybrid/ALNS → 完整 verify → feasible 解按成本 top20% 为高质量解 → 正样本(被选 (i,k))、可行未选为负(训练时按 `neg_per_pos` 采样)、不可行 mask。按**实例** 70/15/15 划分(`seeds.split`)。pickle 存 `data/gcn/dataset.pkl`。`--alns-iter` 可限 ALNS 迭代以加速。
- `model.py`：`GCNScorer` = 故障域 embedding + 2 层 GCN(hidden 64) 编码节点 + task MLP + score MLP；`Score=MLP([h_i,h_k,h_i⊙h_k,z_ik])`。**GCN 卷积用自包含的稠密对称归一化邻接实现(`_DenseGCNConv`)，不依赖 torch-geometric**——保证 train/infer 一致、不受 PyG 版本/导入卡顿影响。
- `train.py`：BCE(pointwise) + `rank_loss_weight`·BPR pairwise ranking，Adam + weight_decay(L2)，early stop on val。参数全读 `config["gcn"]`。存 ckpt(含 `num_domains`)到 `results/gcn_model.pt`。
- `infer.py`：`GCNScorerRunner.load(...)` → `make_hook(instance,factory)` 返回 `gcn_score(i,k)`(sigmoid 分，infeasible 置 0)，整张 [T,N] 表一次算好缓存。

### 阶段7 GCN-ALNS（`algorithms/gcn_alns.py`）—— 已完成
- `GCNALNS(ALNS)`，`use_gcn=True`，`_gcn_hook` 惰性加载模型并返回闭包；`gcn_infer_time` 计时。`ablation={"gcn":False}` 退化为普通 ALNS。`registry.get_scheduler("GCN-ALNS")` 接通。InsertCost 的 `-w6*GCNScore` 已生效。模型路径 `cfg["gcn"]["model_path"]`。

### 阶段8 实验编排（`src/gcnalns/experiments/` + `eval/`）—— 已完成
- `eval/metrics.py`：`evaluate()` 产出 §18 全部指标(可行/违约、c_base/c_worst、r_max/r_eff/avg_recovery/safety_margin、5 项成本分解、active_backup_ratio、|Ω|、runtime、verify_time、gcn_infer_time、convergence_iter、prune_ratio)。
- `eval/aggregate.py`：原始 JSON → `runs.csv` + 按 算法/场景/规模/工作流 的 summary CSV(mean/std/count)。
- `experiments/runner.py`：`run_experiment()` 核心(随机算法多 seed 重复、确定算法 1 次、统一过 verify)。入口：`run_main.py`、`run_ablation.py`(7 变体)、`run_sensitivity.py`(扫 dotted 参数)、`run_scalability.py`(可 `--generate` 多规模实例)。

### 阶段9 绘图（`src/gcnalns/viz/plots.py`）—— 已完成
- 只读 CSV，产出 18 张 PNG(成本/可行率/违约/恢复/安全余量/备份比例/makespan/运行时/verify/GCN推理/收敛/剪枝/|Ω| + 成本堆叠分解 + 按场景违约 + 规模-运行时散点)。缺列/缺文件自动跳过。

## 大规模实验（本轮，新机）
### 并行化 + 鲁棒性改造（已实现并对拍验证）
- `experiments/runner.py`：把 `(算法,实例,rep)` 展平为任务列表，用 `ProcessPoolExecutor` 多进程并行；新增顶层 worker `_run_one`（spawn 安全）+ `_worker_init`（限 `OMP/MKL/OPENBLAS/NUMEXPR=1` 且 `torch.set_num_threads(1)`，避免过订阅）。新增参数/CLI：`--workers`（默认 `min(30,cpu)`）、`--max-iter`（覆盖 `alns.max_iter`）、`--tag`，`--instances` 支持**多目录**(`nargs="+"`)。
- **增量落盘 + 断点续跑**：每完成一条结果立即 append 到 `results/raw_json/<tag>.jsonl`；启动时按 `(tag,algorithm,iid,rep)` 跳过已完成项；结束再写 `<tag>.json` 并 `aggregate`。长任务中途崩溃可无损续跑。
- `gcn/dataset.py`：`build_dataset` 多进程并行 `build_instance_sample`，新增 `--workers` 与多目录 `--instances`。
- 各入口脚本（run_main/run_ablation/run_sensitivity/run_scalability）均透传 `--workers/--max-iter` 与多目录 `--instances`。
- 校验：GCN 推理走 CPU（`infer.py` `map_location="cpu"`），30 worker 不抢 GPU；`--workers 1` vs `--workers 6` 对拍 17 个确定性指标 **0 mismatch**；resume 复跑确认"nothing to do"。

### 数据 / 模型（已产出）
- 实例：`data/instances/{small,medium,large}` = **45 / 30 / 15 = 90 个**（5 工作流 × 3 SLA × N）。
- GCN 数据集：`data/gcn/dataset.pkl`，**38 个可行实例**（train27/val6/test5）；tight 场景多数不可行（论文预期）。建集日志 `data/gcn_build.log`。
- GCN 模型：`results/gcn_model.pt`（A40，early stop@55，best val=0.5749，~39s）。训练日志 `results/gcn_train.log`。

### 实验运行状态
- **主实验（已完成）**：`run_main` 三规模分 tag —— `main_small`(repeats10,iter1000,1260 runs)、`main_medium`(repeats10,iter1000,840 runs)、`main_large`(repeats5,iter800,270 runs)。结果见 `results/raw_json/main_*.json` 与 `results/csv/*.csv`，18 张图见 `results/figures/`。
  - 关键：`main_large` 单作业耗时 ~31h（500 任务 ALNS+verify 极重，个别 large 解单次达小时级）。
- **后台运行中**：`run_ablation`(small+medium, repeats8, 7 变体)、`run_sensitivity`(small, repeats5, 扫 `recovery_penalty_weight`/`insert_cost.w6`)、`run_scalability`(三规模, repeats3, **--max-iter 800** 控时)；链尾自动 `aggregate` + `viz.plots` 重出全量 CSV/图。

## 本轮修复（plan: gcn-alns-experiment-fixes，代码已完成并验证）
四个实验问题的根因修复均已落地、跑通并小规模对拍验证（待三规模全量重跑出报告级结果）：
1. **GCN 增益看不出来 → 已修**：`insertion.py` 把 InsertCost 拆成分量并在「候选集内」min-max 归一化（`insert_scores`），GCN 项改为 `− w6·h_i·gcn`（`h_i=criticality`）。`config` 增 `insert_cost.normalize:true`，`w6` 上调到 3.0。新增 `run_budget.py`（迭代预算 50/100/200/500/1000 扫描）凸显 GCN 在小预算下更快达可行/更省。
   - 验证：60 迭代下 GCN-ALNS 已可行(cost 265)，plain ALNS 不可行(cost 291)；budget 图 GCN-ALNS 可行率显著高于 ALNS。
2. **GCN 接近随机 → 已重训**：`dataset.py` 按建模 sec.13 加权标签 `(1+h_i)(1−Rmax/Δ̄)` 重建数据集（`pos_w`），`train.py` 用加权 BCE+BPR，训练后报告测试集排序指标。**重训结果 test pairwise AUC=0.904 / val 0.899**（旧 best val≈0.575 近随机）。数据集现 33 可行实例(train23/val5/test5)。
3. **4 个消融开关未生效 → 已接线**：`deployment_risk`(insert_scores 关掉 w3·Risk)、`fault_domain`(`_choose_active_replicas` 不再排除同域)、`hybrid`(`_repair_core` 强制全主动/全被动，`hybrid_mode`)、`dynamic_resched`(`recovery.simulate_fault` 限制为本域局部恢复，并经 `metrics.evaluate`→`verify` 透传保证搜索/验证一致)。`run_ablation` Hybrid 拆成 all-passive/all-active 两变体。
   - 验证：各 `w/o *` 数值与 `[Full]` 明显不同（可行率 Full 0.67 vs w/o GCN 0.5 vs w/o RecPenalty/DynResched 0.0 …）。
4. **量纲图跨规模污染 + safety_margin 为负 → 已修**：`aggregate.py` 增「仅可行解」聚合 `*_feas_mean/std/count`（剔除 INF/不可行 r_max）并新增 `summary_by_algo_scale_scenario.csv`。`plots.py` 重写：量纲指标按 scale 分组图、无量纲指标单图、新增「按任务数归一化成本」图、安全余量改用 `safety_margin_feas_mean`（图注「仅可行解」）、主对比图仅核心算法（排除 `[...]`）、消融/敏感性/预算各按 tag 出图。
   - `pytest tests/test_recovery_verifier.py` 仍 **4 passed**（dynamic_resched 默认开，行为不变）。

> **重跑进度（新代码，归档旧结果于 `results/archive_oldcode_*`）**：
> - ✅ `main_small`(45 实例,repeats10,iter1000,1260 runs)、`main_medium`(30,repeats10,iter1000,840 runs) 已完成。
> - ✅ `budget_{50,100,200,500,1000}`(small,repeats8,各720 runs) 已完成 → `fig_budget_convergence`：小预算下 GCN-ALNS 可行率更高、iter1000 成本更低(208.3<210.7)。
> - ✅ 消融 6/8 变体(small+medium,repeats5,iter1000) 已完成：Full / w/o GCN / w/o Recovery Penalty / w/o Fault-Domain / w/o Deployment Risk / w/o Dynamic Resched。
> - ⏳ **未完成**：消融剩 2 变体 `w/o Hybrid (all-passive)` 与 `(all-active)`（用户在 ~31h 后手动终止，正运行到此）；**large 规模(main+ablation) 尚未跑**（重计算，需确认降配/时机）。
> - 续跑方式：`run_ablation --instances data/instances/small data/instances/medium --repeats 5 --max-iter 1000`（同 tag 自动跳过已完成 6 变体，仅补跑 2 个 Hybrid），随后 `eval.aggregate` + `viz.plots`。
> - 注意本机长期被另一项目(DAGFleet)的 GPU/CPU 作业占用，重跑用 `--workers 8` 共存。

## 已知问题 / 待办决策
1. **数据规模**（已扩大）：现为 90 实例（small/medium/large 三规模），GCN 数据集 38 可行实例。tight（及多数 medium）场景对所有方法 infeasible —— 这是论文"baseline 失败"的预期；GCN 主要在 loose/medium 可行实例上学习。若需更多可行样本，可加大 `--n` 或放宽 loose 占比。
2. **主实验已全量**（已完成）：`main_small/medium` 用 `max_iter=1000, repeats=10`，`main_large` 用 `max_iter=800, repeats=5`（large 极慢，降配控时）。结果为报告级，已聚合出图。后台的 ablation/sensitivity/scalability 完成后会再整体重聚合。
3. **PEFT bug 已修**：`heft._greedy_eft_place` 原按优先级全局排序可能违反拓扑序导致 `KeyError`；已加 `_topo_priority_order`(Kahn + 按优先级选 ready)修复，对 HEFT/baseline 行为无影响。
4. **verify 性能**：纯 Python，50 任务约 0.2–0.4s/次、|Ω|≈130–210；ALNS 单次(50 迭代)约 5–7s，1000 迭代×10 重复×多实例会很慢。可选优化：向量化/增量重算、大规模对 Ω 采样（最终报告解仍需完整 verify）。
5. **消融开关接线**：~~仍只透传~~ **已全部接线**（见上「本轮修复」#3）；`run_ablation` 各变体结果与 Full 不同。
6. **torch-geometric 不被模型使用**：已安装但模型用自包含稠密 GCN（见阶段6）。若坚持用 PyG，需把 `model.py` 切回 PyG 卷积并**重训**（权重命名不同，旧 ckpt 不兼容）。

## 关键设计约定（务必沿用）
- 所有算法**必须**调用同一个 `core.recovery.verify` 判定 SLA 可行性。
- 链路参数由 `LinkFactory` 确定性重建（种子 2026 + 节点 id），JSON 不必存全部链路。
- 结构属性 `compute_structural` 在 `load_instance` 时重算（与部署无关）。
- 配置唯一来源 `config/default.yaml`，代码不要硬编码参数。
- 节点执行顺序 O = 各节点上任务的拓扑序（`base.topo_node_order`）。

## 如何在新 session 继续（给用户的提示词）
全链路阶段 0–9 已完成；主实验已出报告级结果，消融/敏感性/可扩展性在后台。新会话可发：

> 继续 gcn-alns 项目（e:\user\zhangyuxuan\gcn-alns\gcn-alns）。先读 `PROGRESS.md`。
> 检查后台实验（ablation/sensitivity/scalability）是否跑完，必要时用同 --tag 续跑，
> 然后重跑 `eval.aggregate` + `viz.plots` 出全量 CSV/图。运行用 `$env:PYTHONPATH="src"`，长任务放后台。

（新 session 会重新读取本文件恢复上下文。）

## 常用命令（本轮大规模，新机）
```powershell
$env:PYTHONPATH="src"
# 阶段1: 三规模数据（每个 scale 单独一次；--instances 后续支持多目录）
python -m gcnalns.data.instance_builder --out data/instances/small  --scale small  --n 45
python -m gcnalns.data.instance_builder --out data/instances/medium --scale medium --n 30
python -m gcnalns.data.instance_builder --out data/instances/large  --scale large  --n 15
# 阶段2: 建数据集(并行,多目录) + 训练(GPU)
python -m gcnalns.gcn.dataset --instances data/instances/small data/instances/medium data/instances/large --out data/gcn --alns-iter 200 --workers 30
python -m gcnalns.gcn.train --data data/gcn --out results/gcn_model.pt --device cuda:0
# 阶段3: 主实验(并行+续跑; 按规模分 tag, large 降配)
python -m gcnalns.experiments.run_main --instances data/instances/small  --out results --workers 30 --repeats 10 --max-iter 1000 --tag main_small
python -m gcnalns.experiments.run_main --instances data/instances/medium --out results --workers 30 --repeats 10 --max-iter 1000 --tag main_medium
python -m gcnalns.experiments.run_main --instances data/instances/large  --out results --workers 30 --repeats 5  --max-iter 800  --tag main_large
# 阶段3: 消融 / 敏感性 / 可扩展性
python -m gcnalns.experiments.run_ablation     --instances data/instances/small data/instances/medium --out results --workers 30 --repeats 8
python -m gcnalns.experiments.run_sensitivity  --instances data/instances/small --out results --workers 30 --repeats 5
python -m gcnalns.experiments.run_scalability  --instances data/instances/small data/instances/medium data/instances/large --out results --workers 30 --repeats 3 --max-iter 800
# 阶段4: 聚合 + 绘图
python -m gcnalns.eval.aggregate --raw results/raw_json --out results/csv
python -m gcnalns.viz.plots --csv results/csv --out results/figures
# 续跑：同 --tag 重跑会自动跳过已完成项 (results/raw_json/<tag>.jsonl)
```
