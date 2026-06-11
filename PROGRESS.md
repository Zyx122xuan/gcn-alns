# GCN-ALNS 实现进度与续作交接 (Handoff)

> 本文件用于跨 session 续作。记录已完成内容、当前状态、剩余工作、已知问题与恢复方式。
> 计划文件位于 `~/.cursor/plans/gcn-alns-experiment-plan_6de824b3.plan.md`（**不要编辑**）。

## 一句话状态
论文 RLC-HFTSP / GCN-ALNS 实验平台：**阶段 0–9 全部完成并端到端验证**。
阶段 6（GCN 训练）、7（GCN-ALNS）、8（实验编排+指标 CSV）、9（绘图）已实现并跑通：
数据集构建→训练→GCN-ALNS 求解→主实验 CSV→18 张图全链路验证通过。
主实验结果（小数据集、降迭代验证版）：**仅 ALNS / GCN-ALNS 达成 SLA 可行**（GCN-ALNS 可行率与成本均略优于 ALNS），所有启发式 baseline 可行率为 0，符合论文预期。

## 技术栈与环境
- Python **3.8**（用户机为 miniconda，`D:\miniconda3`，cp38），运行需 `PYTHONPATH=src`。
- 已安装：numpy, pandas, pyyaml, networkx, matplotlib, scipy, tqdm, pytest。
- **已安装 torch 2.4.1+cpu + sympy/filelock/mpmath**（Adam 优化器依赖 sympy）。`torch-geometric 2.6.1` 也已装，但**模型不使用它**（见下）。
- Windows PowerShell，shell 命令用 `$env:PYTHONPATH="src"; python ...`。
- 注意：本机网络到 pythonhosted 不稳定，pip 常超时；torch 主 wheel 已缓存，必要时用 `--no-deps` + 单独装依赖、`--timeout 300 --retries 20`。
- 注意：matplotlib **首次** import 会构建字体缓存，可能很慢（数十分钟），之后正常。

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

## 已知问题 / 待办决策
1. **数据规模偏小**：现有 `data/instances/` 仅 15 个 small/medium 实例，且 medium/tight 对所有方法都 infeasible（连 ALNS 也找不到可行解，可能 SLA 偏紧），dataset 只剩 **4 个 loose 实例**(train3/val1/test0)。模型与主实验结果仅用于**验证全链路**，非最终报告级。**下一步**应批量生成更大、更可行的训练/测试集（`instance_builder --scale/--load`，或放宽 loose 占比），再重训重跑。
2. **主实验是降配验证版**：上面 CSV/图是 `max_iter=50, repeats=2` 跑的（机器算力有限，全 1000 迭代×10 重复会很慢）。最终报告用 `python -m gcnalns.experiments.run_main`（默认 config 全量），建议放后台跑。
3. **PEFT bug 已修**：`heft._greedy_eft_place` 原按优先级全局排序可能违反拓扑序导致 `KeyError`；已加 `_topo_priority_order`(Kahn + 按优先级选 ready)修复，对 HEFT/baseline 行为无影响。
4. **verify 性能**：纯 Python，50 任务约 0.2–0.4s/次、|Ω|≈130–210；ALNS 单次(50 迭代)约 5–7s，1000 迭代×10 重复×多实例会很慢。可选优化：向量化/增量重算、大规模对 Ω 采样（最终报告解仍需完整 verify）。
5. **消融开关接线**：`gcn`、`recovery_penalty` 已在 ALNS 核心生效；`fault_domain`/`deployment_risk`/`dynamic_resched`/`hybrid` 仍只透传未在 `insertion.py`/`operators.py`/`recovery.py` 内接线（`run_ablation` 会跑这些变体但结果≈Full）。需要严格消融时补接线。
6. **torch-geometric 不被模型使用**：已安装但模型用自包含稠密 GCN（见阶段6）。若坚持用 PyG，需把 `model.py` 切回 PyG 卷积并**重训**（权重命名不同，旧 ckpt 不兼容）。

## 关键设计约定（务必沿用）
- 所有算法**必须**调用同一个 `core.recovery.verify` 判定 SLA 可行性。
- 链路参数由 `LinkFactory` 确定性重建（种子 2026 + 节点 id），JSON 不必存全部链路。
- 结构属性 `compute_structural` 在 `load_instance` 时重算（与部署无关）。
- 配置唯一来源 `config/default.yaml`，代码不要硬编码参数。
- 节点执行顺序 O = 各节点上任务的拓扑序（`base.topo_node_order`）。

## 如何在新 session 继续（给用户的提示词）
全链路阶段 0–9 已完成。后续主要是**扩大数据规模并跑全量实验**。新会话可发：

> 继续 gcn-alns 项目（D:\ProgramData\graduate\gcn-alns）。先读 `PROGRESS.md`。
> 批量生成更大可行的训练/测试实例，重建 GCN 数据集→重训→跑全量 run_main/ablation/
> sensitivity/scalability→出图。运行用 `$env:PYTHONPATH="src"`，长任务放后台。

（新 session 会重新读取本文件恢复上下文。）

## 常用命令
```powershell
$env:PYTHONPATH="src"
# 数据 / 验证器
python -m gcnalns.data.instance_builder --out data/instances --n 20 --scale small --load medium
python -m pytest tests/test_recovery_verifier.py -p no:cacheprovider -q
# 阶段6: 建数据集 + 训练（--alns-iter 限迭代加速建集）
python -m gcnalns.gcn.dataset --instances data/instances --out data/gcn --alns-iter 150
python -m gcnalns.gcn.train --data data/gcn --out results/gcn_model.pt
# 阶段7/8: 主实验（默认全量；验证可降 max_iter/repeats，放后台）
python -m gcnalns.experiments.run_main --instances data/instances --out results
python -m gcnalns.experiments.run_ablation --instances data/instances --out results
python -m gcnalns.experiments.run_sensitivity --instances data/instances --out results
python -m gcnalns.experiments.run_scalability --generate --out results
# 阶段9: 绘图（首次 matplotlib 建字体缓存会慢）
python -m gcnalns.viz.plots --csv results/csv --out results/figures
```
