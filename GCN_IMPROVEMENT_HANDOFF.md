# GCN 增益放大 + tight 可行性攻坚 —— 续作交接 (Handoff)

> 本文件用于跨 session 续作，记录"让 GCN 从只降成本 → 影响可行率"与"攻克 tight 不可行"这一轮改造。
> 主线进度见 `PROGRESS.md`；本轮计划见 `~/.cursor/plans/gcn-impact-and-tight_7df74ef7.plan.md`（**不要编辑**）。
> 运行统一 `$env:PYTHONPATH="src"`（Windows PowerShell）。

## 一句话状态
**代码改造已全部完成并端到端验证通过**（Part A/B/C/D，10 个 todo 全绿，无 lint 错误）。
现有模型 `results/gcn_model.pt` 是**旧模型（无 mode 头）**，因此 **A1 热启动 / A3 定向 destroy / B 系算法改造已生效，A2（GCN 驱动容错模式/副本）需重训后才启用**。
**尚未做**：重建数据集（带 mode 标签）→ 重训 → tight 重跑主对比/消融出报告级结果。

## 本轮要解决的两个问题（用户诉求）
1. **GCN 作用太小**：旧实现 GCN 只在 `insert_scores` 里给"主节点"重排序，结构上只能降成本（消融 w/o GCN 可行率几乎不变、成本仅 -4.5%）。
2. **tight 基本不可行**：`tight={alpha:1.15, beta:0.05}`，恢复预算 `R_max=0.05·T_ref` 极小，可行域近乎为空；旧 ALNS 软惩罚目标在预算内命中不了。

## 本轮改动（已完成、已验证）

### Part A：扩大 GCN 参与面
- **A1 GCN 贪心热启动** —— `src/gcnalns/algorithms/gcn_alns.py`
  - 新增 `_gcn_greedy()`：用带 `gcn_score`/`gcn_mode` 的 `Repairer` + `operators._repair_core`（全部任务当 removed、`_score_insert`）一次性构造 GCN 引导初始解；`_candidate_inits` 把它插到候选首位。
  - 重构 GCN 钩子：`_ensure_hooks()` 一次前向拿到 `(score, mode)` 两个钩子并缓存，`gcn_infer_time` 计时；`_gcn_hook`/`_gcn_mode_hook` 分别返回。
- **A2 容错模式/副本引入 GCN** —— `src/gcnalns/algorithms/alns/operators.py`
  - `_repair_core` 模式决策：`active = recrisk>budget`，**并在 mode 头预测 P(active)≥0.5 时置 active**。
  - `_choose_active_replicas` 选点排序键加入 `gcn_score(i,k)` 先验。
  - 均受 `ablation["gcn_mode"]` 开关控制；`Repairer.__init__` 新增 `gcn_mode` 形参（`src/gcnalns/algorithms/alns/insertion.py`）。
- **A3 定向 destroy** —— `operators.py` + `src/gcnalns/algorithms/alns/solver.py`
  - 新增 `low_gcn_removal`（移除当前落在低 GCN 分节点上的任务），注册进 `DESTROY_OPS`。
  - 所有 destroy 算子签名加 `gcn_score=None`；solver 仅在**有 GCN 且 `gcn_destroy` 开**时把 `low_gcn` 加入算子池（**plain ALNS 行为完全不变**）。
- **A4 调权重** —— `config/default.yaml`：`insert_cost.w6` 3.0 → **0.5**（敏感性最优）。

### Part B：tight 可行性（纯算法，GCN 热启动为其加速）—— `solver.py`
- **B1 可行性优先**：`_quick_eval` 增返回违约量 `V = max(0,cmax-D)+max(0,est_r-r_eff)`；新增 `_search_objective()`：**找到可行解前用"仅 V（成本 1e-4 微权重破平）"，命中后切回 cost+penalty**。搜索循环里 `cur/best` 改为存 `(pen, cost, viol)` 分量，每轮按 `have_feasible` 现算搜索值。
- **B2 停滞重启**：`no_improve_stop` 触发时不再直接 break，而是 `_restart_solution()`（对 best 做 50% 大破坏再 `repair_insert`）+ 重置温度 `T=T0`，`alns.restarts` 次（默认 **2**，见 config）。
- **B3 初始解多样化**：`_initial_solution()` 在 CriticalityHybrid 外加 **HEFT-fastest**，按 `(可行?, 违约V, 目标)` 取优。

### Part C：mode 头 + 数据（支撑 A2）
- `src/gcnalns/gcn/dataset.py`：`InstanceSample` 增 `pos_mode`；`build_instance_sample` 为每个正样本记录 FT 模式标签（取最大权重解的 active/passive）。
- `src/gcnalns/gcn/model.py`：`GCNScorer` 增 `mode_mlp` 第二头 + `mode_pairs()`（预测 P(active)）。
- `src/gcnalns/gcn/train.py`：`_sample_pairs`/`_instance_loss` 透传 mode 标签，加 **mode-BCE 损失**（权重 `gcn.mode_loss_weight=0.5`）；ckpt 存 `has_mode_head=True`。
- `src/gcnalns/gcn/infer.py`：`load()` 用 `strict=False` + 读 `has_mode_head`（**旧 ckpt 无 mode 头自动降级**）；新增 `make_hooks()` 一次前向返回 `(gcn_score, gcn_mode)`（无 mode 头时 `gcn_mode=None`）。

### Part D：消融归因 —— `src/gcnalns/experiments/run_ablation.py`
- `VARIANTS` 新增 3 个变体：`w/o Warm Start`(`{"warm_start":False}`)、`w/o GCN Mode`(`{"gcn_mode":False}`)、`w/o GCN Destroy`(`{"gcn_destroy":False}`)，用于逐点归因 GCN 贡献。

## 新增/变更的配置项（`config/default.yaml`）
- `insert_cost.w6: 0.5`（原 3.0）
- `alns.restarts: 2`（新增）
- `gcn.mode_loss_weight: 0.5`（新增）
- 新增消融开关（默认全开，经 `ablation` dict 传入）：`warm_start` / `gcn_mode` / `gcn_destroy`。

## 验证记录（本轮，均通过）
- 9 个改动文件 **0 lint 错误**。
- 冒烟：plain ALNS 与 GCN-ALNS 端到端跑通；4 个新/旧消融开关（warm_start/gcn_mode/gcn_destroy/gcn off）均正常。
- **关键佐证（同一 small loose 实例, seed=1, iter=400）**：
  - **GCN-ALNS 找到可行解**（r_max 7.57 ≤ r_eff 8.84），**plain ALNS 不可行**（r_max 10.95）→ GCN 首次直接影响可行率。
  - tight 实例仍不可行（r_eff≈1.9s，β=0.05 本就极端），但 GCN-ALNS 更接近；算法无报错、重启/可行性优先路径正常。
  - 注：以上用**旧模型（无 mode 头）**，A2 未启用，收益预计更大。
- mode 头链路小数据验证：`pos_mode` 标签正确（loose 多为 passive、SIPHT 有 4 个 active）、训练带 mode 损失、`has_mode_head` 存取正常、reload 后 `gcn_mode` 钩子输出真实概率、placement AUC 4 epoch 即 0.87。

## 接下来要干什么（按顺序）
> 重计算步骤，本轮**未代跑**（机器与算力由你安排；注意本机可能被 DAGFleet 占用，可 `--workers 8` 共存）。

1. **重建 GCN 数据集（带 mode 标签 + 满迭代 teacher）**
   ```powershell
   $env:PYTHONPATH="src"
   python -m gcnalns.gcn.dataset --instances data/instances/small data/instances/medium --out data/gcn --workers 8
   ```
   - 去掉旧的 `--alns-iter 200` 上限（用满迭代 ALNS 作 teacher，标签更强）。
   - 旧 `data/gcn/dataset.pkl` 无 `pos_mode` 字段，**必须重建**，否则 mode 头只会学到全 passive。

2. **重训（激活 A2）**
   ```powershell
   python -m gcnalns.gcn.train --data data/gcn --out results/gcn_model.pt --device cuda:0
   ```
   - 训练日志应出现 mode-BCE 生效、ckpt 带 `has_mode_head=True`；确认 test placement AUC 仍高。
   - 训完后 `GCNScorerRunner.load` 会自动启用 `gcn_mode` → A2 生效。

3. **tight 重跑主对比 + 消融归因**
   ```powershell
   # 主对比（重点看 tight 场景可行率是否提升）
   python -m gcnalns.experiments.run_main --instances data/instances/small  --out results --workers 8 --repeats 10 --max-iter 1000 --tag main_small
   python -m gcnalns.experiments.run_main --instances data/instances/medium --out results --workers 8 --repeats 10 --max-iter 1000 --tag main_medium
   # 消融（含 3 个新变体 w/o Warm Start / GCN Mode / GCN Destroy）
   python -m gcnalns.experiments.run_ablation --instances data/instances/small data/instances/medium --out results --workers 8 --repeats 5 --max-iter 1000
   # 聚合 + 出图
   python -m gcnalns.eval.aggregate --raw results/raw_json --out results/csv
   python -m gcnalns.viz.plots --csv results/csv --out results/figures
   ```
   - 同 `--tag` 自动跳过已完成项（`results/raw_json/<tag>.jsonl` 断点续跑）。
   - 建议先归档旧结果：`results/archive_oldcode_*`。

4. **验证/复盘要点**
   - 主对比：**tight 场景 GCN-ALNS 可行率**是否较旧结果上升；GCN-ALNS vs ALNS 在可行率/成本的差距是否拉大。
   - 消融：`w/o Warm Start` / `w/o GCN Mode` / `w/o GCN Destroy` 相对 Full 的可行率/成本落差，用来**逐点归因 GCN 现在的贡献**（回答"GCN 作用太小"）。
   - 更新 `results/REPORT.md` 与 `PROGRESS.md`。

## 已知问题 / 注意事项
- **tight 本质极难**：`beta=0.05` 使 r_eff 常仅 ~1-2s，被动恢复（detect1.0+replan0.5+comm+重执行）几乎必然超预算。若重训+A2 后 tight 仍近 0，属场景刻意设成的压力上界；可考虑（另开讨论）加一档"难但可解"的 tight（如 `beta 0.05→0.10`）或做 alpha/beta 可行性前沿扫描。
- **plain ALNS 向后兼容**：`low_gcn` 仅 GCN 且开关开时进入算子池；不含 GCN 的运行行为不变（无需重跑纯 baseline 也可，但主对比会一起重跑）。
- **旧 ckpt 兼容**：`infer.load` 用 `strict=False`，旧模型可直接加载但 `gcn_mode=None`（A2 静默关闭），不会报错。
- **`gcn_infer_time` 冒烟里偏大（~6.8s）**：包含首次 torch import + 模型 load；实际并行 worker 里 torch 已预热，开销回落（旧报告 ~0.58s/实例）。
- 关键设计约定不变：所有算法仍统一走 `core.recovery.verify` 判可行；`config/default.yaml` 为参数唯一来源。

## 本轮改动文件清单
```
config/default.yaml                              # w6=0.5, alns.restarts, gcn.mode_loss_weight
src/gcnalns/algorithms/alns/solver.py            # B1/B2/B3 + GCN 钩子/destroy 透传
src/gcnalns/algorithms/alns/operators.py         # A2(mode/replica) + A3(low_gcn) + destroy 签名
src/gcnalns/algorithms/alns/insertion.py         # Repairer 增 gcn_mode 形参
src/gcnalns/algorithms/gcn_alns.py               # A1 热启动 + make_hooks 缓存
src/gcnalns/gcn/dataset.py                        # pos_mode 标签
src/gcnalns/gcn/model.py                          # mode_mlp 第二头 + mode_pairs
src/gcnalns/gcn/train.py                          # mode-BCE 损失 + has_mode_head
src/gcnalns/gcn/infer.py                          # make_hooks + strict=False 加载
src/gcnalns/experiments/run_ablation.py           # 3 个新消融变体
```
