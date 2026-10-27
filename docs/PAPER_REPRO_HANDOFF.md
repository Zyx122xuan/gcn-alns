# 论文复现交接 (Paper Reproduction Handoff)

> 目的:把 3 篇 CCF-B 论文复现为 GCN-ALNS 的对比基线,在**同一 `core.recovery.verify` 闸门**下与 GCN-ALNS/ALNS 公平比较。
> 复现代码放在**新包 `src/gcnalns/reproductions/`**(不要塞进 `algorithms/`)。
> 运行统一 `$env:PYTHONPATH="src"`(Windows PowerShell)。主项目进度见 `docs/PROGRESS.md` / `docs/GCN_IMPROVEMENT_HANDOFF.md`,实验报告见 `results/REPORT.md`(正文+附录 A/B)。

## 0. 范围
复现 3 篇(均已核实 = **IEEE TSC / CCF-B**);SRDS'25 GCN+DRL 已**放弃**(实为 workshop 论文、非 CCF-B、无容错)。
1. **RECR-D**(首实现,最对口) 2. **HFWPF / Hybrid-FT**(次实现) 3. **MTGP**(契合度低,建议用 GA/NSGA-II 平替,或单独立项)。

学习式基线若之后要补,选 **CCF-A TPDS'24 (Jayanetti DRL)** 或 CCF-C 但相关的 **TCC'24 (GNN+PPO)**——本轮不做。

---

## 1. 目标问题回顾(你的框架,所有基线必须适配到它)
- 单实例**静态**部署:给定固定 **cloud/edge/device 连续体**(节点集固定)+ 一个 DAG 工作流 + SLA(`D=α·T_ref`, `R_max=β·T_ref`)。
- 求 **placement + 容错模式**,最小化成本,可行性由 `core.recovery.verify` 判定(deadline + 最坏情况恢复时间 ≤ R_max;故障集 Ω = task∪node∪link)。
- 容错原语:**active backup(主动副本,m_max 个,故障域隔离)** / **passive(被动重算/reactive recovery)**。
- 场景:tight(α1.15/β0.05)、medium(α1.30/β0.10)、loose(α1.50/β0.20);难档 tight2/tight3 见 config。
- 工作流类型:Montage, CyberShake, Epigenomics, LIGO, SIPHT。规模 small/medium/large=50/150/500。

## 2. 统一"公平对比协议"(务必遵守)
把每篇论文**只复现其"选节点 + 选容错策略"的调度逻辑**,产出一个 `Solution(placement, mode)`;**最终一律经你的 `compute_schedule` + `verify` + `compute_cost` 评估**(和 GCN-ALNS 同闸门)。需裁剪/对齐的维度:

| 维度 | 你的框架 | 对齐方式 |
|---|---|---|
| 能量(RECR-D 有) | 无 | 丢弃(RECR-D 惩罚权重 α_energy=0) |
| 执行时间不确定(HFWPF 有) | 确定 exec | 用确定 exec;HFWPF 的 slack 判据改用确定时间 |
| 在线到达/多工作流(HFWPF/MTGP) | 单实例静态 | 单实例一次性调度 |
| 目标函数 | cost s.t. verify 可行 | 各法产出 placement+mode → 你的 verify 判可行 + 你的 cost |
| 容错原语 | active / passive backup | replication/复制→active;resubmission/重执行→passive |

**在 `results/REPORT.md` 里必须诚实标注**:这些是"忠于原文调度/容错决策逻辑、但去掉其在线/不确定性/能量外壳"的**静态适配版**基线,而非逐比特复现。

---

## 3. 仓库接口速查(实现基线只需这些)
- 基类:`src/gcnalns/algorithms/base.py` → `class Scheduler(cfg, seed)`,实现 `schedule(instance) -> Solution`;可设 `self.runtime`。
- 解:`src/gcnalns/core/model.py` → `Solution`:
  - `placement: dict[task_id -> list[node_id]]`(len==1 被动;==副本数 主动)
  - `mode: dict[task_id -> bool]`(True=active backup)
  - `order`(可留空,评估时按 DAG 拓扑序自动推导)、`clone()`、`is_active()`、`replicas()`。
- 复用助手:
  - `algorithms/heft.py::_greedy_eft_place(instance, factory, task_order, extra_metric=None) -> Solution`(贪心 EFT 放置,返回 placement;`extra_metric(i,k,eft)` 可加选点偏好项)。
  - `algorithms/replicas.py::add_active_replicas(instance, sol, tasks:set, m, cfg)`(把 tasks 提升为主动副本,故障域隔离、偏可靠高容量节点)。
  - `data/links.py::get_factory(instance, cfg)`(链路工厂,选点算通信要用)。
  - `core/dag.py::compute_structural(workflow, continuum, cfg, factory)`(算 ranku/slack/criticality;放置前先调)。
  - `core/schedule_eval.py::compute_schedule`、`core/recovery.py::verify` / `effective_budget` / `estimate_rmax`、`core/cost.py::compute_cost`。
- 参数唯一来源:`config/default.yaml`(不要硬编码)。
- **注册**:`algorithms/registry.py::get_scheduler(name)`。新基线用**懒加载**方式登记(仿照 ALNS/GCN-ALNS 分支),从新包导入,避免污染 baselines 导入链。例如:
  ```python
  if name == "RECR-D":
      from ..reproductions.recr_d import RECRD
      return RECRD
  ```
  同时把名字加进 `run_main` 的 `DEFAULT_ALGORITHMS`(在 `experiments/runner.py`)或用 `--algorithms` 显式指定。
- 运行/评估链:`experiments/run_main.py`(已支持 `--algorithms/--instances/--repeats/--max-iter/--tag/--workers/--model-path`);评估在 `eval/metrics.py::evaluate`(自动过 verify+cost);聚合 `eval/aggregate.py`(summary 只统计 main_* tag);出图 `viz/plots.py`。
- 一个可照抄的最简基线样板:`algorithms/baselines.py::_PassiveBase`(结构:`get_factory`→`compute_structural`→`_greedy_eft_place`→设 `mode`→计时)。

## 4. 新包结构建议
```
src/gcnalns/reproductions/
  __init__.py
  recr_d.py        # RECR-D
  hfwpf.py         # Hybrid-FT 静态核心
  ga_placement.py  # (可选) GA/NSGA-II 平替 MTGP
  README.md        # 简述每个基线的适配裁剪与出处
```
`registry.get_scheduler` 增加对 "RECR-D" / "HFWPF" / "GA-Placement" 的懒加载分支。

---

## 5. 逐篇复现规格

### 5.1 RECR-D —— 首实现
- **书目**:A. Taghinezhad-Niar, J. Taheri, "Fault-Tolerant Cost-Efficient Scheduling for Energy and Deadline-Constrained IoT Workflows in Edge-Cloud Continuum," **IEEE TSC 18(5):2892–2903, 2025**, DOI:10.1109/TSC.2025.3599497。**CCF-B**。开放获取预印本可得(会话中已读全文)。
- **核心算法**(全文伪代码 Alg.1–3 已获取):
  1. 约束分配(Alg.1):前/后向遍历算 EST/EFT/LST/LFT;动态平衡因子 `DB` 把全局 deadline 按 EFT/LFT 凸组合分到每任务子 deadline。(能量分配部分**丢弃**。)
  2. RECR-D 主体(Alg.2):任务按 **EDF**(子deadline 早先)排序;对每任务遍历所有资源,算归一化惩罚
     `PN = ζ·cost + α·energy + β·deadline + γ·reliability`(式43),选 PN 最小的资源;若违反 deadline/可靠性 → 调 TRM。
     - 归一化各惩罚:deadline 违约 `max(0,(EFT−D_task)/D_task)`;cost `MC(i,j)/(MC_max+1)`;reliability `max(0,Θ−R)`。
  3. TRM(Alg.3):为任务选一个**备份资源(主动复制)**降低有效故障率,更新惩罚,择优。
- **适配到你的框架**:
  - **能量项丢弃**(α=0);`cost` 惩罚用你的 `compute_cost` 分量或 EFT 近似;`deadline` 用你的 `D=α·T_ref` 的每任务子 deadline;
  - **可靠性口径替换**:不要用它的 CTMC 成功概率做判定——TRM 触发条件改为"该放置在你的 `verify`/`estimate_rmax` 下 recovery 超预算";TRM 的"加备份"= `add_active_replicas(tasks={i}, m=m_max)`,即 `mode[i]=True`。
  - 产出 `Solution(placement, mode)` 后过 `verify`。
  - 复杂度 O(N³)/workflow,small/medium 可接受;large 需限制候选资源数。
- **默认参数**(原文):λ=1e-4, µ=5e-4, δ=1e-5;Θ=0.9;权重 ζ/β/γ 需在你场景下轻调(能量 α=0)。原文对比 Hybrid-MCD/MSTEC/HREC——**不用**,你只需把 RECR-D 当一个基线。

### 5.2 HFWPF (Hybrid-FT) —— 次实现
- **书目**:Qian Ren, Guangshun Yao, "A Hybrid Fault-Tolerant Workflow Scheduling With Performance Fluctuated Cloud Resources," **IEEE TSC 19(1):44–57, Jan/Feb 2026**(线上 2025-11-21), DOI:10.1109/TSC.2025.3635177。**CCF-B**。
- **全文 PDF 在**:`d:\zyx\paper\A_Hybrid_Fault-Tolerant_Workflow_Scheduling_With_Performance_Fluctuated_Cloud_Resources.pdf`(会话中已读全文)。
- **原文 4 阶段**:Preprocessing(切子 deadline)→ Initial Scheduling(选容错策略+选 VM)→ Online Scheduling(故障重执行)→ Online Adjustment(slack 回填、replication→resubmission 回切);弹性 VM 供给。
- **只复现"静态核心"**(去掉在线/不确定性/弹性无限 VM):
  - **容错策略选择(式8)**:任务 `tij` 若 `slack = dl_ij − st_ij ≥ 2·(Tboot + max_pred(TT) + ψ(i))` → **resubmission = passive**(`mode[i]=False`,单副本);否则 **replication = active**(`mode[i]=True`,2 副本、不同节点)。
    - `ψ(i)` = 该任务在各节点上的最小(确定)执行时间;`Tboot` 可取 0 或一个小常数(你的模型无 VM 启动,建议 Tboot=0 并在 README 注明)。
    - `st_ij/dl_ij` 用你的每任务子 deadline(可复用 RECR-D 的 DB 分配,或用 EST/LFT)。
  - **选点**:满足时序可行的候选节点里,replication 选**容量最小**(省成本)、resubmission 选**容量最大**(留余量);用 `_greedy_eft_place` + `extra_metric` 或自写选点。active 副本用 `add_active_replicas(m=m_max)`。
  - 产出 `Solution` 过 `verify`。
- **默认参数**(原文):Tboot=15s(你可置 0)、NB=10Gbps、FR 7.5%、β 方差 0.3、α∈{0.5–2.5};真实工作流含 Montage/CyberShake/Epigenomics/Inspiral(=LIGO)。**报告需注明:去掉了其执行时间不确定性(论文核心),故其相对 GCN-ALNS 的优势会被削弱,属预期。**

### 5.3 MTGP —— 建议 GA 平替
- **书目**:Meng Xu, Yi Mei, Shiqiang Zhu, Beibei Zhang, Tian Xiang, Fangfang Zhang, Mengjie Zhang, "Genetic Programming for Dynamic Workflow Scheduling in Fog Computing," **IEEE TSC 16(4):2657–2671, 2023**, DOI:10.1109/TSC.2023.3249160。**CCF-B**。公开代码(Java/ECJ):https://github.com/MengXu95/Fog-Computing
- **问题差异大**:动态到达 + makespan 最小化 + **无容错/无 SLA 可行性**;直接比不公平。
- **两条路**(择一,建议 A):
  - **A(推荐,省事且公平)**:在新包实现一个 **GA/NSGA-II 放置基线 `ga_placement.py`**——个体=每任务(节点选择 + active/passive 位),适应度=你的 `compute_cost` 且以 `verify` 可行为硬约束(不可行重罚),作为"进化式元启发式 vs 你的 ALNS/GCN-ALNS"的对照。报告中说明"以标准 GA 代表进化式方法(MTGP 原问题为动态 makespan,不可直接比)"。
  - **B(忠于原文,成本高)**:把 MTGP 的 GP terminals/fitness **重定义**到你的静态容错问题并重新进化(超启发式训练),单独立项,工作量大。

---

## 6. 实施顺序 & 验收
1. 建 `reproductions/` 包 + registry 懒加载 + 加入 `DEFAULT_ALGORITHMS`(或用 `--algorithms` 指定)。
2. **RECR-D** → 小规模冒烟:
   ```powershell
   $env:PYTHONPATH="src"
   python -m gcnalns.experiments.run_main --instances data/instances/small --algorithms RECR-D --repeats 1 --max-iter 50 --tag smoke_recrd --workers 4
   ```
   确认产出 `Solution` 过 verify、有可行解、成本合理(与现有 baselines 同量级)。
3. **HFWPF** → 同样冒烟。
4. **GA-Placement**(或 MTGP)→ 冒烟。
5. 全量对比:把 RECR-D/HFWPF/GA 加入主对比,与 GCN-ALNS/ALNS/现有 baselines 同跑(small/medium 各 repeats 视时间;large 极慢,可选)。评估→`eval.aggregate`→`viz.plots`。
6. 在 `results/REPORT.md` 增一节"外部基线对比",报告可行率/成本/安全余量,并**如实写明各基线的静态适配裁剪**。

**验收标准**:每个基线都能被 `run_main` 调用、产出经 `verify` 判定的解、数值合理、无 lint 错误;报告有诚实的裁剪说明。

## 7. 注意事项
- 所有算法必须走同一 `core.recovery.verify`;参数只从 `config/default.yaml` 取。
- 默认 GCN 模型 = `results/gcn_model.pt`(已是 A2 修复版,mode AUC 0.975)。
- `eval/aggregate.py` 的 summary 只统计主对比 tag(`main_small/medium/large`);新基线要进主对比图,请用 `main_*` tag 或相应调整(见 `aggregate._MAIN_TAGS`)。
- 机器:32 核 + 1×A40 + ~64GB RAM;large 单解 ~4.5h,谨慎;可能与 DAGFleet 共存,用 `--workers 8~16`。
- 给下一个会话的起手提示:
  > 读 `docs/PAPER_REPRO_HANDOFF.md`。在新包 `src/gcnalns/reproductions/` 实现 RECR-D、HFWPF(静态核心)、GA 放置基线,注册进 registry,各自小规模冒烟过 `core.recovery.verify`,再并入主对比。运行用 `$env:PYTHONPATH="src"`,长任务放后台。
