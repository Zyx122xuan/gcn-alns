# 独立协议全流程重跑队列（2026-10-27）

用户已授权 teacher 重建完成后，继续训练、验证、正式主对比及补充实验，无需再次确认。后台接续由 `scripts/rerun_independent_v1.py` 执行；当前 teacher 仍是已经启动的原任务，不会重复启动。

## 查看运行状态

- `results/independent_v1/pipeline_state.json`：协调器 PID、当前阶段、心跳、子进程 PID 或 teacher 进度。
- `results/independent_v1/logs/teacher_build.log`：现有 teacher 日志（Windows PowerShell 的 UTF-16 编码已兼容）。
- `results/independent_v1/logs/pipeline_stdout_v2.log` / `pipeline_stderr_v2.log`：当前协调器总日志。
- `results/independent_v1/logs/<阶段>.log`：每步子进程日志。
- `results/independent_v1/stages/*.json`：阶段签名、完成状态和产物哈希。
- `results/independent_v1/pipeline_plan.json`：在测试求解前登记的范围、预算和模型选择规则。

## 执行顺序

1. 等待现有 teacher 文件完整写入，检查协议、75 个 train/val 输入、固定角色、完整 ALNS 预算与源码来源；train/val 必须都有可行示范，test 样本必须为空。核验结果写入 `logs/teacher_build_status.json`。
2. 从同一份新数据分别训练 A2-on（mode loss=0.5）和 A2-off（mode loss=0），自动选用可用的 CUDA，否则 CPU。每个模型完整执行默认最多 100 epoch / early stopping，不使用旧模型权重。
3. 在固定 15 个验证实例上评估 ALNS 和两个 GCN 模型，各 5 次重复、1000 次迭代。模型按预登记顺序选择：可行运行数更高者；持平时比较相同实例/随机重复且双方均成功的相对成本；仍持平选 A2-on。`w6=0.5` 等其余求解参数沿用固定配置。选择记录写入 `model_selection.json`，选中模型固定到 `gcn_model.pt`，之后不根据 test 调整模型或参数。
4. 三规模主对比，随后消融、预算、模型 A/B，最后聚合、作图和生成新报告。

## 预登记的正式测试范围

- 主对比：全部 14 算法，small（45 实例）/medium（30 实例）均为 repeats=10、iter=1000；large（15 实例）为 repeats=5、iter=800。分别为 2250 / 1500 / 450 条，共 **4200 条**。
- 消融：small+medium，14 变体，repeats=10、iter=1000，共 **10500 条**。
- 预算：small，ALNS/GCN-ALNS，50/100/200/500/1000 五档，repeats=10，共 **4500 条**。
- 模型 A/B：small，新训 A2-off、新训 A2-on、ALNS，repeats=10、iter=1000，共 **1350 条**。
- 正式测试共 **20550 条**；验证另有 225 条。该队列不新增 hard-tier 或其他未登记的实验。
- 普通阶段使用 8 个 worker，large 主实验使用 12 个 worker。每个 worker 限制 BLAS/torch CPU 线程，阶段之间顺序执行。

为沿用各规模原有预算，实验入口新增 `--instance-scales small medium` 等过滤。它只从原冻结 split 中选择子集，保留原协议 SHA，模型来源和 test 隔离检查继续生效。该选择进入实验 manifest，因此改变规模选择不能用同一 tag 续跑。

## teacher 运行期间的源码记录

排队实现仅改动了 `experiments/runner.py`、`run_scalability.py`、`run_sensitivity.py`，用于规模筛选和透传参数；这三处不参与 teacher 的示范生成。现有 teacher 的 core/data/gcn/ALNS 数学计算源码没有改变。接续器会比较开始与结束快照，除这三处未使用的实验入口外，任何其他 teacher 源码变化均拒绝继续；差异明细会保存在 teacher 审计记录中。

当前计划形成前的初始预检快照位于 `plans/initial_preflight_utf8_only.json`，只对应日志编码兼容修正前的预检，没有训练或正式测试结果与之关联。实际执行以根目录 `pipeline_plan.json` 为准。

## 续跑与失败处理

协调器持有 Windows OS 文件锁，重复启动会被拒绝。进程异常退出后锁由操作系统释放。手动恢复：

```powershell
$env:PYTHONPATH="src"
python -u scripts/rerun_independent_v1.py --workers 8 --large-workers 12
# 或：
powershell -File scripts/run_independent_pipeline.ps1
```

完成的阶段先核对产物哈希再跳过；中断的实验通过原 runner 的版本检查和 JSONL 续跑。任何命令失败、来源不符或产物变更都会停止队列并更新 `pipeline_state.json`，不会跳过失败阶段进入正式测试，也不会自动降预算或改协议。

原始指标、manifest 和部署解集中保存在 `results/independent_v1/test/`。全部阶段完成后，新报告输出到 `results/independent_v1/REPORT.md`，图表位于 `test/figures/`，摘要位于 `summary.json`。配对统计先按实例/随机重复对齐，再以实例为独立统计单位；消融的各指标分别对变体比较做 Holm 校正。

代码验证：**48 tests passed**，覆盖恢复验证器、隔离与版本续跑、规模筛选、验证集模型选择、实验记录数、PowerShell 日志编码、阶段续跑与报表配对统计。当前只是运行中的队列，不代表正式实验已经完成。
