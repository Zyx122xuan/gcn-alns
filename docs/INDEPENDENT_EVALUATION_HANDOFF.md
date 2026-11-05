# 独立评测与实验溯源改造（2026-10-27）

本轮完成第二步代码改造：固定训练/验证/最终测试边界，记录模型来源与实验版本，拒绝不兼容续跑。第一步的恢复验证器修复继续保留。历史主实验与旧默认模型没有被本轮替换。

## 已准备的数据协议

正式清单位于 [`data/protocols/independent_v1/protocol.json`](../data/protocols/independent_v1/protocol.json)。

- 训练候选 **60** 个、验证候选 **15** 个：从已有 small+medium 的 75 个实例按固定种子 2026 分组划分，验证比例 20%。分配发生在 teacher 求解前。
- 最终测试 **90** 个：新种子生成 small=45、medium=30、large=15；五类工作流齐全；tight/medium/loose 各 30 个。各规模起始种子为 12026 / 112026 / 212026。
- 最终测试实例存于 `data/protocols/independent_v1/test/{small,medium,large}`，ID 使用 `heldout_` 前缀。它们尚未用于 teacher、训练、调参或正式求解评测。
- 协议 SHA-256：`0e20eb8f74bce033de5b4e881e319d667d1bf3c87bafb87b8e312f89b33a98cb`。

60/15 是协议中的输入数量，**不是已生成的可行示范数量**。完整 teacher 数据集和正式新模型尚未构建，不能据此报告新的模型质量或算法排名。

协议记录实例文件哈希、工作流内容哈希、资源快照哈希及不含 ID/元数据/SLA 的问题内容哈希。文件改名或只改实例 ID/SLA 不能把相同内容伪装成独立输入；复用工作流或资源快照的实例不能跨角色。划分学习数据时，相关变体一起分组。加载协议时重新核验所有输入文件；协议生成器拒绝覆盖已有清单或测试实例。

## 训练与评测边界

`gcn.dataset --protocol` 仅让 train/val 进入 teacher。某实例没有可行示范时，仅在它原本所属角色中缺席；不会重新洗牌，也不会从 test 补齐。训练产物中的 `split.test=[]`。最终求解测试则完整保留全部 test 实例，包括不可行实例。

模型 checkpoint 和 `<checkpoint>.provenance.json` 保存协议标识、所有 teacher 输入（含未成功生成示范的输入）、实际样本角色、配置快照、代码指纹；CLI 训练还记录 dataset.pkl 的文件哈希。新样本类使用稳定模块路径，支持跨 CLI 和多进程 pickle 读写。

`--protocol ... --split test` 下，GCN 模型必须有与 checkpoint 文件匹配的来源记录，属于同一协议，且 teacher 输入与 test 无交叉。旧默认 `results/gcn_model.pt` 缺少这些记录，会被最终测试入口拒绝。旧模型仍可用于明确标为 development 的探索运行。

所有实验入口支持 `--protocol/--split`；`--instances` 与 `--protocol` 互斥。敏感性扫描在协议模式下只允许 `--split val`。最终评测不自动跳过缺失的 GCN 模型。

## 实验版本与续跑

新结果目录的布局：

```text
evaluation_scope.json         固定该输出目录的角色和协议
manifests/<tag>.json           完整配置、代码/模型/实例指纹与实验参数
solutions/<tag>/<hash>.json    每次求解的 placement / mode / order / meta
raw_json/<tag>.jsonl           增量指标及版本、解快照引用
raw_json/<tag>.json            完成后的指标集合
csv/                          原有聚合输出
```

续跑前核对配置、源代码内容与 Git 提交、模型及来源文件、实例、算法列表、重复数、消融和标签后缀。任何不匹配都报错，要求换 tag 或输出目录。已经落盘的解缺失或被修改时也不静默跳过。并行 worker 数不参与语义版本指纹，可以改变并行度继续相同实验。

**旧 `results/` 中的 raw_json 没有新版本记录，不能直接追加新实验。** 新旧记录、train/val/test 或不同协议不能共用同一个聚合目录；需使用下面的新输出路径。模型 A/B 现在分别使用 `<tag>_a2_off`、`<tag>_a2_on`、`<tag>_alns`，避免两个模型共享续跑版本。旧报告分析脚本中的硬编码 tag 仍描述历史数据，新 A/B 表格应按这些新 tag 选择记录。

## 后续正式运行命令

以下为后续重建与评测命令，本轮没有执行正式 teacher/训练/测试全量计算。调参先看 val，配置和模型确定后再运行 test。

```powershell
$env:PYTHONPATH="src"

# 已有协议无需重新生成；新实验设计应另建协议目录。
python -m gcnalns.gcn.dataset --protocol data/protocols/independent_v1/protocol.json --out data/gcn_independent_v1 --workers 8
python -m gcnalns.gcn.train --data data/gcn_independent_v1 --out results/independent_v1/gcn_model.pt --device cuda:0

# 开发/调参只用 val，输出与正式 test 分开。
python -m gcnalns.experiments.run_main --protocol data/protocols/independent_v1/protocol.json --split val --out results/independent_v1/validation --model-path results/independent_v1/gcn_model.pt --algorithms ALNS GCN-ALNS --repeats 10 --workers 8 --tag main

# 冻结模型与配置后再运行正式测试；包含所有 test 实例。
python -m gcnalns.experiments.run_main --protocol data/protocols/independent_v1/protocol.json --split test --out results/independent_v1/test --model-path results/independent_v1/gcn_model.pt --repeats 10 --workers 8 --tag main
```

准备其他协议时可使用：

```powershell
python -m gcnalns.data.protocol --learning-instances data/instances/small data/instances/medium --out data/protocols/another_protocol --test-seed 32026
```

## 已完成的验证

- **38 个测试通过**，包含上一轮 21 个恢复验证器测试，以及内容泄漏、冻结文件变更、teacher 不接触 test、失败样本不换角色、checkpoint 来源核验、完整测试集保留不可行实例、版本变化拒绝续跑、解快照变更拒绝续跑和参数扫描保护。
- 10 个相关 CLI 的 `--help` 可正常加载；实验 CLI 均暴露 `--protocol/--split`。
- 实例生成器提取为可复用函数后，用同一配置与种子对照旧实现生成 15 个微型实例，文件名和 JSON 内容逐字节一致。
- 另用一套独立微型协议完成真实 CLI 链路：2 worker 建集 → 2 epoch CPU 训练 → 2 worker 运行 HEFT/ALNS/GCN-ALNS，共 6 条测试记录；原参数续跑不重复求解，修改迭代预算被拒绝，不可行测试实例保留。该微型协议与正式 90 个测试实例无关，其训练指标不代表模型研究效果。
- 冒烟摘要：[`results/diagnostics/protocol_smoke_20261027/summary.json`](../results/diagnostics/protocol_smoke_20261027/summary.json)，最终运行记录位于其 `results_test_v2/` 子目录。

本轮没有重新认证或替换旧论文数字。下一项重计算工作是使用修复后的验证器，在上述固定 train/val 上重新生成 teacher 并从头训练模型；正式 test 保留到模型与参数确定之后。
