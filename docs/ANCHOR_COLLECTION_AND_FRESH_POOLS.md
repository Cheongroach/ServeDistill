# 从完整 DEV 产物走到新的独立确认

本页延续分步阅读方式，集中说明输入、检查、输出与验收边界。

## 第 1 步：先固定本轮比较范围

原轮四方法见 `configs/anchor-development-v1.json`；新权重扩展见 `configs/anchor-weight-development-v2.json`。原轮的 0.5 监督锚定、pure AOPD、offline SFT、oracle SFT 共四个种子；扩展增加 0.1、1.0 两组各四个种子。训练预算、初始 adapter、Teacher 和 prompt 日程保持配对。

用户在 2026-10-28 取消全局 8 卡上限。当前扩展采用 12 个单卡训练任务，另有一个单卡专业工具诊断任务。每项任务仍有明确的运行时限；排队、准备、失败和有效计算分别保留。

## 第 2 步：原轮结束后收集全部产物

```bash
python -m scripts.collect_anchor_development \
  --protocol configs/anchor-development-v1.json \
  --submissions "$ANCHOR_SUBMISSIONS" \
  --training "$ANCHOR_DATA/train.jsonl" \
  --hdfs-root "$SERVEDISTILL_HDFS_ADVANTAGE" \
  --output "$RAW_ANCHOR_EVIDENCE"
```

`ANCHOR_SUBMISSIONS` 是按协议种子顺序排列的 JSON 数组，每项包含 `seed` 和平台 `sid`。HDFS root 是包含 `anchor-development-v1/` 的实验根目录；不要把已脱敏的公开路径当作真实挂载路径。

收集器要求全部任务成功终结。它核查实际 schedule、配置、完整 320 次更新、最终 BF16 adapter 指纹、worker completion，并下载每个方法的完整预测和日志。`collection-audit.json` 将文件大小与 SHA256 绑定到这次收集。

`cost.json` 使用每个 Pod 的 `schedule_at → stop_at`，包含准备阶段和所有 trial；未闭合区间拒绝出最终成本。这个数不是账单，也不是有效计算时间。

## 第 3 步：独立重评完整输出

```bash
python -m scripts.analyze_anchor_development \
  --root "$RAW_ANCHOR_EVIDENCE" \
  --protocol configs/anchor-development-v1.json \
  --dev "$ANCHOR_DATA/dev.jsonl" \
  --output results/anchor-development-analysis.json
```

该入口重评 4 seeds × 5 outputs（initial 加四方法）× 1,100 rows，共 22,000 条预测；重新计算 schema、逐类 recall、macro-F1 和 nDCG。已保存的分数、摘要或输入 ID 不一致会失败。

`advance_to_fresh_confirmation` 只代表通过原 0.5 候选的 DEV 门。新增权重存在后，最终选择仍须等待三组权重的全部结果，按扩展协议统一排序。不能把这个入口提前产生的 true 当成整个扩展实验已经完成。

## 第 4 步：收齐两个新增权重，执行一次统一选择

```bash
python -m scripts.collect_anchor_weights \
  --protocol configs/anchor-development-v1.json \
  --extension configs/anchor-weight-development-v2.json \
  --submissions "$WEIGHT_SUBMISSIONS" \
  --training "$ANCHOR_DATA/train.jsonl" \
  --hdfs-root "$SERVEDISTILL_HDFS_ADVANTAGE" \
  --output "$RAW_WEIGHT_EVIDENCE"

python -m scripts.analyze_anchor_weights \
  --base-root "$RAW_ANCHOR_EVIDENCE" --weight-root "$RAW_WEIGHT_EVIDENCE" \
  --protocol configs/anchor-development-v1.json \
  --extension configs/anchor-weight-development-v2.json \
  --training "$ANCHOR_DATA/train.jsonl" --dev "$ANCHOR_DATA/dev.jsonl" \
  --output results/anchor-weight-development-analysis.json
```

`WEIGHT_SUBMISSIONS` 的每项包含 `variant`、`seed`、平台 `sid`；两个 variant 为 `anchor01`、`anchor10`，各四种子。八个任务缺一不可，未结束的任务不能产生最终收集审计。

原轮和扩展共重评 30,800 条预测。收集器还逐步检查 320 条 update 与 rollout 记录：实际 sample ID、版本、学习率、有限数值、非零参数更新，以及 checkpoint 中的累计 response token 数。它不能只凭一个 `completed=true` 接受训练完成。

统一分析先对每个权重应用相同门槛：四种子全部通过绝对质量门，相对两个 SFT 对照均有正的平均业务差值、至少三对种子为正，各业务域和数学回归满足原非劣效门。只在合格者中按平均 DEV business macro 排序，精确平分时选较小权重；没有合格者则返回 `selected_dev_candidate=null`。pure AOPD 保留为消融，不参加选择。

`dev_allocated_gpu_hours` 合并这十二个任务的完整 Pod 分配区间；历史 Teacher 准备、标签、初始化和 profiler 成本仍单列。原始平台响应可能包含私有字段，`RAW_*_EVIDENCE` 必须放在 Git 忽略目录中，公开结果只导出审核过的统计和文件指纹。

## 第 5 步：数清真正未使用的输入

旧确认失败后，旧 test 和 replication 已经被观察。本项目历史输入审计结果：

| 数据池 | 排除历史输入后剩余 |
|---|---:|
| ESCI small-version 官方 test | 1,273 个查询组 |
| ESCI 完整版官方 test | 17,626 个查询组 |
| GSM8K 官方 test | 119 道原题 |
| GSM8K 官方 train | 4,573 道原题 |
| SVAMP | 0 道原题 |

对应证据为 `reports/advantage/FRESH_INPUT_POOL_AUDIT.json` 和 `FRESH_EXPANDED_INPUT_POOL_AUDIT.json`。查询同时排除跨官方 train/test 的同名组；候选产品至少两个且去重后计数。数学题同时按输入哈希、source ID 与规范化原题排除，不能通过换提示词获得“新样本”。

“本项目未使用”不等于基础模型预训练未接触。ESCI 完整版与此前 small-version 的抽样总体不同；GSM8K train 剩余题也必须明确标成从官方 train 划出的项目 holdout，不能称为官方 test。

## 第 6 步：复现输入池审计

固定源版本和完整 SHA256 在 `configs/fresh-pool-expanded-sources.json`。`ESCI_EXAMPLES` 指固定版本的 examples parquet；`GSM8K_TRAIN` 指固定版本的原始 train.jsonl。

```bash
python -m scripts.audit_fresh_pools \
  --examples "$ESCI_EXAMPLES" --gsm8k "$GSM8K_TRAIN" \
  --source configs/fresh-pool-expanded-sources.json --esci-version full \
  --previous "$OLD_DATA" "$OLD_SEARCH" "$OLD_EXPANDED" \
    "$OLD_SCREEN" "$OLD_HOLDOUT" \
  --svamp "$SVAMP_SOURCE" --output results/fresh-input-pool-audit.json
```

所有历史根目录都要提供，包含各轮 train/dev/test/replication。输出记录每个原文件的内容 SHA256、行数和去重统计。源数据字节与锁文件不符、历史 prompt 与保存哈希不符，审计立即失败。

## 第 7 步：确认协议必须在新预测前冻结

数据池足够大只解决可用样本问题。下一轮还需按 DEV 方差确定样本量，固定一个候选、两个 SFT 对照、完整种子集合、主指标、各域非劣效门、两个不重叠确认批次，以及后续重复确认的显著性预算。

可以继续做 DEV 迭代；不能反复测相同 holdout、看见有利结果便停止或更换比较指标。每次新的确认必须使用未观测输入，并把其消耗登记到后续排除列表。
