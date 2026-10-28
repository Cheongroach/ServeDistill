# 四种子 CE 锚点 DEV 终态复核

三种固定权重均未通过原有 DEV 晋级门；没有封存新的确认输入，也没有新的确认预测。以下为固定 update320 的 BF16 评估，不能据此宣称稳定质量优势。

## 结果与失败条件

所有 12 个单卡任务已经成功终结。收集器校验 428 个文件，绑定 24 份训练后的 adapter、完整 320 次更新和配对 prompt 日程；分析器独立重评 30,800 条预测。四种子为 101、211、307、401。每份 DEV 输出含 900 个业务查询、100 道数学题及 100 条 instruction 样本。

| CE 权重 | 业务 macro 均值 | 相对 offline SFT / pp | 相对 oracle SFT / pp | 数学相对 oracle / pp，查询 bootstrap 95% CI | 关键失败 |
|---|---:|---:|---:|---|---|
| 0.1 | 0.688771 | +0.3981，3/4 种子为正 | +1.2016，4/4 为正 | −4.50 [−8.00, −1.50] | 数学下界未超过 −5 pp |
| 0.5 | 0.687398 | +0.2608，3/4 为正 | +1.0642，4/4 为正 | −2.50 [−5.25, 0.00] | 数学下界未超过 −5 pp |
| 1.0 | 0.682245 | −0.2545，2/4 为正 | +0.5489，2/4 为正 | −0.25 [−2.00, +1.50] | 业务均值、种子胜率及 substitute 非劣门失败 |

三种权重的四个种子均通过绝对业务质量、类别 recall 和 JSON 门。绝对合格不代表优于对照。0.1/0.5 相对 offline 的业务 CI 仍跨零；原 DEV 门只要求正均值与至少三对种子为正，最终优势仍必须由新的独立确认检验。

数学来自已观察的 DEV 混合题集，不能称为新的 SVAMP 确认。CI 按查询重采样并保持四种子配对；四种子不是四倍独立查询，也不证明任意训练种子稳定。业务域下界必须超过 −3 pp，数学下界必须超过 −5 pp，且须同时相对 offline、oracle 与 initial 满足非劣条件。完整数值和所有失败见 [联合分析](ANCHOR_JOINT_DEV.json)。

## 成本与复算

原四任务含四种方法及 initial 评估，分配 28.475 H20·h；八个权重扩展任务分配 28.193611 H20·h，合计 **56.668611 H20·h**。两组 Pod 集合互不重叠，全部准备、训练、评估及导出均进入 schedule-to-stop 区间。共同教师修复、标注、初始化和历史监控成本单独保留；此数不是全项目成本或有效计算时间。

公开保留联合分析、两份成本账本、收集审计和逐查询评分。`ANCHOR_JOINT_PUBLIC_MANIFEST.json` 区分原始与公开 SHA256；模型权重及完整文本留在原始证据目录。使用原始证据复算：

```bash
python -m scripts.analyze_anchor_weights \
  --base-root "$RAW_ANCHOR" --weight-root "$RAW_WEIGHTS" \
  --protocol configs/anchor-development-v1.json \
  --extension configs/anchor-weight-development-v2.json \
  --training "$ORIGINAL_DATA/train.jsonl" --dev "$ORIGINAL_DATA/dev.jsonl" \
  --output "$RECOMPUTED_JOINT_ANALYSIS"
```

## 下一轮假设

弱锚点在本轮业务均值较高，强锚点数学保留较好。这支持在 DEV 上检验分领域 CE 权重：业务分别使用 0.1/0.5，数学和 instruction 固定 1.0。它尚不证明分领域权重有效。新协议须绑定更新后的源码，保留相同初始化、fresh Adam、日程、320 次更新、Teacher320 和原门槛；旧协议及失败结果保持原样。
