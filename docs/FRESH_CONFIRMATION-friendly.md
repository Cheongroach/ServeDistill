# 一步步从 DEV 候选走到新的独立确认

本页对应 `freeze_fresh_confirmation.py → evaluate_fresh_confirmation.py → analyze_fresh_confirmation.py`。这些入口已通过 CPU 检查；当前十二项训练尚未全部终结，没有新确认预测，也没有已确认的质量优势。

## 第 1 步：确定本轮究竟要回答什么

问题是：固定训练后的候选，在新的查询上能否同时优于 offline SFT 和 oracle SFT，并保持各业务域、数学和输出格式的质量。

候选必须先通过 `analyze_anchor_weights` 对全部三组权重、四种子和两个 SFT 对照的统一 DEV 门。选择固定在第 320 次更新的 BF16 adapter；pure AOPD 仍作为消融保留。新确认不能帮助选择权重、检查点或种子。

自检：如果只有 0.5 权重跑完，另外两组还未结束，能否进入确认？不能。

## 第 2 步：给持续尝试一个明确的统计预算

前几轮已观察的确认失败仍保留。新的前瞻确认使用 `configs/fresh-confirmation-family-v1.json`，第 $r$ 次尝试的单侧预算为：

$$
\alpha_r = \frac{0.05}{r(r+1)}, \qquad
\sum_{r=1}^{R}\alpha_r = 0.05\frac{R}{R+1}.
$$

前四次分别是 0.025、约 0.00833、约 0.00417 和 0.0025。准备失败、GPU 失败、输出缺失或质量门失败都消耗该轮预算，后面不能重新从第 1 次开始。ledger 用连续编号和前一条 receipt 的 SHA256 绑定顺序。

每轮必须同时通过两个 SFT 对照、两个不重叠批次及所有质量门。这是“所有条件成立”才接受的交并检验规则；不能在多个结果中挑一个胜出者。若以后要接受“某个对照或某个批次通过即可”，必须另行预设多重比较校正。

统计边界也要明确：本实现使用 query percentile bootstrap，覆盖率是渐近近似。递减预算解决前瞻的名义重复检验记账，不能把 bootstrap 变成有限样本严格保证，也不能修复此前探索性分析的多重比较。

自检：这轮结果不理想，可以换一组查询却继续沿用同一个 $\alpha_r$ 吗？不能。

## 第 3 步：固定未使用的输入和样本量

新总体为 ESCI 完整版官方 test 查询，以及 GSM8K 官方 train 中本项目未使用的原题。源版本、examples、products 和 GSM8K 文件的 SHA256 见 `configs/fresh-confirmation-sources-v1.json`。它扩展了产品原表的校验；以前的输入池审计锁没有改写。

抽样先读取 query、官方 split 和 product ID。排除五个历史数据根目录中的全部查询、规范化数学原题和 input ID；排除跨官方 train/test 的同名查询；查询至少含两个去重产品。随后固定 domain 分配、产品抽样种子及两个批次的输入，最后才加载 relevance 标签和产品事实。

样本量根据已观察 DEV 的不确定性和剩余数据池确定，并在 `--sample-size-rationale` 留下依据。确认标签和预测不能用于改样本量、补少数类别或重新抽样。缺少规定类别支持时，该轮不能通过。

两个批次都包含 search、relevance、substitute、math。每个查询只进入一个 domain 和一个批次；不能把同一查询的多个产品当成独立样本。本项目未使用也不等于模型预训练未见过，GSM8K train holdout 更不能改称官方 test。

自检：修改旧数学题的空格或提示词，可以把它当作新题吗？不能。

## 第 4 步：生成一个封存包

在全部 DEV 产物收齐并独立重评之后执行：

```bash
python -m scripts.freeze_fresh_confirmation \
  --family configs/fresh-confirmation-family-v1.json \
  --source configs/fresh-confirmation-sources-v1.json \
  --base-root "$RAW_ANCHOR_EVIDENCE" --weight-root "$RAW_WEIGHT_EVIDENCE" \
  --protocol configs/anchor-development-v1.json \
  --extension configs/anchor-weight-development-v2.json \
  --training "$ANCHOR_DATA/train.jsonl" --dev "$ANCHOR_DATA/dev.jsonl" \
  --examples "$ESCI_EXAMPLES" --products "$ESCI_PRODUCTS" --gsm8k "$GSM8K_TRAIN" \
  --previous "$OLD_DATA" "$OLD_SEARCH" "$OLD_EXPANDED" "$OLD_SCREEN" "$OLD_HOLDOUT" \
  --ledger "$FRESH_LEDGER" \
  --business-count "$N_BUSINESS_FROM_DEV" --math-count "$N_MATH_FROM_DEV" \
  --sample-size-rationale "$DEV_SAMPLE_SIZE_RATIONALE"
```

`business-count` 是每个业务域、每个批次的查询数，`math-count` 是每个批次的原题数。这里的两个值必须在新预测前依据 DEV 确定，文档没有预填一个看起来有利的数。

封存包包含完整 DEV 重评分、reservation、输入 selection、两个 JSONL、manifest 和 protocol。protocol 绑定四种子的全部十二份 adapter、公共 Student manifest、抽样规则、代码 SHA256 和预算。bootstrap 至少 20,000 次，并保证名义尾部至少有 200 次抽样；实现分块计算以控制内存。

没有合格 DEV 候选时，入口在选择新输入之前失败。已登记后遇到准备失败，reservation 留在 ledger 中。原始封存包放在 `.ops/` 等 Git 忽略目录，公开资料使用统计和指纹。

自检：传入一个旧报告里的 `advance=true`，能否绕过完整 DEV 重评？不能，入口重新运行联合分析。

## 第 5 步：让每份预测对应唯一的模型和输入

每个种子分别评估 candidate、offline_sft、oracle_sft：

```bash
python -m scripts.evaluate_fresh_confirmation \
  --packet "$FROZEN_PACKET" --family configs/fresh-confirmation-family-v1.json \
  --student "$COMMON_STUDENT" --student-manifest "$COMMON_STUDENT_MANIFEST" \
  --adapter "$FROZEN_ADAPTER" --arm "$ARM" --seed "$PAIRED_SEED" \
  --output "$ARM_OUTPUT"
```

入口先验证封存包、执行代码、Student 和 adapter，再按固定 batch size 8、greedy decode、最多 128 个新 token 生成。它保存每个 input ID、query group、输出文本、评分与摘要，同时记录 protocol 和 adapter 指纹。两个批次都完成才写 arm completion。

Worker 的打包必须自包含：模型、封存包、冻结代码及环境依赖不能依赖开发机 `/mlx_devbox/` 路径。平台的启动、失败和分配时段继续进入独立成本账本。这个入口完成推理并不等于成本已闭合。

自检：GPU 重启后少一份 seed 输出，能否把其他三份平均一下继续？不能。

## 第 6 步：独立重评，再应用冻结的门槛

```bash
python -m scripts.analyze_fresh_confirmation \
  --packet "$FROZEN_PACKET" --family configs/fresh-confirmation-family-v1.json \
  --root "$ALL_FRESH_OUTPUTS" --output "$NEW_CONFIRMATION_ANALYSIS"
```

输入目录为 `seed101/candidate/`、`seed101/offline_sft/` 等十二个 arm，每个包含两个批次的预测和摘要。分析器要求每个 arm 的输出完整、一一对应，并用冻结题目重新评分；保存的 success、class、nDCG 或摘要不同都会失败。

bootstrap 以独立查询为单位，四种子保持配对；macro-F1 每次从混淆计数重算。主指标为三个业务指标的等权平均差值。只有它相对两个 SFT 的单侧下界都大于 0，并且两个批次各自通过，才支持本轮优势。各业务域下界须大于 −0.03，数学下界须大于 −0.05；四种子都须通过绝对质量门，每个对照至少三对种子的业务差值为正。

结论限于所声明的新输入总体和四个已观察种子。四种子不证明任意训练种子稳定，也没有证明吞吐或成本收益。

自检：confirmation 通过、replication 失败，能否合并两批数据救回结论？不能。

## 第 7 步：失败后怎样继续

保留失败预测、日志、adapter、平台成本和该轮 budget。新的 DEV 改进可以使用已观察结果，但下一次确认必须重新冻结一个候选、采用新的未观察输入并使用下一轮预算。前面各轮已经物化的 JSONL 自动加入后续排除列表。

若未使用数据池不足，或类别支持与样本量不能满足设计，当前来源不能继续提供新的独立确认。届时需要新的真实输入来源；不能把重复采样包装成独立复制。
