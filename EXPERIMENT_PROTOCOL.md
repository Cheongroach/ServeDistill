# ServeDistill：业务 OPD 实验协议

冻结日期：2026-10-24。状态：28训练阶段、最终评测与原生HTTP/回退测试已回收；冻结候选未通过最终替代品保留门。详见 reports/FINAL_REPORT.md。项目目标是判断方法是否值得进入业务试点；模拟任务成绩不能表述为真实业务收益。

数据方向更新：根据用户对团购、搜索推荐广告数据的要求，主实验改用公开搜索相关性数据，优先 Amazon ESCI；原模拟业务约束保留为补充评测。任务定义、数据隔离及扩展优先级以 [DATASET_SELECTION.md](DATASET_SELECTION.md) 为准。以下损失、teacher 资格、训练监控与成本原则继续适用。固定更新数、有效 tokens、GPUh 分别报告，生成长度不同的方法不宣称同时严格匹配更新数与 tokens。

## 业务与上线目标

主任务是商品搜索的结构化决策助手。三个业务域为搜索重排、E/S/C/I 相关性判断和替代品识别，输入包含真实公开 query 与商品文本，输出固定 JSON。ESCI 人工标签用于独立评分；排序主指标为采用官方线性 gain 的 NDCG，分类主指标为 macro-F1，同时报告 MRR、准确率及协议错误。

餐饮、酒店和到店服务的人工模拟任务保留为补充约束检查，覆盖预算、距离、可退款、人数和信息不足。它们不包含真实交易数据，不用于宣称团购成交收益。

模型负责需求理解、约束筛选和候选选择。检索提供事实，确定性程序检查动作；不把缺失的商家知识交给蒸馏补齐。业务上线要求单个小模型处理常规决策，低置信度/协议错误按可观测信号转人工或强模型。回退系统质量、单模型质量分别报告。

使用公开 post-trained checkpoint 作为起点，主 teacher 为 Qwen3.5-9B，student 为 Qwen3.5-2B，4B 用于大小模型质量/成本参考。使用原生 text-only 语言组件，明确关闭 teacher/student 的 thinking；LoRA 同时覆盖 full attention 和 Gated DeltaNet 投影。它们不能称作经过本项目独立 RL 训练的领域专家。领域 teacher 通过固定、版本化的提示路由；提示专家融合与独立权重专家融合明确区分。

通用回归包含公开 GSM8K 子集和确定性指令任务。它只能支持对应能力的保持结论，不能代表完整“通用能力”。公开测试集可能存在预训练污染。主数据目录的 `ood` 为额外独立官方 test query 复验，不能宣称真实分布漂移；人工模拟任务的规则变化测试另行标注。

## 数据、监督与划分

ESCI 沿用官方 train/test，dev 仅从官方 train 划分。归一化 query 在各 split 间隔离；跨官方 split 的重复 query 排除。候选随机选择与顺序不依赖标签，最多 8 个，保留类别不平衡与多语言分布，并报告商品 ID 重叠。每例保存 `sample_id`、`domain`、`split`、输入哈希和人工标签目标；prompt 不含标签。训练 reward 使用独立标签和确定性字段检查，不使用 teacher 作为唯一评分器。

训练分布默认业务 80%、通用 20%，三个业务域等份。记录 prompt 占比、有效 response token 占比和域内梯度/损失贡献。长文本不能凭样本数量独占更新。混合 batch 与分桶 batch 使用同一任务序列及相同总域预算，分桶只改变相邻样本组织。

test/OOD 标签不参与 reward、超参选择、动态系数调整或 early stopping。dev 用于选 checkpoint；最终配置冻结后才运行最终测试。盲测指实验选择层面的隔离，公开标签不会被宣称为保密资产。

## Teacher 资格门

在相同输入、检索事实、输出预算和确定性解码下比较 teacher/student：

1. 三个业务域分别报告准确率、约束违反率、JSON 有效率，以及配对差值置信区间。
2. teacher 对 student 访问的错误前缀进行条件评分，记录 logratio、entropy 和数值异常。
3. 核对完整 token ID 映射、added/special tokens、EOS、chat template 和 response 边界。不同 token 事件空间不得逐 token 直接相减。
4. 短 pilot 必须证明非零梯度、参数变化、teacher 无梯度、行为 logprob 与 learner 重算一致。

teacher 未优于 student 的域不给出“有可迁移能力”的结论。可继续作为 teacher 失败对照，或只通过独立 dev 改提示并重新冻结版本。

## 损失与更新粒度

令 \(s_t=(x,y_{<t})\)，student 为 \(p_\theta\)，teacher 为 \(q_d\)：

\[
D_{\mathrm{KL}}(p_\theta\Vert q_d\mid s_t)
=\sum_{v\in V}p_\theta(v\mid s_t)
\log\frac{p_\theta(v\mid s_t)}{q_d(v\mid s_t)}.
\]

传统序列蒸馏在 teacher 轨迹上用 CE 训练；OPD 先由 student 生成，再在 student 实际前缀上取得 teacher 分布监督。[cite:1](#source-1)

sampled-token PG OPD 使用
\[
A^{\mathrm{OPD}}_t=\operatorname{stopgrad}
[\log q_d(y_t\mid s_t)-\log p_{\mathrm{old}}(y_t\mid s_t)].
\]

在固定状态上，它给出 reverse-KL 的动作采样梯度估计；训练时固定采样前缀，不能称为完整轨迹期望的无偏梯度。行为采样使用 temperature=1、top-p=1 和完整支持；停止 token 计入动作，padding/prompt token 不计入。teacher 不反传，advantage detach。负的单 token logratio 保留。

本轮 BF16 缓存解码与 teacher forcing 重算的最大误差触发 0.15 门限；FP32 student + PyTorch DeltaNet 的诊断误差降至 4.6e-5，四次更新验收为 4.55e-5。主训练采用这一路径，teacher 仍为 BF16；部署单独以 BF16 合并 adapter 评测。FP32 的额外耗时显式入账。[cite:2](#source-2)[cite:3](#source-3)

GRPO 的可验证轨迹 reward 按同 prompt 的 group 标准化，然后广播到 response 动作；组内 std=0 时记零有效组，不伪造 advantage。联合方案分别计算两项：

\[
L=L_{\mathrm{GRPO}}+\lambda L_{\mathrm{OPD}}.
\]

主实现每完成一批 rollout/评分后进行一次 optimizer update；OPD 信号是 token 级，GRPO reward 是轨迹级。这不表示“每 token 更新参数”。在一条正在生成的轨迹中不切换权重；逐次更新保留 student 版本和行为 logprob。

OPD 和 GRPO 不共用 advantage 标准化。交替方案每次 optimizer update 二选一，保证累计任务和 token 预算可比。动态方案只能使用训练信号与 dev，系数有上下界并记录每次调整理由；它是待验证工程策略，不预先称作优于静态消融。

另设全词表 reverse-KL 小规模数值/训练对照。sampled-token、直接分布 loss 和裁剪 surrogate 必须分别命名，不把 top-k softmax 称为全词表 KL。

## 实验矩阵与预算

| 实验 | 数据与目标 | 回答的问题 |
|---|---|---|
| 不更新 student、teacher | 全部资格集 | 初始能力差与部署成本 |
| 离线 teacher 序列蒸馏 | 固定 teacher 轨迹 CE | OPD 是否优于传统蒸馏 |
| oracle SFT | 确定性正确决策 | 可验证标签充足时是否仍需要 OPD |
| 纯 GRPO | 可验证轨迹 reward | outcome 优化基线 |
| 纯 OPD | student 轨迹、teacher token 信号 | 密集监督的收益和代价 |
| 混合 \(\lambda=0.25,1,4\) | RL + OPD 加权相加 | 权重与梯度竞争 |
| 交替 RL/OPD | 更新级二选一 | 是否需要同时优化 |
| 动态混合 | 有界系数、dev 控制 | 自动调权是否比固定权重稳定 |
| 分桶 OPD | 相同域预算 | batch 组织影响 |
| 业务单独 SFT → 恢复 OPD | 保留中间 checkpoint | 是否发生遗忘以及是否能恢复 |

先以 1 个短 pilot 验证运行环境，后扩展独立实验。主对照至少 2 个训练种子；未达到预算或验收的实验如实标为未完成。主比较固定 student 更新数，另外报告有效 response tokens 及固定总 GPU 小时下的曲线，不宣称生成长度不同的方法同时严格匹配更新数与 tokens；teacher 成本不会被遗漏。

用户授权窗口为 30 小时墙钟时间。项目额外设置 32 卡并发硬上限、240 GPU 小时保守内部预算、每任务独立 timeout 和全局截止时间。先检查资格再扩卡，失败尝试也入账；不通过反复提交重置预算。启动与加载先分批，复用带 SHA-256 的资产包。

## 训练监控与崩溃判定

每次更新记录：各域 reward/成功率、组内 std 和有效组比例、OPD logratio 与 entropy、KL 对固定初始模型、PPO clip fraction/importance ratio/ESS、梯度 norm、参数 delta、Adam 状态、输出长度/EOS/截断/JSON 错误、teacher/rollout/learner 时间、有效 tokens/GPUh 和显存峰值。

立即停止：loss/gradient/参数非有限值；token 或 sample ID 不匹配；teacher 被更新；训练版本与 rollout 契约不符。保存最后一个健康 checkpoint 和触发样本。

连续恶化的组合用于报警，并在 dev 验证后回退：reward 与独立 dev 成功率同时下降，entropy 快速坍塌，reference KL/clip fraction 或梯度 norm 同时上升；或截断/重复率上升伴随有效组比例与 JSON 有效率下降。单独 loss 升高或 sampled KL 为负不判崩溃。具体阈值基于 pilot 稳态冻结，不宣称存在跨模型通用阈值。

## 最终收益、统计与上线成本

每个域报告 student 初始/teacher/最终模型原始分数、提升百分点、teacher headroom 恢复比例及配对 bootstrap CI；teacher headroom 小或非正时不使用归一化比例。业务报告宏平均、预先规定的流量加权分数、最差域、约束违反率、OOD 和通用回退。

多种子报告均值与种子范围，CI 的抽样单位是独立任务；不能把同一题重复采样当成独立题扩大样本量。训练 dev 选配置，最终 test 一次验收，保留未达标与负结果。

部署方法在主矩阵 12 个方法完成后冻结：先对每方法的两种子 dev 分数取均值，使用 \(0.8\,business+0.2\,general\) 排序；五个域的平均分相对各自初始 dev 不得下降超过 5 个百分点。方法内选 dev 分数较高的种子，平分时按名称。各运行的 checkpoint 已由既定 dev 规则选择。若全部未过门，只选择研究候选并明确拒绝上线资格。另保留相同规则下最好的 OPD 方法，避免最优方法为 SFT 时遗漏 OPD 的部署代价。规则于 2026-10-24 10:34 UTC、主矩阵最终结果比较前记录；不使用最终 test 或 HTTP 成绩调整规则。

在同一 H20、相同输入/输出上限、相同精度下测 teacher、初始 student、候选 student 的 warm latency、P50/P95/P99、TTFT、输出 tokens/s、并发吞吐、显存和冷加载时间。框架参考实现与上线推理引擎分别标注；本地直调延迟不称作线上端到端 SLA。

报告 GPU 秒/请求、每百万有效请求的 GPU 小时、teacher 标注/评分与 student 训练完整成本。未提供 GPU 单价时用符号化单价 \(c\) 元/GPUh，货币成本为实测 GPUh × \(c\)；不编造价格。计算训练成本/单请求节省的摊销请求数，说明失败、回退和有效回答率的影响。

## 业务迁移决策

teacher 可以来自已上线强模型、业务 SFT/RL checkpoint、历史通用 checkpoint，或具有固定检索工具与规则提示的开源模型。必须取得可用输出或概率接口，并在真实业务资格集验证质量；仅有闭源文本 API 时先比较离线序列蒸馏，不能假定可获得 logits。

当已有可信 teacher、希望密集行为迁移或恢复历史能力时选 OPD；当 outcome 可可靠验证、teacher 在目标业务不够强或需要超过 teacher 时选 GRPO。稀疏 reward 与强 teacher 同时存在时比较混合；高质量标签足够时保留 SFT 的成本优势。新知识优先检索/数据建设。多专家合版只是 OPD 的一个用途。

## 来源

<a id="source-1"></a>
[cite:1] Agarwal et al., ICLR 2024. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*。
https://deepmind.google/research/publications/on-policy-distillation-of-language-models-learning-from-self-generated-mistakes/

<a id="source-2"></a>
[cite:2] Thinking Machines, Kevin Lu, 2025-10-27. *On-Policy Distillation*。
https://thinkingmachines.ai/blog/on-policy-distillation/

<a id="source-3"></a>
[cite:3] verl 官方文档，页面更新时间 2026-10-03，检索日期 2026-10-24。*On-Policy Distillation (OPD)*。
https://verl.readthedocs.io/en/latest/algo/opd.html

资格集实测：替代品 teacher macro-F1 0.4012，低于 student 0.4409；保留为不合格 teacher 负对照，不用其成绩宣称能力迁移。所有任务配置、域预算与种子在最终模型测试前冻结。
