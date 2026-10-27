# 一步步认识 ServeDistill 项目

本文按“一个步骤增加一个概念”的顺序解释项目。默认你已了解模型、token、embedding、GPU 和训练，不再复述这些基础知识。

每一步结束先回答问题；回答不清楚就回到这一步。代码入口是查证线索，不要求立即读完整文件。各阶段最后有检查点。

文中的历史数字来自归档实验。2026-10-27 新一轮监控与 DEV 实验正在执行，未完成结果不会写成结论。最新状态见 [README](../README.md) 和 [性能监控文档](../PERFORMANCE_MONITORING.md)。

---

## 第一阶段：先确定项目的问题

### 第 1 步：项目需要输出业务判断

输入是搜索词和候选商品。输出可能是商品排序、相关性类别或是否可替代的 JSON。项目不以自由文本的“看起来合理”作为正确标准。

只记住：**每种业务判断都有可检验的输出规则。**

停下来回答：一段流畅的回答，是否必然是正确业务判断？

### 第 2 步：三个业务任务分别评分

`search` 用 NDCG 评价排序；`relevance` 区分 E/S/C/I；`substitute` 区分 yes/no。后两者用 macro-F1。三种分数各在自己的任务内计算，不能直接混合所有样本的 accuracy。

代码入口：`servedistill/statistics.py`、`runtime.summarize()`。

只记住：**先得到三个域的分数。**

停下来回答：搜索排序和二分类是否使用同一个评分函数？

### 第 3 步：业务 macro 是三个域的等权均值

项目的业务汇总量为

```text
business_macro = (search_NDCG + relevance_macro_F1 + substitute_macro_F1) / 3
```

它给三个域相同权重。某个域的样本更多，不会自动获得更大的域权重。

只记住：**汇总量表达了一个预先选择的业务权重。**

停下来回答：增加 relevance 样本数量，会直接提高它在这个均值中的权重吗？

### 第 4 步：通用回归是额外的约束

数学与指令任务检查业务训练有没有损害原有能力。这些有限检查不能代表完整通用智能，也不能被一个更高的业务均值抵消。

只记住：**项目同时要求业务收益与能力保持。**

停下来回答：业务分提高，但数学明显退步，是否一定能通过验收？

### 第一阶段检查

你应该能说明输入、三个业务分数和通用回归各自承担什么作用。此时先不要讨论蒸馏损失。

---

## 第二阶段：认识公平的对照

### 第 5 步：Student 是最终希望部署的模型

主模型路线使用 Qwen3.5-2B Student。它承担最终业务回答；训练过程中的 Teacher 不必部署到每个线上请求中。

只记住：**最终被评价和部署的是 Student。**

停下来回答：Teacher 在训练时参与，是否意味着每个线上请求都必须调用 Teacher？

### 第 6 步：Teacher 提供条件概率反馈

Teacher 对给定提示和已生成的前缀评分。项目的主要 Teacher 来自 Qwen3.5-9B；后续实验还使用经过训练集修复的版本。更大的参数量不自动保证业务标签可靠。

只记住：**Teacher 的实际任务质量需要先检查。**

停下来回答：模型更大，是否足以证明它是合格 Teacher？

### 第 7 步：oracle SFT 使用真实训练标签

它用训练集的独立标注答案做交叉熵训练。这里的 oracle 指可获得标注的训练对照，并不指模型知道测试集答案。

只记住：**有标注时，oracle SFT 是必须认真比较的强对照。**

停下来回答：为什么只胜过一个较弱方法还不够？

### 第 8 步：offline SFT 使用冻结的 Teacher 回答

Teacher 先对训练提示生成答案，Student 再学习这些固定答案。Teacher 的错误答案仍须保留，不能按正确率过滤后假装是原始数据分布。

代码入口：`scripts/run_advantage_labels.py`、`advantage_contracts.verify_labels()`。

只记住：**offline SFT 的监督是冻结的 Teacher 轨迹。**

停下来回答：离线标注生成结束后，Student 每次更新是否需要重新向 Teacher 生成答案？

### 第 9 步：共同起点避免比较起跑位置

比较各方法时使用相同 Student adapter，分别建立 fresh Adam，并绑定同一随机种子的提示顺序。继承不同训练阶段的 adapter 或 Adam，会引入额外变量。

代码入口：`scripts/run_anchor_development.py`。

只记住：**共同起点包含实际权重和优化器初始化规则。**

停下来回答：模型名字相同但 adapter 不同，是否算相同起点？

### 第二阶段检查

画出 Student、Teacher、oracle SFT 和 offline SFT 的关系，并指出每一种监督从哪里来。

---

## 第三阶段：理解 on-policy 的含义

### 第 10 步：轨迹是 Student 自己生成的回答

Student 根据当前权重逐 token 生成回答。这串被采样的 token 就是本次更新的轨迹。

代码入口：`Runtime.rollout()`、`train.collate_rollouts()`。

只记住：**轨迹来自正在训练的 Student。**

停下来回答：用 Teacher 预先生成的固定回答，是否就是 Student 的 on-policy 轨迹？

### 第 11 步：on-policy 关注 Student 实际访问的前缀

Teacher 对 Student 已经生成的每个前缀给出下一 token 的概率。Student 即使生成了不理想的开头，Teacher 也在这个实际前缀上评分。

```text
Student 生成 token → Teacher 在相同前缀评分 → Student 更新
```

只记住：**Student 轨迹和 Teacher 自己会生成的轨迹是不同分布。**

停下来回答：Teacher 评分时，是否可以偷偷把 Student 前缀换成正确答案前缀？

### 第 12 步：behavior logprob 记录采样时的概率

`behavior_logp` 来自实际生成时的分数。它记住更新前 Student 对被采样 token 给出的概率，后续不能用更新后的概率替代。

只记住：**behavior probability 属于生成这一版轨迹的权重版本。**

停下来回答：同一条轨迹的 behavior 概率能否随权重更新而改写？

### 第 13 步：重算检查防止训练不同的概率

更新前重新评分同一轨迹，检查生成概率与 learner 重算概率的最大绝对差。历史 BF16 路径没有通过既定概率门，训练因此采用 FP32/PyTorch DeltaNet 路径。

代码入口：`train.apply_rollout()`；默认误差阈值为 0.15。

只记住：**这道检查保护采样与训练的一致性。**

停下来回答：训练 loss 很低，是否能替代生成/重算概率检查？

### 第 14 步：response mask 确定哪些 token 参与训练

项目包含第一次 EOS，排除它之后的 padding 和后续 EOS。提示 token 不属于本次回答的训练动作。

代码入口：`runtime.action_mask()`、`supervision.gold_responses()`。

只记住：**分母和梯度都必须使用同一个回答 mask。**

停下来回答：为什么第一次 EOS 和 EOS 之后的填充不能同等处理？

### 第三阶段检查

你应能从一条回答定位：权重版本、采样 token、behavior logprob、Teacher 前缀和有效 mask。缺少其中一项时，先不要解释 loss 曲线。

---

## 第四阶段：一次只理解一个损失

### 第 15 步：sampled OPD 的即时信号是概率对数差

本项目的即时信号为

```text
A_t = stop_gradient(log p_teacher(y_t | prefix) - log p_behavior(y_t | prefix))
```

正值支持这个动作，负值抑制这个动作；负信号不会被删去。

代码入口：`losses.opd_advantages()`。

只记住：**信号是 Teacher 与采样 Student 对同一动作的比较。**

停下来回答：删除所有负信号，会不会改变方法？

### 第 16 步：policy loss 把信号转成 Student 梯度

`policy_loss()` 使用 current/behavior 概率比，并执行预设的 clipping。behavior 概率和优势信号被 detach，梯度只进入 current Student。

只记住：**Teacher 评分不是要一起更新 Teacher。**

停下来回答：这次反向传播应该更新哪一侧权重？

### 第 17 步：full reverse-KL 比较完整下一 token 分布

完整词表的 reverse-KL 在 Student 分布下计算 `log p_student - log p_teacher` 的期望。它需要完整分布信息，和只评价采样 token 的即时 policy loss 不相同。

代码入口：`losses.full_reverse_kl()`。

只记住：**full-KL 和 sampled OPD 是不同的损失实现。**

停下来回答：两个实现都出现 logprob 差，是否就证明它们完全等价？

### 第 18 步：AOPD 增加非正优势位置的指导

当前 AOPD 在非正优势位置使用截断 top-k forward 指导，在正优势位置保留采样 policy 更新。它是对纯 sampled OPD 的方法改动。

代码入口：`losses.asymmetric_opd_loss()`；默认 `top_k=32`、`tau=0`。具体归一化由实现和 CPU 数值测试共同定义，不能把截断分布写成完整词表 KL。

只记住：**AOPD 不应冒充原始纯 OPD。**

停下来回答：论文中 AOPD 的收益，是否已经证明本项目这份实现有收益？

### 第 19 步：gold CE anchor 是新增的标注监督

新 DEV 假设使用

```text
L = L_AOPD + 0.5 × L_gold_CE
```

gold 回答来自当前训练提示的真实 target，使用独立的有效回答分母。这一项可能改善标签漂移，也可能让方法更像 SFT；实验仍需比较两种 SFT。

代码入口：`Config.anchor_ce_weight`、`train.apply_rollout()`、`supervision.gold_ce()`。

只记住：**带 gold anchor 的 AOPD 是混合监督方法。**

停下来回答：这个候选胜出时，能否直接宣布“纯 OPD 胜出”？

### 第四阶段检查

把 sampled OPD、full-KL、AOPD、AOPD+anchor 分别写成一句话。能否说明它们哪里用到完整分布，哪里用到标注？

---

## 第五阶段：让更新预算可比较

### 第 20 步：一次 update 只有一次 optimizer step

一个更新可以包含多个 microbatch，但梯度累积结束后只更新一次参数。Student 在生成轨迹和整次 learner 更新之间保持同一 ownership。

只记住：**microbatch 数不等于 optimizer 更新次数。**

停下来回答：把一个 batch 切成四段，是否应该自动执行四次 optimizer step？

### 第 21 步：token 等权会放大长回答的贡献

`aggregation=token` 对所有有效 response token 求均值。长排序回答可以比短 yes/no 回答贡献更多训练权重。

代码入口：`losses.reduce_actions()`。

只记住：**归约分母会改变多任务训练的权重。**

停下来回答：相同提示数量，是否必然意味着各任务获得相同损失权重？

### 第 22 步：trajectory 等权先在回答内部平均

`aggregation=trajectory` 先计算每条回答的 token 均值，再平均回答。microbatch 累积必须按该 microbatch 的回答数占总回答数的比例缩放。

只记住：**拆分 microbatch 不应偷偷改变全 batch 的目标。**

停下来回答：最后一个 microbatch 只有两条回答时，能否和四条回答的 microbatch 使用同一累积权重？

### 第 23 步：同 prompt/update 不等于同计算预算

OPD 需要采样和 Teacher 评分，SFT 学习固定轨迹。即使各自都训练 320 次、每次 10 个提示，response token 和耗时仍可能不同。

只记住：**更新收益、时间收益和成本收益要分别测量。**

停下来回答：哪个额外证据能支持“同成本更好”？

### 第 24 步：学习率曲线也是实验配置

新 DEV 轮固定前 160 次使用基础学习率，之后线性衰减至第 320 次的 0.1 倍。不是看到 DEV 下降才临时调学习率。

代码入口：`scheduling.update_learning_rate()`。

只记住：**每一步实际使用的学习率应进入日志。**

停下来回答：只保存最初的学习率，能否复现一个衰减实验？

### 第五阶段检查

你应能检查更新数、提示数、序列数、有效 token 数、学习率和 microbatch 缩放。不要把其中任意一个量当成全部计算预算。

---

## 第六阶段：把状态恢复变成可以查证的事实

### 第 25 步：checkpoint 不只有 adapter

项目保存 adapter、Adam、Python/NumPy/Torch/CUDA RNG、controller、step、token 计数和 immutable config。只恢复 LoRA 权重属于重新开始一个实验。

代码入口：`train.save_checkpoint()`。

只记住：**无缝续跑需要恢复完整训练状态。**

停下来回答：为什么相同 adapter 加 fresh Adam 不能叫 seamless resume？

### 第 26 步：指纹检查验证实际加载的状态

冷进程恢复后重新计算 adapter、optimizer 和 RNG 的 SHA256，并与 checkpoint 的 `state.json` 对比。配置变化或状态不一致会终止执行。

只记住：**恢复成功需要实际内容一致。**

停下来回答：文件名是 checkpoint-0160，是否足以证明恢复了正确状态？

### 第 27 步：fork 和 resume 是不同操作

`--adapter` 建立 fresh Adam；`--resume` 恢复全状态；`--fork-parent` 在明确的新实验规则下携带旧 Adam 并重置指定状态。三条路径不能混写成“相同起点”。

只记住：**状态继承规则必须写进协议。**

停下来回答：四个对照中只有一个继承旧 Adam，会引入什么额外变量？

### 第 28 步：profiling 应使用诊断副本

采集任务从已验证检查点恢复到新的输出目录，保留一个完整 warmup update，再捕获预定 update。它会继续更新诊断副本，不会写回源 checkpoint。

代码入口：`scripts/profile_training.py`、`scripts/profile_run.py`。

只记住：**采集范围和源训练状态要隔离。**

停下来回答：为什么不把重新加载模型的时间当成稳定 learner kernel 时间？

### 第六阶段检查

找到 `state.json`、`restoration-*.json`、`manifest.json` 和新的输出 checkpoint。说明各自证明什么，以及无法证明什么。

---

## 第七阶段：先测量，再解释速度

### 第 29 步：真实耗时来自无 profiler 的重复测量

固定 workload bank，warmup 后多次同步测量，配对同一个 bank 的 microbatch 1/4。否则输入长度变化可能被误当成优化收益。

代码入口：`scripts/benchmark_microbatch.py`、`scripts/run_expanded_performance.py`。

只记住：**计时比较先固定工作量。**

停下来回答：两个方法碰巧生成不同长度，是否足以直接比较 learner tokens/s？

### 第 30 步：NVTX 只给执行范围命名

`rollout`、`learner_forward`、`teacher_forward`、`backward`、`optimizer` 等范围给 timeline 提供语义。CPU 范围结束不意味着异步 CUDA kernel 已经完成。

代码入口：`observation.TrainingObservation.span()`。

只记住：**一个 host span 不是天然的 GPU 完成时间。**

停下来回答：CPU range 只有 1 ms，是否证明其中的 CUDA 工作只用了 1 ms？

### 第 31 步：Nsight Systems 解释时间线上发生了什么

它关联 CUDA API、kernel、复制、NVTX 和可选 OS runtime。适合判断 CPU launch 间隙、阶段重叠与同步。CLI 名称是 `nsys`。

只记住：**先用 nsys 定位时间分布。**

停下来回答：timeline 的空白是否一定由 GPU 算力不足造成？

### 第 32 步：Nsight Compute 解释一个 kernel 的硬件行为

`ncu` 采集 kernel counter，分析 SM、DRAM、occupancy 等。多次 replay 可以串行化工作并改变缓存状态，因此报告时间不属于无 profiler 的端到端计时。

只记住：**用 NCU 解释已定位 kernel 的瓶颈。**

停下来回答：NCU replay 下的 kernel 时间可否直接当成应用实际延迟？

### 第 33 步：PyTorch profiler 把 kernel 关联回 operator

它记录 operator、input shape 和可选内存事件。适合解释 microbatch 改变了哪些 operator/shape，和 nsys 的系统时间线互补。

只记住：**框架归因和系统归因回答不同问题。**

停下来回答：operator 表的 inclusive 时间相加，是否必然等于端到端时间？

### 第 34 步：nvidia-smi 提供较低频的设备状态

遥测记录功率、显存、时钟、温度和设备利用率。500 ms 的采样无法定位一个持续数十微秒的 kernel。

只记住：**GPU utilization 不等于 Tensor Core 利用率。**

停下来回答：GPU utilization=100%，是否证明计算已经达到硬件峰值？

### 第 35 步：主机观测解释 GPU 为什么可能等待

`/proc` 记录进程树 CPU、RSS、page fault 和 IO；`perf stat` 提供 CPU counters。节点级数字可能包括其他租户，需结合进程范围判断。

只记住：**GPU 间隙的原因需要 CPU/IO 证据。**

停下来回答：共享主机 load average 很高，能否直接归因于本实验？

### 第七阶段检查

按“无 profiler 计时 → nsys 定位 → PyTorch 归因 → NCU 核查 → 主机/设备遥测解释”的顺序读 [工具文档](../PERFORMANCE_MONITORING.md)。

---

## 第八阶段：区分数值有效与业务有效

### 第 36 步：运行保护负责拒绝无效更新

训练检查 finite loss、finite gradients、finite parameters、Teacher 冻结、行为概率误差和真实非零更新。失败必须留下日志，不能删掉再统计幸存者。

只记住：**正常退出和数学有效还需要具体检查。**

停下来回答：退出码为零但没有目标 capture update，是否通过采集有效性？

### 第 37 步：JSON schema 有效不等于答案正确

`{"answer":"E"}` 可以是可解析 JSON，却不符合 yes/no 任务的取值规则；合法 yes/no 也可能预测错误。报告要分别保留解析、schema 和业务评分。

代码入口：`data.grade()`。

只记住：**语法、字段规则和语义是三个检查层次。**

停下来回答：为什么不能把历史六项取值错误写成“JSON 解析失败”？

### 第 38 步：每个类别的 recall 阻止常数预测掩盖失败

多数类 accuracy 很高时，少数类 recall 仍可能为零。历史 expanded 实验的 substitute 最终模型都预测 no，正类 recall=0，所以速度改善不代表可用质量。

只记住：**看 macro-F1 时也要核查类别 support 和 recall。**

停下来回答：为什么 yes 的 support 必须和 recall 一起保存？

### 第 39 步：绝对质量门先于方法比较

当前延续的业务门包括 search≥0.80、relevance≥0.45、substitute≥0.55，schema≥0.99；还要求 relevance 各类 recall≥0.20、substitute 两类≥0.30。

门槛来自冻结协议，不因结果接近门槛而降低。

只记住：**胜过一个对照，不代表通过业务最低要求。**

停下来回答：两种方法都很差，但其中一个较好，能否直接声称可部署？

### 第 40 步：非劣效门防止某个域被均值掩盖

延续协议要求业务每域差异不低于 −0.03、有限数学不低于 −0.05。具体确认使用配对统计区间；DEV 判断仅用于开发。

只记住：**总体进步不能代替每个受保护域的检查。**

停下来回答：search 大幅提高、substitute 明显退步，为什么仍可能被拒绝？

### 第八阶段检查

同时读成功率、schema、macro-F1、逐类 recall、非劣效和数值保护。任选一个维度，举出它通过而另一个维度失败的情况。

---

## 第九阶段：理解稳定优势需要的证据

### 第 41 步：DEV 允许迭代但不提供独立确认

开发集用于选学习率、adapter 规则和方法。看过 DEV 很多次之后再测同一批输入，增加的是预测记录，不是独立新 query。

只记住：**DEV 结果必须保留开发身份。**

停下来回答：四个 checkpoint 各评 1,100 条，是否得到 4,400 个独立输入？

### 第 42 步：多个种子检查训练波动

四个种子绑定相同任务集合，各自产生模型和配对差异。当前协议要求至少三个正向种子配对，同时保留全部种子，不挑选最好一个。

只记住：**多种子不是把同一个 query 的重复预测当成更多样本。**

停下来回答：只有两个种子提高、平均分略高，能否满足三个正向种子的门？

### 第 43 步：query 配对区间表达样本不确定性

每次重采样对候选和对照使用相同 query，并绑定种子。macro-F1 在每次重采样中重算，不能把它当作逐行可加的 accuracy。

代码入口：`advantage_statistics.py`。

只记住：**统计单位要与业务指标一致。**

停下来回答：为什么简单平均逐行“F1 差”不成立？

### 第 44 步：独立复验需要另一批未用于开发的输入

冻结候选后才生成 test 和不相交 replication 输出。新确认失败后，这两批输入已被观察，后续改方法需要新留出集。

只记住：**冻结后失败的测试不能重新命名为 unseen。**

停下来回答：调整 anchor 权重后重测旧 test，能否称为第一次独立确认？

### 第 45 步：多次确认需要预先限制选择性检验

若准备继续多轮正式确认，应先声明总体错误率预算和每轮独立数据，不能边看 p 值边决定停止。既有 SVAMP 全部原始题已被观察，不能再声称新一轮 SVAMP 原题是独立数学留出。

只记住：**持续迭代不意味着可以持续试到某个偶然显著结果。**

停下来回答：为什么新方法需要新的确认协议和数据来源？

### 第 46 步：已完成的失败是研究结果

冻结路由对 offline SFT 的旧确认业务差为 −0.147 pp，95% CI [−0.592, +0.307]；独立复验为 −0.202 pp，[−0.936, +0.577]。它们没有支持稳定优势。

来源：`reports/advantage/CONFIRMATION_AOPD_DOMAIN_ROUTER.json`。

只记住：**不能用后来的开发改善覆盖这个失败。**

停下来回答：哪个区间事实阻止了“已证明提高”的表述？

### 第 47 步：DEV 规则胜出不等于质量优势建立

最近 last2 平均规则在 DEV 上对 offline 的业务差是 +0.122 pp，但只有 2/4 种子为正。last2 对 final640 仅约 +0.007 pp；这些结果只支持继续开发。

来源：`reports/advantage/ROUTER_STABILITY_DEV.json`。

只记住：**选择一个规则和证明稳定优势是两个步骤。**

停下来回答：这组数字允许发布哪一种结论？

### 第九阶段检查

用自己的话区分：开发选择、多种子稳定、独立 test、复验和持续检验。然后回到 README，判断项目当前究竟在哪一层。

---

## 第十阶段：开始运行和读证据

### 第 48 步：main 是运行入口

代码、测试、配置、历史报告和本教程都在 `main`。从独立仓库安装，避免把父项目的 Python 包或 pytest 配置带进实验。

```bash
python -m pip install -e '.[data,profiling,test]'
python -m pytest -q tests
```

CUDA PyTorch 应与宿主驱动匹配。`models.lock.json` 和资产 manifest 固定实际模型文件。

只记住：**运行目录和依赖也是复现输入。**

停下来回答：为什么“我用了同名模型”不足以说明版本一致？

### 第 49 步：先用原训练配置做一个有界采集

在已准备模型、训练数据和 checkpoint-0160 的环境运行：

```bash
python scripts/profile_run.py \
  --profile configs/monitoring/baseline.json \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-baseline
```

环境变量是你自己的绝对路径。原 training config 必须匹配 checkpoint。第 161 步 warmup、第 162 步观测；其他起点用 `--first-update` / `--last-update` 显式调整。

只记住：**先确认基线恢复与目标 update 完整。**

停下来回答：checkpoint 已到第 320 步时，直接使用第 162 步的配置会发生什么？

### 第 50 步：安装官方工具后逐个采集

```bash
python scripts/install_profile_tools.py --root "$TOOLS_ROOT" --tools nsys ncu
python scripts/profile_run.py \
  --profile configs/monitoring/ncu.json --tools-root "$TOOLS_ROOT" \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --config "$TRAIN_CONFIG" --resume "$CHECKPOINT" \
  --output results/profile-ncu
```

`tools.lock.json` 固定包版本和 SHA256。工具缺失、counter 权限不足或没有真实 trace 都是失败记录，不自动当作通过。

只记住：**工具安装成功与采集成功要分别检查。**

停下来回答：仅有 `ncu --version` 输出，证明了什么？

### 第 51 步：按证据文件判断结果

从 `manifest.json` 核查输入，从 `tool-audit.json` 核查退出码，从 `validity.json` 核查数值和窗口，从实际 trace/CSV 阅读采集数据。`outputs.json` 提供产物哈希。

完整字段、命令和局限见 [PERFORMANCE_MONITORING.md](../PERFORMANCE_MONITORING.md)。

只记住：**必须同时存在元数据和对应的真实产物。**

停下来回答：一个写着 passed 的 JSON，为什么还不能替代 trace 内容核查？

### 第 52 步：doc 分支保留研究过程

`doc` 在同一仓库保存对话摘录、手稿和实验决策。`scripts/open_reading_workspace.sh` 将 main 与 doc 放在两个并排目录，并生成 VS Code 多根 workspace。

```bash
bash scripts/open_reading_workspace.sh ../ServeDistill-doc
code ServeDistill.code-workspace
```

只记住：**两个分支需要两个工作目录才能同时阅读。**

停下来回答：只在一个目录执行普通 branch switch，能否同时显示两个分支的文件？

### 第 53 步：完整成本包括失败与准备

GPU·h 使用每个 Pod 的实际分配区间乘卡数，不只统计训练循环。准备、标注、评测、失败和等待边界分开记录。未知区间不能用估计数字补成已审核成本。

只记住：**更高 tokens/s 不自动意味着更低有效结果成本。**

停下来回答：失败试验消耗的 GPU 时间是否应从总账删除？

### 第 54 步：把结论限制在实际完成的验收

历史 expanded 对照测得 learner 2.98×、完整训练 1.98× 的速度变化；业务绝对质量未通过。新 anchor 方法目前只有实现、CPU 回归和任务提交证据，仍需最终模型、独立评分与完整成本。

只记住：**哪个层次实际通过，就报告哪个层次。**

停下来回答：现在是否已经能写“稳定质量优势已建立”？

### 第十阶段检查

现在你应能运行一个观测副本，解释每类产物，读懂失败报告，并从 main/doc 同时查看实现与研究决策。后续先跟踪冻结 DEV 协议的四对种子结果，再决定是否具备开启新独立确认的依据。
