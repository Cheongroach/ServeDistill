# OPD 配方核对：后续假设，不改变当前实验

2026-10-25。本记录只核对原始方法及已观察开发证据。当前 Teacher240 衰减轮、Teacher320 后续门禁、四配对种子、双强 SFT 和双独立留出集规则保持原样。

## 原始方法

- [Thinking Machines，2025-10-27](https://www.thinkingmachines.ai/blog/on-policy-distillation/#loss-function-reverse-kl)：Student 自采样，Teacher 对同一响应 token 评分，优势为 stop_gradient(log p_teacher - log p_student)，以 importance-sampling policy loss 更新。其实现明确取 discount=0，并报告 discount>0 未改善其实验。当前 losses.opd_advantages / train.apply_rollout 与该即时信号一致；没有累计回报不能直接诊断成实现错误。这是对该来源配方的核对，不是本项目效果验证。
- [MiniLLM，Eq. 2–3，2024-04-10 版本](https://arxiv.org/html/2306.08543v4#S2.SS2)：完整序列 reverse-KL 梯度包含后缀回报，进一步分解为单步全词表项和后续回报项；还使用教师混合采样和长度归一化。它与当前 sampled-token、纯 Student 采样配方不同，论文收益不证明本项目已复现同算法或会得到同收益。
- [GKD，2023-06-23](https://arxiv.org/html/2306.13649v1#S4)：允许不同 divergence 和采样来源；其直接条件分布损失不反传采样分布。它与单个 sampled-token policy-gradient 信号应分开描述。
- [Rethinking OPD，2026-04-14，§3、§5–6](https://arxiv.org/html/2604.13016v1)：在其推理任务中，教师分数更高并不保证有效迁移；公共高概率 token 的匹配、教师新增能力和初始化可能影响结果。其 cold-start 配方先做 Teacher 轨迹 SFT，再做 OPD。该机制尚未在本项目 ESCI 短 JSON 输出上验证。
- [AOPD，2026-05-13 版本](https://arxiv.org/html/2605.06387v3#S5)：在非正优势位置引入截断 forward-KL 指导，改变当前纯 sampled-token OPD 损失。其收益不能作为当前纯 OPD 的结果。

上述配方定义各自直接引用原始来源；跨任务机制仍属于待验证假设，没有将互相引用的论文当作三个独立实验复现。

## 本项目证据与缺口

当前 Runtime 固定采样 temperature=1、top_p=1、top_k=0，并从实际生成 scores 保存 behavior logprob。不能凭未调温度猜测 rollout/learner 失配。DECAY640_OBSERVED_RUNTIME.json 逐步验证重算误差、实际学习率及总损失等于 OPD 损失；它不替代最终权重、Adam/RNG、质量和成本审计。

已逐行重判的 Teacher 实际提示 DEV：Teacher240 business=0.7208744 / general=0.62；Teacher320 business=0.7178789 / general=0.65，均为 1,100 行 BF16 开发评分。Teacher320 业务分略低，不能预先称为更强教师。其新合并模型仍需独立 GPU 资格及全部无过滤训练标签验证。

DECAY640_CLASS_DRIFT.json 的完整训练曲线使用 FP32、1,400 行；部分 sampled KL 随训练下降，业务指标未同步稳定提高。平均 KL 下降只证明该描述量变化，不能证明类别决策 token 已校准，也不能推断损失方差、公共 top-k 概率质量或教师信号的因果作用：这些逐 token 分布尚未采集。

## 条件性后续假设

若完整 180 候选及全部来源、终态成本闭合而选择门仍明确拒绝优势，可在既有 train/dev 上检验 Teacher 标签产生的公共 Student 初始化。OPD、offline 和 oracle 必须共用同一真实起点、fresh Adam、配对 prompt 和共同更新预算，并保留不继续训练的起点对照，防止通过过训 SFT 制造优势。公共预训练成本对所有方法一致入账。OPD 阶段保留完整正负即时信号；初始化收益、纯 OPD 方法收益和成本收益分别判定。

这只是文献支持的待验证假设，不是已准备、已提交或获益的实验。当前全池通过时原确认路线优先；一旦真实确认冻结或留出集输出存在，禁止据此继续开发选型。本文没有读取独立留出集输出、改变损失或阈值、申请 GPU 或提交任务。
