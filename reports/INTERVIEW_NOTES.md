# ServeDistill 面试讲法

负责公开搜索业务的大模型到小模型蒸馏闭环，用 Qwen3.5-9B→2B，以独立 ESCI 标签校验 teacher 资格、业务提升与通用回退；实现OPD/GRPO/混合λ/交替/动态/分桶/全KL等两种子对照，28阶段各80更新。
最终 full_kl 业务综合分 49.25%，相对初始变化 +23.39pp；通用子集变化 +17.75pp。各域置信区间和负结果必须同时披露，不能称为真实业务收入增益。
最终替代品回退触发5pp质量门，保留拒绝上线结论；离线SFT赢得dev选型，未证明OPD优于它。

工程难点是行为概率契约：BF16 rollout/recompute 不一致被验收门拒绝，通过 FP32 student+PyTorch DeltaNet 保住精度，并单独核算成本；不是靠调大容忍阈值。checkpoint 保存 adapter/Adam/RNG/controller，冷恢复指纹通过；第二阶段恢复能力另开 optimizer，避免混淆两个实验。

完成原生 vLLM 合并模型、流式请求ID/usage/截断校验及无标签协议回退压测；累计 59.75 H20 小时。按每百万有效答案计算 GPU 成本，明确本地HTTP与线上SLA的差异。

面对“为什么不用SFT/GRPO”：先看 teacher 是否有该域 headroom、标签是否够好、reward是否可靠，再引用实测消融；不用固定流程代替业务选择。面对“token级还是轨迹级”：信号与聚合是token/trajectory，optimizer每批更新，不在生成中切换权重。面对“多专家”：本轮是提示路由，独立权重专家融合尚未验证。

不要声称：异步OPD吞吐提升、完整通用能力恢复、真实团购转化率、teacher全参数逐元素哈希、固定tokens/算力公平最优或已满足线上SLA。
