# ServeDistill 扩展实验状态

2026-10-25 终态验收完成：两个任务均正常退出，全部 10 张实验 GPU 已释放。完整结果见 [REPORT.md](REPORT.md)，统计判据见 [冻结协议](PROTOCOL.md)，观测方法和复算练习见 [OBSERVATION_GUIDE.md](OBSERVATION_GUIDE.md)。

## 已完成工作

- 扩展包中的 59 个文件在 worker 端逐项校验；模型资产、源码与数据已完成复制和 hash 验证。
- 32 个互不重复的 workload bank、384 次正式计时、128 次预热；256 prompts / 1,024 条固定轨迹 / 14,375 response tokens。4 个 Nsight CUDA/NVTX 捕获通过验收。
- 两个性能 worker、8 个训练 arm 恢复共同 update40 的 adapter、Adam 和 RNG 指纹。四对种子完成全部 1,920 次 optimizer updates、61,440 条生成轨迹、881,401 个有效 response tokens；实际访问 5,776 个不同训练 prompt。
- 各 arm 固定 update240 的 6,600 test 和 2,200 replication 全覆盖，共 70,400 条主评测输出；父 checkpoint 另完成 8,800 条。input hash、独立重判、概率与有限值门禁通过。
- 3 个较早的 dev-selected checkpoint 另完成 26,400 条评测输出，完整覆盖与重判通过；其余 5 个复用最终 checkpoint。摘要见 [secondary-dev-selected.json](secondary-dev-selected.json)，不用于替换主终点。
- 执行关键源码、配置和数据的公开 hash 与 worker 一致；原始产物文件 hash 已保存。后续分析、测试、绘图和文档变更单独记录。
- 完整 CPU suite 日志显示 50 项测试通过，但执行工具因 `/proc/.../task/.../comm` 的 sandbox 写限制返回非零；不记为完整 shell 验证成功。后续针对终态、评测覆盖、生成轨迹、token ledger 与 runtime 合同的 5 项检查独立退出 0。CPU 测试不证明 GPU 性能或质量。
- 新训练池 14,000 条、dev 1,400 条、test 6,600 条、独立 query replication 2,200 条；排除旧 3,300 个 query group 与旧 5,500 个 input。新增 split 间 input/query overlap 为 0，但商品 ID 仍可跨 split 重复。

## 最终结论与限制

m4/m1 无 profiler learner 吞吐比 **2.983×**，bank 配对 95% CI **[2.880, 3.096]**；四对种子的完整 train 时间总和之比 **1.98×**。m8/m4 的探索性比值 1.023×，CI [0.978, 1.076]，未确立额外稳定收益；allocated 峰值从 m4 的 29.20 GiB 增至 m8 的 34.39 GiB。

两个 split 的三业务域 −5 pp 非劣效联合判据全部通过，但 **substitute 的 8 个最终模型都只预测 `no`，正类 recall=0**，相关性分类也偏向 E/I。非劣效不证明绝对业务质量；逐类证据见 [classification-audit.json](classification-audit.json)。此前 OPD 对 SFT 和能力保持的负面结果不因此改变。

原生 m4 post-clip 梯度相对 L2 中位 3.02%、最大 10.96%，参数 update 相对 L2 中位 1.56%、最大 4.89%。原始 FP32 cosine 曾对相同长向量产生 >1 的非法值，保留原字段但排除其数值等价结论。独立固定 Teacher 四序列 FP64 GPU 检查退出 0：cosine=0.9999999999335603，梯度相对 L2=1.1508e-5，update 相对 L2=2.0127e-5；该片段不能覆盖原生全 bank 差异。

性能 CI 以 bank 为单位；质量 CI 按 query 配对并绑定四个种子。重复计时和跨种子输出不增加独立样本数；replication 不是真实 OOD。未建立 SM occupancy、HBM 带宽或线上 serving P95 结论。

## 资源与成本

[终态成本](cost-audit.json)：扩展 **43.6433 H20 GPUh**，峰值 10 卡，活跃 0；计入准备、编译、评测、等待和释放。原研究 + 原观测 + 扩展合计 **104.2664 GPUh**。两个任务 exit_status=0，performance 于 2026-10-24 20:24:00 UTC、training 于 22:13:44 UTC 释放。本次计划内实验和产物验收均已完成。
