# 从单窗口观测到配对扩展实验

2026-10-25；对应 main `233cd27`。完整结果、统计和复算入口见 [扩展报告](../reports/expanded/REPORT.md)，代表时间线与练习见 [观测指南](../reports/expanded/OBSERVATION_GUIDE.md)。本页记录设计判断和证据边界，不替代原研究的负面结果。

## 原问题与本轮改变

单个 update 的 283 个 response token 和百万级 kernel 足以提示细碎发射，不能证明稳态加速，更不能证明改变 microbatch 后训练质量保持。原捕获还混有 profiler、冷形状编译和更新状态差异。扩大实验首先解决对照设计，而不是单纯增加 GPU 数量。

- 执行对照：32 个不重复 bank，每 bank 8 prompts × 4 completions；固定父 adapter/Adam/RNG、token、EOS mask 和 behavior logprob。microbatch=1/2/4/8，每格先预热，再正式重复 3 次；状态恢复在计时之外。384 次正式计时只有 32 个独立 workload 单位。
- 质量对照：四对种子，从共同 update40 fork；重置实验步号、累计 token、监控历史和种子 RNG。每 arm 240 updates，相同 prompt schedule；固定 update240 为主终点。共同父状态限制外推，不能称为四个独立初始化实验。
- 数据隔离：新训练池 14,000、dev 1,400、test 6,600、replication 2,200；排除原 query/input，新增 split 之间 query/input overlap=0。商品 ID 仍可重复，replication 没有制造真实分布偏移。
- 统计冻结：性能对 bank 中位耗时做配对 bootstrap；质量对 query 重采样并绑定四对种子。每业务域 −5 pp 为非劣效界限，联合结论要求每域通过。m8/m4 和逐类退化审计另标为探索性/描述性分析。
- 路径固定：Student FP32/PyTorch DeltaNet、Teacher BF16/FLA、LoRA r=16、Full reverse KL、token aggregation、lr=2e-5、clip=1。没有同时改变精度、融合 kernel 或损失。

## 实际工作与性能证据

完成 1,920 次非零更新、61,440 条训练轨迹、881,401 个有效 response tokens。四对种子实际访问 5,776 个不同 prompt；不能把 14,000 条训练池全部说成已训练样本。每 arm 平均 response 长度约 14.3–14.4 token，128 只是生成上限。

| microbatch | learner token/s | 对 m1 / bank 95% CI | allocated 峰值，GiB |
|---|---:|---|---:|
| 1 | 10.11 | 1.00 | 25.33 |
| 2 | 18.68 | 1.85 / [1.83, 1.87] | 26.62 |
| 4 | 30.15 | 2.98 / [2.88, 3.10] | 29.20 |
| 8 | 30.84 | 3.05 / [2.83, 3.32] | 34.39 |

本 workload 更支持 m4：m8/m4=1.023，CI [0.978, 1.076]，未确立额外稳定收益，却增加 5.19 GiB allocated。显存是 PyTorch 步骤高水位，不是全进程显存。

四种子完整 train 时间总和之比为 1.98；m1 约 209–211 分钟，m4 约 105–108 分钟。这个边界包含 rollout、形状编译、dev 和 checkpoint，排除 final eval 与外层环境准备；不能用 learner 的 2.98 倍替代端到端收益。实际 on-policy 输出略有差异，完整训练时间比不是固定轨迹吞吐 CI。

## 非劣效通过与分类退化同时成立

8 个最终 checkpoint 各覆盖全部 test/replication，共 70,400 条主评测输出；父状态另有 8,800 条。3 个 dev-selected 较早 checkpoint 另完成 26,400 条，其他复用最终 checkpoint。所有输入 hash、覆盖和独立重判通过。dev-selected 摘要单独存放，未用 test 重选主终点。

| 域 | Test m4−m1，pp / query 95% CI | Replication，pp / query 95% CI |
|---|---|---|
| search | +0.25 / [−0.09, +0.59] | −0.19 / [−0.73, +0.33] |
| relevance | −0.18 / [−0.66, +0.21] | −0.15 / [−0.75, +0.47] |
| substitute | 0.00 / [0.00, 0.00] | 0.00 / [0.00, 0.00] |
| math | −0.21 / [−0.75, +0.38] | +0.12 / [−0.87, +1.12] |
| instruction | +0.98 / [+0.30, +1.63] | +1.40 / [+0.15, +2.70] |

三业务域均通过预设非劣效。但逐类检查发现 **8 个最终 substitute 模型都只预测 `no`，正类 recall=0**。Test 的 528/1,500 条正类全部漏掉；其 macro-F1=39.32%，恰好等于常数多数类基线。Replication 同样退化。差和 CI 全为零来自相同常数预测，不是有用能力的等价证据。相关性分类也偏向 E/I、S/C recall 薄弱，逐类证据保留在 [classification-audit.json](../reports/expanded/classification-audit.json)。

这些 query CI 条件于四个观察到的种子；探索性种子 CI 更宽，instruction 的种子区间包含零。因此不把正向 query CI 宣称为跨训练随机性的稳定增益，也不声称 OPD 优于 SFT。原研究的 substitute 负迁移和跨 schema 失败仍成立。

## 数值证据不能被一个通过标记压缩

概率 tolerance 沿用父实验 0.15；实际性能/训练最大误差为 8.87e-5 / 1.13e-4。Teacher requires_grad=False，冻结指纹检查每个命名参数前 32 个元素，不是完整权重逐位 hash。

原生 m4 对 m1 的 post-clip 梯度相对 L2 中位 3.02%、最大 10.96%；参数 update 相对 L2 中位 1.56%、最大 4.89%。Teacher batch/padding、累积顺序和 shape 路径都可能贡献差异。未保存完整跨 microbatch token logprob 差和全 bank pre-clip 向量，不能宣称训练轨迹等价。

原 FP32 长向量 cosine 对自比较产生 >1 的非法值，保留原字段但排除其结论。独立固定 Teacher 四序列 GPU 检查使用 FP64 归约：cosine=0.9999999999335603，梯度相对 L2=1.1508e-5，update 相对 L2=2.0127e-5。小片段差异缩小支持继续审查 batch 路径，但不能证明原生差异全部由 Teacher 引起。

## Nsight 如何解释收益

四个已预热捕获验收完成。worker0 的 m1/m4 kernel 数为 2,066,865 / 519,910；worker1 为 2,163,697 / 543,229，均减少约 75%。每 update 32 条序列的 learner 循环由 32 次减到 8 次，optimizer 都只更新一次。

worker0-m4 第一个 microbatch 的主机 NVTX 范围：Student 326.10ms、Teacher 88.75ms、显式 KL 0.83ms、reference 229.39ms、backward 813.18ms。这些是主机调用范围；异步 GPU 完成另看实际 kernel 区间。Backward 包含 checkpoint 重算，reference 是关闭 adapter 后的监控 forward。

Full-KL 显式范围很短不代表全词表监督便宜，投影和 entropy 仍在 forward 内。单份 FP32 response logits 为 `4×B×Tresponse×248,320 bytes`；B=4、T=128 约 0.474GiB，未计其他中间张量，完整 forward 还投影 prompt 位置。

CUDA correlation join 约束进程，autograd 跨线程的时间归因保留较弱标记。CUDA 活动并集不是 SM occupancy/HBM 带宽；没有 NCU 实测，不能做底层硬件利用率结论。Profiler 时间完全排除出主吞吐 CI。

## 验收、成本和下一步优先级

完整 CPU suite 的 50 项日志通过，但工具因 `/proc/.../task/.../comm` sandbox 限制返回非零，不记成干净的整套 shell 成功。5 项 artifact guards 独立退出 0；实际 GPU 终态产物、token ledger、主/次级评测覆盖、类别指标重算、源码一致性、语法/JSON/diff 检查分别通过。

本轮 2 卡 performance + 8 卡 training 正常退出并全部释放；按 Pod 调度到释放计 **43.6433 H20 GPUh**，包含准备、编译、评测、等待和释放。加原研究/观测，项目合计 **104.2664 GPUh**。价格未提供，不虚构货币成本。原始 checkpoint、向量与 trace 保留本地和持久化存储；公开计分、计时、steps、图表与 hash 排除生成文本及内部标识。

若继续追求可用业务模型，优先修复 Teacher 的类别能力和 Student 的正类覆盖，再做 SFT/OPD 的预算对照与 schema 回归。单纯继续扩大学习步数已经不足以解决全 `no` 的 substitute 结果。若追求执行优化，再逐项验证融合/精度路径，并重新过概率、梯度、显存和任务质量门；这些是后续方向，未报告未执行收益。

## 可复用的 Owner 表述

负责将 Qwen3.5-9B→2B OPD 的单窗口性能诊断推进到可复核的执行与质量对照；设计固定状态/轨迹的 32-bank 配对计时和四种子训练，完成 1,920 更新与 88 万有效 token，在 43.64 H20 卡时内验证 m4 learner 吞吐 2.98×、完整训练 1.98×。结合 Nsight 和显存判据舍弃没有稳定追加收益的 m8，并通过逐类审计识别非劣效检验掩盖的常数分类器，保留业务质量不合格结论。
