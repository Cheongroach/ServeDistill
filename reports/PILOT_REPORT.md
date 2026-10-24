# Qwen3.5 H20 试跑与资格评估

2026-10-24。此报告为完整实验前的运行验收，不是最终能力或吞吐收益结论。

模型：Qwen3.5-9B teacher → Qwen3.5-2B student；关闭 thinking；原生 text-only 组件。训练使用 FP32 student / BF16 teacher，部署评测使用 BF16 合并 adapter。H20 单卡，LoRA 可训练参数 15,630,336。

BF16 第一轮产生两次非零更新，冷恢复 adapter、Adam、RNG 指纹一致，但下一批 cached rollout 与 teacher forcing 重算误差 0.193709，超过 0.15 门限，拒绝更新。对 10 条训练样本的诊断：BF16 最大误差 0.148020，仅输出层 FP32 为 0.089907，全 student FP32 + PyTorch DeltaNet 为 0.000046253。诊断变体顺序存在 kernel warmup 差异，耗时不作为精度路径的公平性能对比。

采用全 FP32 student 后，四次非零更新、teacher 冻结和冷恢复均通过，最大概率误差 0.000045538，共 692 个有效 response tokens。完成记录中的 elapsed_seconds 和 phase_seconds 属于恢复进程；不得直接解释成四次更新的总时间。仅 4 条/域的 dev 用于运行验收，不作为能力增益依据。

完整资格评估为 100 条/域，沿用独立 ESCI 标签和固定 dev 划分：

| 域 | student | teacher | 指标 |
|---|---:|---:|---|
| 搜索排序 | 0.2818 | 0.7223 | NDCG，线性 gain，97 个有效 query |
| 相关性 | 0.1667 | 0.3375 | E/S/C/I macro-F1 |
| 替代品 | 0.4409 | 0.4012 | binary macro-F1 |
| 指令 | 0.2900 | 0.8500 | 确定性成功率 |
| 数学 | 0.1000 | 0.3200 | GSM8K 子集成功率 |

替代品 teacher 不满足能力 headroom 假设；完整实验保留该域作为负对照。资格 CI 见 qualification-paired.json，不能由小类别切片得出稳定业务结论。

冻结矩阵：两种训练种子，每种 13 个一阶段对照，80 次更新；另做业务 SFT 后通用能力恢复的二阶段实验，以及 2B/4B/9B 质量、延迟、成本对照。独立更新数、有效 tokens、驻留 GPUh 分别报告。

旁路专用基线 Qwen3-Reranker-0.6B 在冻结的 300 条搜索 test 上 NDCG 0.8807，查询评分 P50 71.2ms、P95 102.9ms。边界为进程内对全部候选评分，不含检索或 HTTP，也不能据此预先判断与最终蒸馏模型的优劣。
