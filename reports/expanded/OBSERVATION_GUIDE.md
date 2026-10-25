# 如何复核这次 Full-KL OPD 观测

最终数值以 [REPORT.md](REPORT.md) 和对应 JSON 为准；终态验收见 [STATUS.md](STATUS.md)。本说明解释怎么读产物、怎样判断参数影响，主实验终点保持冻结。

## 先确定正在回答哪一个问题

| 问题 | 应读证据 | 统计单位与边界 |
|---|---|---|
| microbatch 是否提高 learner 吞吐？ | `performance.json`、`performance-measurements.jsonl.gz` | 同一父状态和冻结轨迹，32 bank 配对；不含 rollout、状态恢复、profiler |
| 完整训练是否更快？ | `training-summary.json`、`training-steps.jsonl.gz` | 含 rollout、编译、dev、checkpoint；最终评测单独计时，输出长度可能变化 |
| 时间花在哪里？ | `profiles/profiles.json`、各 `*-cuda-summary.json`、时间线 PNG | 已预热 bank0 的实际捕获；只解释结构，不代替性能分布 |
| microbatch 是否改变数值？ | `validation.json`、`numerics.json`、本地 `raw/numerics-v1/vectors.pt` | 原生 Teacher 路径与固定 Teacher 诊断分开；均为 post-clip 梯度 |
| 训练质量是否保持？ | `quality-test.json`、`quality-ood.json`、`task-metrics.jsonl.gz` | query 配对，绑定四对种子；业务非劣效下界为 −5 pp |
| 绝对分类能力是否有用？ | `classification-audit.json` | 查看每类 support/预测数/recall；非劣效可以保留一个常数分类器 |
| GPU 成本是多少？ | `cost-audit.json` | Pod 调度到释放，包含准备、编译、评测、等待；Running 时只是下界 |

`quality-ood.json` 沿用代码的 split 名；本实验是独立 query replication，没有制造真实分布偏移。

## 读一个代表性 learner microbatch

打开 `profiles/worker0-m1-timeline.png` 和 `profiles/worker0-m4-timeline.png`，对照同名 `*-cuda-summary.json` 的 `representative_window`。时间原点是 `fixed_bank_update`；代表窗口从第一个 `learner_forward` 开始，到第一次 `backward` 主机范围结束。完整 update 的最终 CUDA 同步也在捕获内。

按以下顺序复核，避免把异步主机调用和 GPU 完成混在一起：

1. **learner_forward**：Student 的 forward、词表投影、logprob 和 entropy。Student 为 FP32/PyTorch DeltaNet；看到重复的小 kernel 时，先查它们的调用次数与 CUDA launch correlation。
2. **teacher_forward**：Teacher 对相同 response 位置打分，使用 BF16/FLA。NVTX 是主机范围；随后有显式 CUDA 同步。不能直接用这一范围的宽度代表全部 Teacher GPU 时间。
3. **full_vocabulary_kl**：计算每个位置的全词表 reverse KL。显式范围很短，也不意味着全词表监督便宜；词表投影位于两个 forward 内。
4. **reference_forward**：关闭 adapter 后的 reference 打分，仅用于监控 sampled KL；它仍消耗真实 forward 时间和显存。
5. **backward**：反传包括 activation checkpoint 重算。被关联到 backward 的 forward 类 kernel 可以是重算，不能仅凭 kernel 名称归因。
6. **optimizer**：所有 learner microbatch 累积后只执行一次 AdamW。完整 update 还包含有限值检查、clip、参数 delta 与同步，因此阶段 NVTX 之和不必等于完整 update 时间。

`phase_device_launch_attribution` 通过 CUDA runtime correlation ID 并约束进程关联。autograd 跨线程的时间关联保留 `cross_thread_temporal` 标记；它的归因强度低于同线程范围。对争议片段回到原始 `.nsys-rep`，同时查看 NVTX、CUDA API 和 stream。

| 观测量 | 可支持的判断 | 不能直接支持的判断 |
|---|---|---|
| kernel 次数、小 kernel 时长、launch 密度 | 发射结构是否变得稀疏，重复调用是否减少 | SM occupancy、HBM 带宽 |
| CUDA kernel/copy/memset 区间并集 | 捕获 update 中至少一项 CUDA 活动覆盖了多少时间 | 每个 SM 都忙、设备没有带宽瓶颈 |
| kernel 时长之和 | 某类 kernel 累积工作量 | 存在并发 stream 时的实际墙钟占比 |
| NVTX 主机范围 | 主机调用顺序和范围 | 所有对应 GPU 工作的完成时刻 |
| PyTorch allocated/reserved 高水位 | 该步骤的框架分配峰值 | 全进程显存、模型静态权重的精确分解 |

本实验没有 NCU 的 SM/HBM 实测结果。若以后补采，只对已预热、可重放的一小段选定 kernel 采集，并将 NCU replay 开销排除出吞吐比较。

## 参数为什么会影响耗时、显存和训练信号

当前每一步为 8 prompts × 4 completions = 32 trajectories。microbatch 取 1/2/4/8 时，learner 累积循环分别为 32/16/8/4 次；optimizer 都只更新一次。

token aggregation 使用每个 microbatch 的有效 response token 数作权重：

\[
L=\sum_j \frac{n_j}{\sum_k n_k}L_j,\qquad
L_j=\frac{1}{n_j}\sum_{b,t\in j}m_{bt}
\sum_v p_\theta(v\mid x,y_{<t})
\left[\log p_\theta(v\mid x,y_{<t})-\log q(v\mid x,y_{<t})\right].
\]

因此不能把每个 microbatch 的均值直接等权相加。相同数学目标也不保证实际梯度逐位一致：Teacher batch/padding、浮点累积顺序与不同 shape 的执行路径都可能变化。固定 Teacher logits 的诊断帮助分离这些因素；训练质量由独立最终评测回答。

| 参数 | 直接改变什么 | 应同时跟踪什么 |
|---|---|---|
| `microbatch` | learner forward/backward 的批量与循环次数 | 无观测 token/s、allocated/reserved、padding、梯度差、最终质量 |
| `prompts_per_step` | 每步不同 prompt 数与总轨迹数 | prompt visits/unique prompts、有效 token、每步时间；固定 updates 下预算会变化 |
| `group_size` | 同一 prompt 的生成数与总轨迹数 | 生成预算、重复输出、EOS 长度；本次 Full-KL 主损失不采用 group-normalized reward |
| `rollout_batch` | 生成时并行批量 | rollout 时间、生成显存与 padding；它不等于 learner microbatch |
| `max_new_tokens` | 生成长度上限 | 实际 response 长度、截断率与任务成功率；上限不是实际 token 数 |
| Student 精度/执行路径 | 算子实现、吞吐与概率误差 | rollout/recompute 同一概率契约、有限值、梯度与质量；本次没有修改 |
| checkpoint 重算 | 保存 activation 与额外 forward 的交换 | backward kernel、update 时间与显存；本次没有单独做开关消融 |

单份 FP32 **response 范围** logits 的规模约为
`4 × microbatch × Tresponse × 248,320 bytes`。当 microbatch=4、Tresponse=128 时约 0.474 GiB；还未计入其他 logits/probability 张量、activation、梯度与临时区。完整 forward 还投影 prompt 位置；batch 内最大长度及 mask 决定实际 padding。不能拿这个公式当作总显存估计。

microbatch 增大通常减少重复发射，但更高 padding 和活跃张量可能抵消收益。m8 是否优于 m4 应读实际配对数据，不能预设单调加速。实际 on-policy response 长度也可因参数轨迹改变，因此端到端时间比不能替代固定轨迹的执行对照。

## 五个可在已保存产物上完成的练习

1. 从 `performance-measurements.jsonl.gz` 排除 `warmup` 和 `profiled`，对每 bank/microbatch 的三次耗时取中位数。复算 `sum(bank_tokens) / sum(bank_median_seconds)`；检查结果与 `performance.json` 一致。不要平均逐次 token/s。
2. 保持 bank 配对，复算 m4/m1 的时间总和之比，再按 bank bootstrap。解释为什么 384 次正式计时仍只有 32 个独立 workload 单位。
3. 在两个代表窗口中找调用数最高的小 kernel，核对 runtime correlation 与所属 NVTX；再比较 kernel 次数、活动区间并集和完整 update 时间。三者变化不必成比例。
4. 读取 `training-steps.jsonl.gz`，按 arm 复算 response token ledger、平均轨迹长度与截断率。检查 `8 × 4 × 240` 条实际轨迹是否完整，而不是把 `128 × 轨迹数` 当成已生成 token。
5. 阅读每域 `paired_query_ci95` 与 `paired_seed_deltas`。即使均值接近，只要 CI 下界没有超过 −5 pp，该域仍未确定；全部业务域分别通过才支持预设的联合非劣效结论。再读取 `classification-audit.json`：substitute 的每个最终模型都只预测 `no`，正类 recall=0。复算 Test 常数多数类 macro-F1：`972 / (2×972 + 528) = 39.32%`，解释为何配对差与 CI 全为零仍不代表可用的替代品识别能力，也为何四种子限制随机性外推。

复算入口为 [analyze_expanded.py](../../scripts/analyze_expanded.py)、[expanded_statistics.py](../../servedistill/expanded_statistics.py)、[analyze_profile.py](../../scripts/analyze_profile.py)。主损失与阶段边界直接对应 [losses.py](../../servedistill/losses.py) 和 [train.py](../../servedistill/train.py) 的 `apply_rollout`。原始文本、checkpoint、向量与 trace 保留在本地 raw/；公开 gzip 指标不包含生成文本。
