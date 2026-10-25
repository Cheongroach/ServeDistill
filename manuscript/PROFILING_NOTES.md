# H20 训练观测：定位、对照与解释边界

2026-10-24；对应 main `7733cce`。原始质量研究之外的追加诊断，完整实测与复现命令见 [profiling report](../reports/profiling/REPORT.md)。

用户要求：做 Nsight 一类的实际观测和工具检查，展示跟踪方法，选代表片段解释结果及参数影响。使用原来的 Qwen3.5-9B→2B，单 H20，恢复 seed17/update40 的 adapter、Adam、RNG，预热 update41，捕获 update42。

## 从概率契约到执行代价

此前用 FP32 Student 与 PyTorch DeltaNet 路径通过概率一致性门。观测发现其代价：m1 一次更新 151 万 kernel，大量 1–3 μs 的 elementwise；Backward 主机范围显著长于关联 kernel 的累计设备时间。保住数值正确性是前提，不能以切 BF16、放宽门限来抹掉这个瓶颈。

Teacher 虽为9B，BF16/FLA 的执行路径不同。Reference 是禁用同一 Student adapter；Full-KL 的参考 KL 仅作漂移监控。Teacher 规模、Teacher 监督质量、Teacher forward 延迟是三个不同问题。

## microbatch 的对照结果

同采集参数：m1→m4，20条→仍20条序列、283有效 response token、相同样本/token/mask；learner microbatch 循环20→5次，optimizer仍每批一次。

- kernel 数：1,509,666→544,560，−63.9%。
- 捕获 update：36.341→15.039s；CUDA kernel/copy/memset 活动覆盖27.0%→45.8%。
- 进程 PyTorch allocated 峰值：24.809→27.107GiB，+2.30GiB。
- 概率门禁通过，最大误差6.70e-5/4.67e-5；norm12.996→12.818、loss0.333260→0.335480。

没有把结果包装成“同质量2.42倍训练加速”：profiler开销、运行顺序、编译缓存和微批引起的数值差异都需要界定。warmed m1 baseline28.663s，nsys36.341s；cold-shape m4 baseline54.466s，后续nsys15.039s。缺 warmed无profiler m4 的重复分布；2.42倍只是本次带观测窗口时间比。update41也用了不同microbatch，不能声称update42参数逐位相同。

## 工具边界也是结果

CUDA/NVTX 实采成功；system-wide CPU sampling受权限限制，ncu未安装，所以不报SM occupancy/HBM带宽或Roofline。CUDA活动时间并集只回答“什么时候有GPU活动”，不是硬件单元利用率。

整步 PyTorch shapes 首次超过240s；改为第一个m4 microbatch后得到626,070事件、318,632 CPU op、63,612实际CUDA kernel。Student MLP `[1088,2048]×[2048,6144]`、词表projection `[120,2048]×[2048,248320]` 把发射时间连到实际shape。

Backward 的 GPU annotation只有87.522μs而CPU范围1.135s，不能把主线程label当作全部autograd GPU工作。Nsight的correlation join约束PID，另保留cross-thread temporal归因；PyTorch统计仅把`cat=kernel`计作kernel，不把scope/op/kernel行的百分比相加。

## 参数优先级与后续验证

先改善learner microbatch和细碎DeltaNet执行，再验证融合/精度路径；逐项核验概率、梯度、显存及业务质量。`B×Tresponse×V`解释Full-KL显存；实际m4、T=30、V=248320的一份FP32 logits约113.7MiB，多份中间张量另算。

`group_size`改变轨迹预算，`rollout_batch`改变生成执行粒度，`microbatch`改变训练累积粒度；三者不能混称batch size。`full_kl`直接用OPD loss，调`opd_weight`或PPO clip不直接改变当前训练目标。checkpointing换显存为重算，`dev_every`改变评估与驻留成本，均不能仅从update42推算总训练预算。

下一步需重复稳态无profiler m1/m4，固定轨迹和同一模型状态再分离Teacher batch舍入/累积因素，并在概率与质量门通过后评估融合核或BF16。此处记录验证方向，没有把未执行工作写成收益。

追加两次allocation均已释放：0.8686H20 GPUh，含失败shape采集、准备和发布；原质量研究59.7544GPUh，累计60.6231GPUh。32 CPU回归通过，第二轮六case全部退出0，原始trace和插桩源码hash回读一致。

可用的面试表述：负责把业务OPD训练从数值验收推进到可观测执行；用NVTX、CUDA correlation和PyTorch shape识别FP32参考路径的百万级kernel发射，对照microbatch减少63.9%kernel、增加2.30GiB显存；同时识别profiler/编译冷启动混杂，保留数值差异和稳态验证缺口，避免把观测窗口时间比宣称为同质量生产加速。

2026-10-25 后续：固定轨迹重复计时和四种子扩展已完成，见 [EXPANDED_STUDY_NOTES.md](EXPANDED_STUDY_NOTES.md)。新 workload 的无 profiler learner 吞吐比为 2.98×，完整训练时长比 1.98×；同时发现 substitute 全 `no` 退化。上述原诊断数字与混杂边界保持不变，不能用新结果改写旧捕获的实验条件。
