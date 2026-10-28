# rollout batch1 memcheck 终态

无 profiler 基线完成 update162 并通过运行门；Compute Sanitizer 在 3,600 秒上限退出 124，没有完成 update161/162，也没有输出 `ERROR SUMMARY`。本次 memcheck **未通过**。现有日志不能把这次超时判为 OOM，也不能把没有错误汇总解释为零错误。

严格恢复原 checkpoint 后，仅诊断副本的 rollout batch 从 8 改为 1。每步仍含 10 个 prompt、group2、max128 tokens，learner microbatch1，两个副本均使用 no_cudnn 和关闭 expandable segments 的 allocator。基线与仪器化副本共享恢复起点和观察配置；改变 rollout batching 可能改变生成轨迹。

25 个基线索引文件和 16 个 sanitizer 索引文件均核验大小及 SHA256；51 个 terminal 源文件分别记录原始指纹和公开导出指纹，见 [导出审计](sanitizer-rollout1-probe-v1-terminal/PUBLIC_EXPORT_MANIFEST.json)。父任务退出 0 只说明收齐子测试；sanitizer 的应用退出码与未完成窗口仍使整套诊断失败。

分配 4,730 秒，即 **1.313889 H20·h**，包含准备、基线、超时和导出。原始收集器成本说明使用了通用 DEV 文案；本任务实际为两份恢复窗口诊断，区间与数值不变。见 [成本](SANITIZER_ROLLOUT1_V1_COST.json) 和 [独立复核](SANITIZER_ROLLOUT1_V1_REVALIDATION.json)。

本次无 memcheck 最终 adapter/optimizer 可用于数值对照，因此不能宣称 memcheck 与基线等价。基线通过仅证明所声明运行窗口，不证明业务质量、完整模型内存安全或吞吐优势。
