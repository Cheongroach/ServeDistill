# 关闭 expandable segments 后，memcheck 仍未完成

2026-10-28。这轮单卡诊断检验非 expandable allocator 能否让原完整 workload 通过 memcheck。两个副本从同一 checkpoint160 冷恢复，使用相同五个输入文件、完整 student/teacher、no_cudnn、learner microbatch1、rollout batch8；每步10个 prompt、group2、最多128个新 token。计划完整执行更新161/162，161为 warmup，162为观察窗口。

| 观察 | 无 profiler 基线 | Compute Sanitizer memcheck |
|---|---:|---:|
| 冷恢复 fingerprint | 通过 | 通过 |
| 完整 update162 | 通过 | 未到达 |
| wrapper 退出码 | 0 | 1 |
| 工具/application 调用退出码 | 0 | 86 |
| nvidia-smi 采样显存峰值 | 28,348 MiB | 97,266 MiB |
| capture 墙钟 | 284.662 s | 257.935 s |

memcheck 在更新161的 student prefill 中失败。堆栈位于 Qwen3.5 MLP（`modeling_qwen3_5.py:803`）调用 PEFT LoRA linear（`layer.py:807`）；日志记录6次 `cudaMalloc` / `cudaGetLastError` 内存分配错误，最终汇总为6 errors。PyTorch请求74 MiB时仅余18.38 MiB，已分配91.96 GiB。本次没有出现上一轮的 `cuMemCreate` 错误签名，但应用没有完成，不能由此认定 allocator 兼容性问题已解决。表中墙钟覆盖不同完成范围，不能计算速度比。

无 profiler 基线与上一轮 no_cudnn、原 allocator 设置基线的 RNG、step、token 数相同；264/300 adapter tensor 与300/900 optimizer tensor超出原容差，最大绝对差分别为 $2.8776\times10^{-5}$ 和 $7.1256\times10^{-5}$。原门槛保留；这不是单独隔离数值漂移机制的实验。memcheck没有最终checkpoint，无法与本轮基线比较最终状态。

父任务退出0、worker完成且资产验证通过，说明编排器完成了收集；子测试总判定仍为 `all_cases_passed=false`。完整 scheduled-to-stop 区间为1,304秒、0.362222 H20·h，包含准备、失败和全部发布，单独计入成本。

40个子测试索引文件的大小/SHA256全部核验。50个terminal源文件的导出与运行中48文件快照分开保留：47个已有文件字节不变，`task-events.json`补入第二项任务结束事件，新增worker/pair完成记录。原始binary、checkpoint和失败日志保留，公开脱敏文件有各自指纹。

下一项冻结配对诊断把rollout batch从8改为1，保留所有20条trajectory、完整模型及两次更新。分批采样可能改变轨迹，故只检验新条件下的memcheck行为；提交及167项CPU检查不构成GPU或质量验收。

证据入口：

- [terminal完成记录与源文件指纹](sanitizer-allocator-probe-v1-terminal/PUBLIC_EXPORT_MANIFEST.json)
- [独立复核](SANITIZER_ALLOCATOR_V1_REVALIDATION.json)与[结构化结果](SANITIZER_ALLOCATOR_V1_SUMMARY.json)
- [保留原门槛的逐tensor比较](SANITIZER_ALLOCATOR_V1_BASELINE_NUMERICS.json)
- [完整分配成本](SANITIZER_ALLOCATOR_V1_COST.json)
- [下一项诊断的冻结源码与输入指纹](../../experiments/sanitizer-rollout1-probe-v1/manifest.json)
