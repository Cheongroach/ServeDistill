# 专业监控验证状态

2026-10-28。

统一监控配置和运行器已实现，当前 CPU 回归通过 167 项，见 `CPU_VALIDATION_ROLLOUT1.json`；此前 156/166 项验证按各自源码版本保留。首次 1-H20 套件已结束：实际环境为 NVIDIA H20 / PyTorch 2.9.1 / CUDA 12.9 / driver 535.261.03。none、Nsight Systems 和 PyTorch profiler 均退出 0，冷恢复后的 update162 运行门通过；实际 CUDA/API/NVTX 和 PyTorch GPU kernel trace 已核查。

NCU 2026.1.1 以 application code 11 报告驱动不兼容；套件最终失败、Pod 退出 1，详见 `NCU_COMPATIBILITY_FAILURE.json`。随后的 2025.2.1、2024.1.1、2023.3.1 都在 Teacher SDPA 的 cuDNN 初始化处失败，未捕获 kernel；45 个索引文件哈希全部核验，见 `NCU_COMPATIBILITY_V1_SUMMARY.json` 和 `ncu-compatibility-v1/`。

后续单卡诊断已结束，显式 no_cudnn 的 NCU 2025.2.1 实际捕获 8 个基础 kernel 和 1 个指定 GEMM。旧验证器误把宽 CSV 当成长 CSV，产生两项 false verdict；原记录完整保留。当前独立复核通过对应 runtime/counter 门，156 个原始索引文件全部核验，见 `PROFILE_RECOVERY_V1_REVALIDATION.json`；公开文本及 188 个源文件的指纹见 `profiler-recovery-v1/PUBLIC_EXPORT_MANIFEST.json`。该诊断改变 eligible SDPA backend，不能与默认训练时间混算。

NCU 基础采集、GEMM 和 nsys 的 RNG、step、token 数与同 dispatch 基线一致，300 个 adapter / 900 个 optimizer tensor 全部通过原容差；精确最终状态仍不同，见 `PROFILE_RECOVERY_NUMERICS_*.json`。这不证明生成质量等价。已验证环境的工具包锁定在 `tools-h20-r535-no-cudnn.lock.json`。

91 个输出文件与原始 `outputs.json` 的大小/SHA256 一致；公开导出见 `profile-professional-v1/PUBLIC_EXPORT_MANIFEST.json`。nsys/PyTorch 与 none 的 RNG、step 和 token 数一致，300 个 adapter / 900 个 optimizer tensor 通过预设数值容差；精确哈希不一致，见两份 `PROFILE_NUMERICS_*.json`。这不证明生成质量等价。

首次套件的 terminal Pod 分配区间与成本见 `PROFILE_PROFESSIONAL_V1_COST.json`，包括准备、失败工具和导出；兼容性任务单独消耗 1,376 秒、0.38222 H20·h，见 `NCU_COMPATIBILITY_V1_COST.json`。恢复任务消耗 3,727 秒、1.035278 H20·h，见 `PROFILE_RECOVERY_V1_COST.json`。任何 profiler 的墙钟都不进入速度优势表。

独立 system-watch 的 host/GPU 遥测完成；DCGM、perf 未安装。Compute Sanitizer 在 student rollout prefill 记录 cuMemCreate 的 CUDA API 错误，并在 DeltaNet 路径 OOM，memcheck 汇总 3 errors。没有完整 update162，也没有通过 sanitizer；原因仍待隔离。见 `PROFILE_RECOVERY_V1_SUMMARY.json` 和公开原始日志。

1-H20 allocator 诊断已结束，冻结源码在 `experiments/sanitizer-allocator-probe-v1/`。关闭 expandable segments 后，无 profiler 基线完成 update162；memcheck 在 student prefill 的 MLP/LoRA 路径 OOM，记录 6 次 `cudaMalloc` / `cudaGetLastError` 分配错误，未完成 warmup161。父编排器退出 0，只说明它收齐两个子测试；`all_cases_passed=false`。40 个索引文件全部核验，50 个 terminal 源文件已单独导出，先前运行中快照继续保留。分配 1,304 秒、0.362222 H20·h，包含准备、失败和全部发布。见 `SANITIZER_ALLOCATOR_V1.md` 与相应 JSON。

这轮基线与上一轮 no_cudnn、原 allocator 设置基线的 RNG、step 和 token 数相同，但 264/300 adapter 与 300/900 optimizer tensor 超出原容差；未放宽门槛。它不是通过的 memcheck 数值对照，后者没有最终状态。

新的 `experiments/sanitizer-rollout1-probe-v1/` 配对诊断已提交。它在严格恢复原 checkpoint 后将 rollout batch8 改为1，保留每步10个 prompt、group2、max128 tokens、完整模型及更新161/162；两个副本同用 no_cudnn、learner microbatch1 和非 expandable allocator。观察配置单独记录，采样轨迹可能变化；GPU 结果和最终成本待收集。

历史有效 nsys/PyTorch 证据在 `reports/profiling` 和 `reports/expanded/profiles`。新增采集任务不自动修改历史性能结论。
