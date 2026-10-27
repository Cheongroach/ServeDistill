# 专业监控验证状态

2026-10-28。

统一监控配置和运行器已实现，当前 CPU 回归通过 156 项。首次 1-H20 套件已结束：实际环境为 NVIDIA H20 / PyTorch 2.9.1 / CUDA 12.9 / driver 535.261.03。none、Nsight Systems 和 PyTorch profiler 均退出 0，冷恢复后的 update162 运行门通过；实际 CUDA/API/NVTX 和 PyTorch GPU kernel trace 已核查。

NCU 2026.1.1 以 application code 11 报告驱动不兼容；套件最终失败、Pod 退出 1，详见 `NCU_COMPATIBILITY_FAILURE.json`。随后的 2025.2.1、2024.1.1、2023.3.1 都在 Teacher SDPA 的 cuDNN 初始化处失败，未捕获 kernel；45 个索引文件哈希全部核验，见 `NCU_COMPATIBILITY_V1_SUMMARY.json` 和 `ncu-compatibility-v1/`。

后续单卡诊断已结束，显式 no_cudnn 的 NCU 2025.2.1 实际捕获 8 个基础 kernel 和 1 个指定 GEMM。旧验证器误把宽 CSV 当成长 CSV，产生两项 false verdict；原记录完整保留。当前独立复核通过对应 runtime/counter 门，156 个原始索引文件全部核验，见 `PROFILE_RECOVERY_V1_REVALIDATION.json`；公开文本及 188 个源文件的指纹见 `profiler-recovery-v1/PUBLIC_EXPORT_MANIFEST.json`。该诊断改变 eligible SDPA backend，不能与默认训练时间混算。

NCU 基础采集、GEMM 和 nsys 的 RNG、step、token 数与同 dispatch 基线一致，300 个 adapter / 900 个 optimizer tensor 全部通过原容差；精确最终状态仍不同，见 `PROFILE_RECOVERY_NUMERICS_*.json`。这不证明生成质量等价。已验证环境的工具包锁定在 `tools-h20-r535-no-cudnn.lock.json`。

91 个输出文件与原始 `outputs.json` 的大小/SHA256 一致；公开导出见 `profile-professional-v1/PUBLIC_EXPORT_MANIFEST.json`。nsys/PyTorch 与 none 的 RNG、step 和 token 数一致，300 个 adapter / 900 个 optimizer tensor 通过预设数值容差；精确哈希不一致，见两份 `PROFILE_NUMERICS_*.json`。这不证明生成质量等价。

首次套件的 terminal Pod 分配区间与成本见 `PROFILE_PROFESSIONAL_V1_COST.json`，包括准备、失败工具和导出；兼容性任务单独消耗 1,376 秒、0.38222 H20·h，见 `NCU_COMPATIBILITY_V1_COST.json`。恢复任务消耗 3,727 秒、1.035278 H20·h，见 `PROFILE_RECOVERY_V1_COST.json`。任何 profiler 的墙钟都不进入速度优势表。

独立 system-watch 的 host/GPU 遥测完成；DCGM、perf 未安装。Compute Sanitizer 在 student rollout prefill 记录 cuMemCreate 的 CUDA API 错误，并在 DeltaNet 路径 OOM，memcheck 汇总 3 errors。没有完整 update162，也没有通过 sanitizer；原因仍待隔离。见 `PROFILE_RECOVERY_V1_SUMMARY.json` 和公开原始日志。

另一个 1-H20 allocator 诊断已提交，冻结源码在 `experiments/sanitizer-allocator-probe-v1/`。它关闭 expandable segments，对同一完整 workload 比较无 profiler 与 memcheck；模型、输入、rollout 和训练窗口不变。提交和 CPU 检查不证明其运行成功，结果及分配成本仍待收集。

历史有效 nsys/PyTorch 证据在 `reports/profiling` 和 `reports/expanded/profiles`。新增采集任务不自动修改历史性能结论。
