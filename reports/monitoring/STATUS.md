# 专业监控验证状态

2026-10-28。

统一监控配置和运行器已实现，当前 CPU 回归通过 140 项。首次 1-H20 套件已结束：实际环境为 NVIDIA H20 / PyTorch 2.9.1 / CUDA 12.9 / driver 535.261.03。none、Nsight Systems 和 PyTorch profiler 均退出 0，冷恢复后的 update162 运行门通过；实际 CUDA/API/NVTX 和 PyTorch GPU kernel trace 已核查。

NCU 2026.1.1 以 application code 11 报告驱动不兼容；套件最终失败、Pod 退出 1，详见 `NCU_COMPATIBILITY_FAILURE.json`。随后的 2025.2.1、2024.1.1、2023.3.1 都在 Teacher SDPA 的 cuDNN 初始化处失败，未捕获 kernel；45 个索引文件哈希全部核验，见 `NCU_COMPATIBILITY_V1_SUMMARY.json` 和 `ncu-compatibility-v1/`。

新单卡诊断显式比较 default/no_cudnn，并运行 NCU 基础指标、指定 GEMM 的详细 section、nsys 和可用系统工具。该诊断修改 attention 可选 backend，不能与默认训练时间混算；质量实验未修改。尚无新有效 NCU counter。

91 个输出文件与原始 `outputs.json` 的大小/SHA256 一致；公开导出见 `profile-professional-v1/PUBLIC_EXPORT_MANIFEST.json`。nsys/PyTorch 与 none 的 RNG、step 和 token 数一致，300 个 adapter / 900 个 optimizer tensor 通过预设数值容差；精确哈希不一致，见两份 `PROFILE_NUMERICS_*.json`。这不证明生成质量等价。

首次套件的 terminal Pod 分配区间与成本见 `PROFILE_PROFESSIONAL_V1_COST.json`，包括准备、失败工具和导出；兼容性任务单独消耗 1,376 秒、0.38222 H20·h，见 `NCU_COMPATIBILITY_V1_COST.json`。后续 recovery 成本仍待闭合。任何 profiler 的墙钟都不进入速度优势表。

DCGM、perf、Compute Sanitizer 配置可通过文档中的入口运行；本轮未验证它们的硬件/权限环境。

历史有效 nsys/PyTorch 证据在 `reports/profiling` 和 `reports/expanded/profiles`。新增采集任务不自动修改历史性能结论。
