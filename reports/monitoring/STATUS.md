# 专业监控验证状态

2026-10-27。

统一监控配置和运行器已实现，独立仓库 CPU 回归通过 132 项。已提交一个 1-H20 任务，分别冷恢复执行 none、NCU、Nsight Systems 和 PyTorch profiler；真实 GPU 环境已报告 NVIDIA H20 / PyTorch 2.9.1 / CUDA 12.9，任务仍在准备资产。

NCU 的官方安装包与 SHA256 已核对，但尚未收齐真实 counter、有效 capture、完整状态对比和终态成本，所以本轮 NCU 验证状态仍为待完成。

DCGM、perf、Compute Sanitizer 配置可通过文档中的入口运行；本轮未验证它们的硬件/权限环境。

历史有效 nsys/PyTorch 证据在 `reports/profiling` 和 `reports/expanded/profiles`。新增采集任务不自动修改历史性能结论。
