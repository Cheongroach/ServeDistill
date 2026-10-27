# 专业监控验证状态

2026-10-27。

统一监控配置和运行器已实现，独立仓库 CPU 回归通过 132 项。已提交一个 1-H20 任务，分别冷恢复执行 none、NCU、Nsight Systems 和 PyTorch profiler；真实 GPU 环境已报告 NVIDIA H20 / PyTorch 2.9.1 / CUDA 12.9，任务仍在准备资产。

基线和 nsys 副本已成功退出并通过 update162 的 runtime gate。NCU 2026.1.1 在实际 driver 535.261.03 上以 application code 11 失败，日志报告 CUDA driver 不兼容；详见 `NCU_COMPATIBILITY_FAILURE.json`。版本与 query-metrics 成功不替代 capture。下一步测试兼容的官方旧版，真实 counter、状态对比和终态成本仍未闭合。

DCGM、perf、Compute Sanitizer 配置可通过文档中的入口运行；本轮未验证它们的硬件/权限环境。

历史有效 nsys/PyTorch 证据在 `reports/profiling` 和 `reports/expanded/profiles`。新增采集任务不自动修改历史性能结论。
