# 研究手稿与对话记录

按以下顺序阅读：

1. [54 步项目教程](../docs/一步步认识ServeDistill项目-friendly.md)：从业务问题到训练与证据。
2. [本轮对话摘录](CONVERSATION_RECORD_20261027.md)：可核验请求与已执行答复。
3. [监控与 anchor 方法手稿](ANCHOR_AND_MONITORING_NOTES.md)：假设、对照、冻结规则及验收边界。
4. [独立性能监控文档](../PERFORMANCE_MONITORING.md)：每个工具的输入、参数、输出和选择原因。
5. 历史 `DESIGN_NOTES.md`、`PROFILING_NOTES.md`、`EXPANDED_STUDY_NOTES.md`、`FINAL_REFLECTION.md`：保留旧研究判断，不能以新结果覆盖旧失败。
6. [解除卡数限制后的记录](CONVERSATION_RECORD_20261028.md)：新增配对权重、未观测输入审计与 cuDNN 诊断。

`main` 和 `doc` 可以同时检出。默认阅读 workspace 将目录 `ServeDistill` 与 `ServeDistill-doc` 放在同一父目录。本分支保留当前 main 的运行代码，并在 `manuscript/` 增加研究材料，便于从注释与决策跳回实现。

研究记录的可见对话片段不等同于完整聊天备份；未在当前记录中恢复的逐字内容不会被补写。平台资源标识和无关对话不进入公开手稿。
