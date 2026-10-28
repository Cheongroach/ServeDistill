# 分领域 CE：冻结的执行与分析源码

两组候选的 train/DEV、初始化、Teacher320、种子、更新预算和门槛与原轮配对，业务 CE 为 0.1/0.5、math/instruction CE 为 1.0。新协议绑定更新后的源码；旧协议及失败证据保持原样。训练中没有打包新的确认输入。

```bash
python -m scripts.materialize_experiment \
  --experiment domain-anchor-development-v1 \
  --output ../ServeDistill-domain-anchor-v1
```

还原器校验两个 archive 和协议指纹。新源码清单完整覆盖 domain runner、收集器、分析器及其训练/评分依赖；它不会用旧协议的源码指纹覆盖更新后的训练实现。`anchor-protocol.json` 保存原输入/对照协议，`domain-anchor-protocol.json` 保存本轮协议。

按原协议绑定 Student、Teacher320、train/dev、初始化 adapter，并在还原目录运行：

```bash
python -m scripts.run_domain_anchor_development \
  --student "$STUDENT" --teacher "$TEACHER320" \
  --data "$ORIGINAL_DATA" --initial "$COMMON_INITIAL" \
  --base-protocol anchor-protocol.json --protocol domain-anchor-protocol.json \
  --variant domain01 --seed 101 --output "$NEW_OUTPUT"
```

每个候选的四个种子都完成后，才能进行完整重评和 DEV 选择。[分步文档](../../docs/DOMAIN_ANCHOR-friendly.md)说明 gold CE 分母、实际更新绑定、历史对照复核、成本和新的确认边界。archive 不含数据、模型权重、平台凭据或新 holdout。
