# Anchor 权重扩展：冻结的执行源码

原 `anchor-development-v1` 保持不变。此扩展仅把 `anchor_ce_weight` 改为 0.1 或 1.0，各用 101、211、307、401 四个种子；0.5、pure AOPD 和两个 SFT 对照来自原轮。用户已取消全局 8 卡上限，历史协议中的 `max_live_gpus: 8` 保留其原始字节。

## 第 1 步：还原执行源码

```bash
python -m scripts.materialize_experiment \
  --experiment anchor-weight-development-v2 \
  --output ../ServeDistill-anchor-weight-v2
```

入口会核对 source、support 和两个协议的 SHA256。`source.tar.gz` 保存原训练代码及新 runner，避免后续监控代码修改导致原协议校验失败。

## 第 2 步：绑定原始输入

沿用原轮已验证的 Student、Teacher、`anchor-data` 和 `anchor-initial`。不要用下载后未经 manifest 校验的其他版本替换。train/dev、adapter 和 adapter config 的哈希在原协议；新协议绑定原协议哈希。

## 第 3 步：运行单个配对实验

在新目录安装项目后运行：

```bash
python -m scripts.run_anchor_weight_development \
  --student "$STUDENT" --teacher "$TEACHER" \
  --data "$ANCHOR_DATA" --initial "$ANCHOR_INITIAL" \
  --protocol anchor-protocol.json --extension anchor-weight-protocol.json \
  --variant anchor01 --seed 101 --output results/anchor01-seed101
```

`anchor10` 表示权重 1.0。每次使用新输出目录；输出包含实际 schedule、训练日志、最终 checkpoint、完整 1,100 条 BF16 DEV 预测、adapter 指纹和 completion。`--destination` 可选，用于持续发布到已有挂载目录。

## 第 4 步：等待全部比较项完成

三个权重、四个种子和两个 SFT 对照全部完成后，才应用新协议中的 DEV 选择规则。每个候选必须通过原来的绝对质量门、相对两个 SFT 的平均业务改善和至少 3 个正向种子配对，以及相对两个 SFT/initial 的各域与数学非劣效门。在合格候选中按平均 DEV business macro 排序，相同分数选较小权重。

这里的合格只允许冻结一次新的独立确认。所有失败、未合格候选及完整 GPU 分配成本都保留。
