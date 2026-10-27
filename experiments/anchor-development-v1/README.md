# 固定 gold CE anchor 的 DEV 消融

问题：在同一真实训练起点和同一 Teacher320 下，固定 gold CE 项能否减少 AOPD 的类别/值域漂移，并对两个强 SFT 得到稳定 DEV 改善？

`configs/anchor-development-v1.json` 在提交前冻结。候选是 `L_AOPD + 0.5 L_gold_CE`，控制包含纯 AOPD、offline SFT、oracle SFT，以及不继续训练的共同初始 adapter。四个种子为 101/211/307/401，各自 320 updates × 10 prompts。每个方法重新初始化 Adam；全部方法绑定同一实际 adapter 与同一提示 schedule。offline 标签来自与 AOPD 评分完全相同的 Teacher320，覆盖全部训练 ID，不按正确率过滤。

每个种子独占 1 H20，四个方法顺序执行，本轮训练并发为四卡。连同另一个一卡监控验证任务，当前最多五卡，低于用户八卡上限。队列与资源私有标识不进入公开仓库。

训练内 DEV 每域 40 条用于观察；主 DEV 结果固定最终第 320 步 adapter，以 BF16 评估 search/relevance/substitute 各 300 条和 math/instruction 各前 100 条，共 1,100 个输入。不会挑选 test checkpoint，也不会把原失败确认的数据重新称为 unseen。

只有候选全部四 seed 通过绝对质量门、对两个 SFT 的均值业务差为正、各至少三对 seed 为正，并满足受保护域的非劣规则，才考虑重新冻结独立确认。DEV 门通过仍不证明质量优势。

原 Teacher320 标注的约 2,933.9 秒执行、既有初始化、Teacher 修复及全部新 Pod 分配均需单独入账。此轮同 updates/prompts，不保证同 response tokens/GPU 时间；带人工标注监督的候选不能冒充纯 OPD。

精确执行源代码保存在 `source.tar.gz`。它包含冻结协议要求的全部 package 源文件和 driver，不含模型、数据、凭据或平台请求：

```bash
python scripts/materialize_experiment.py --output ../ServeDistill-anchor-v1
```

在生成目录安装项目依赖并准备协议绑定的输入：

```bash
python scripts/run_anchor_development.py \
  --student "$STUDENT" --teacher "$TEACHER" --data "$DATA" \
  --initial "$INITIAL_ADAPTER" --labels "$TEACHER_LABELS" \
  --protocol anchor-protocol.json --seed 101 \
  --output results/seed101
```

`DATA` 必须是协议绑定的 train/dev 文件；labels/initial 必须匹配真实内容 SHA256。运行器拒绝修改后的输入和源码。协议哈希与归档哈希在 `manifest.json` 中；后续监控代码更新不改变已提交训练源版本。
