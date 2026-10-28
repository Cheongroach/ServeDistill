# 一步步检验分领域 CE 锚点

本轮只使用已观察的 train/DEV。两组候选已冻结在 [协议](../configs/domain-anchor-development-v1.json)：`domain01` 的业务权重为 0.1，`domain05` 为 0.5；两者的 math/instruction 权重均为 1.0。它们不是已确认胜出的模型。

## 第 1 步：把上一轮失败写成可检验假设

[三种标量权重的终态](../reports/advantage/ANCHOR_JOINT_DEV.md)显示：0.1/0.5 的业务均值较好，但数学非劣门失败；1.0 的数学保留较好，业务门失败。新假设是不同领域需要不同锚点强度。不能仅凭这组趋势推断分领域权重有效。

自检：为让 0.5 晋级，把数学 margin 从 −5 pp 放宽到 −6 pp 是否合理？不合理；本轮保持原门槛。

## 第 2 步：先明确损失的分母

本轮采用 trajectory 聚合。设一个完整更新中有 $N$ 条生成轨迹，gold 回答含 $T_i$ 个响应 token，包含 EOS：

$$
L = L_{\mathrm{AOPD}} +
\frac{1}{N}\sum_{i=1}^{N}w_{d_i}
\frac{1}{T_i}\sum_{t=1}^{T_i}-\log p_\theta(y_{it}\mid x_i,y_{i,<t}).
$$

$w_{d_i}$ 由该行 domain 决定。分母仍是全部轨迹数，不能改成权重之和或某个微批的领域数；否则会改变 CE 对 AOPD 的相对强度。token 聚合的实现则使用全部 gold token 数作分母，不使用生成长度。

入口为 `train.apply_rollout → supervision.gold_ce`。先算未加权 AOPD 的完整更新份额，再独立算 gold CE 的份额。微批只划分计算与显存，不改变完整梯度。`anchor_ce_domain_weights` 未配置时保留原标量行为；缺少某个领域时用标量 fallback，本轮显式覆盖全部五个领域且 fallback 为零。

自检：某条 gold 回答较长，trajectory 模式下是否应自动占更大权重？不应；先取该条回答的 token 均值。

## 第 3 步：保持可归因的配对条件

四个种子为 101/211/307/401。每个候选单独训练，fresh Adam；所有方法共用原初始化 adapter、Student 与 Teacher320。320 次更新，每步 10 个 prompt、group2、rollout batch8、learner microbatch4、max128 tokens，FP32 学生训练；学习率与领域日程保持原样。最终只评估固定 update320 的 BF16 adapter。

旧对照来自原始终态，收集器重新校验全部文件、adapter、320 次实际更新、prompt 日程和成本；不靠报告中的分数直接复用。旧配置缺少新增空 map 字段时，通过 dataclass 默认值规范化比较，原始配置文件保持原样。新协议绑定更新后的完整源码，不冒充旧源码指纹。

自检：相同 seed 是否足以证明训练配对？不够；还要核对初始化、优化器、学习率、实际 prompt 日程与更新次数。

## 第 4 步：运行并收齐终态

```bash
python -m scripts.run_domain_anchor_development \
  --student "$STUDENT" --teacher "$TEACHER320" \
  --data "$ORIGINAL_DATA" --initial "$COMMON_INITIAL" \
  --base-protocol configs/anchor-development-v1.json \
  --protocol configs/domain-anchor-development-v1.json \
  --variant domain01 --seed 101 --output "$NEW_SEED_OUTPUT"
```

对两组候选和四个种子分别执行。Worker 使用带 SHA256 的自包含 overlay、模型 manifest 和依赖资产，不能依赖开发机路径。每个任务保留完整训练日志、生成轨迹、指标、最终 adapter、DEV 输出和终态平台响应。CPU 验证见 [本轮记录](../reports/monitoring/CPU_VALIDATION_DOMAIN_ANCHOR.json)，不代表 GPU 质量验收。

```bash
python -m scripts.collect_domain_anchor_development \
  --base-protocol configs/anchor-development-v1.json \
  --protocol configs/domain-anchor-development-v1.json \
  --submissions "$ALL_EIGHT_SUBMISSIONS" --training "$ORIGINAL_DATA/train.jsonl" \
  --hdfs-root "$EVIDENCE_ROOT" --output "$NEW_RAW_EVIDENCE"
```

成本按每个 Pod 的 schedule-to-stop 计算；失败启动、重启、准备和导出都保留。新任务成本与历史对照成本分别列出，后者不能再次计入本轮新增消耗。

自检：任务父进程退出 0，能否跳过 child completion 和 adapter 检查？不能。

## 第 5 步：统一重评两个候选后才选择

```bash
python -m scripts.analyze_domain_anchor_development \
  --base-root "$RAW_ORIGINAL_CONTROLS" --root "$NEW_RAW_EVIDENCE" \
  --base-protocol configs/anchor-development-v1.json \
  --protocol configs/domain-anchor-development-v1.json \
  --training "$ORIGINAL_DATA/train.jsonl" --dev "$ORIGINAL_DATA/dev.jsonl" \
  --output "$DOMAIN_DEV_ANALYSIS"
```

分析器重评 22,000 条预测：8 个新候选输出加 12 个 initial/offline/oracle 输出，每份 1,100 条。所有四种子须通过绝对业务质量、类别 recall 与 JSON 门；相对两个 SFT 的业务均值须为正，至少三对种子为正；相对两个 SFT 和 initial 的各业务域及数学均须通过原非劣门。仅在两个候选全部收齐后，按合格候选业务均值排序；精确平局选择较小业务 CE 权重。

DEV 合格只允许继续设计新的确认。现有 `freeze_fresh_confirmation` 入口绑定上一轮标量权重选择器，不能直接用它封存本轮候选；若本轮晋级，必须接入本轮完整重评分与新 adapter 清单，沿用前瞻 alpha ledger，再在未观察输入上完成确认和复制两批。禁止靠手填 `advance=true` 绕过此步骤。

自检：只跑完最有希望的一组，能否先用新确认题选它？不能。
