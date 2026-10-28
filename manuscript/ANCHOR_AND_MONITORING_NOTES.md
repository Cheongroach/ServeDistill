# 监控完整性与固定 gold anchor：研究手稿

当前需要回答两个独立问题：现有执行速度解释是否有足够系统和框架证据；修订训练目标是否能在公平强对照下改善业务质量。二者分别验收，不能用 profiler trace 代替质量结果。

## 1. 从失败结果提出假设

旧冻结域路由相对 offline SFT 的主测试差是 −0.147 pp，95% CI [−0.592,+0.307]；独立复验为 −0.202 pp，[−0.936,+0.577]。seed401 同时未满足 99% schema 门。这个确认失败不因后处理开发改善而被覆盖。

随后在同一 DEV 上比较 fixed320、final640、last2、all4。last2 排名第一，但对 offline 的业务差仅 +0.122 pp，2/4 seed 为正；对 final640 仅 +0.007 pp。平均规则降低了这批 DEV 的值域异常，但未形成稳定质量优势。

历史 FP32 DEV 的六项错误是可解析 JSON 使用 C/I 而非 yes/no，未截断。它支持调查类别/值域漂移，不建立损失、数值精度或解码的因果机制。

## 2. 固定一个新的训练目标

候选为 `L_AOPD + 0.5 L_gold_CE`。同一训练提示同时获得 Student on-policy 轨迹的 Teacher 反馈，以及真实标注答案的 CE 监督。

gold 与 sampled response 使用各自的回答 mask/分母。trajectory 模式先对各回答求 token 均值；microbatch 累积按序列数缩放。token 模式使用 gold 有效动作数的比例，避免 microbatch 切分改变完整 batch 的目标。CPU 测试比较实际完整 optimizer 更新，覆盖两种归约和不同切分；该测试不是 H20 结果。

它可能减少类别/取值漂移，也可能主要依靠 SFT 标签获益。所以必须保留纯 AOPD、同一 Teacher 的 offline SFT 和 oracle SFT，并将候选称为混合监督 AOPD。

## 3. 把对照真正放在同一起点

本轮绑定同一个实际 Student adapter，所有方法使用 fresh Adam、同 seed 的提示 schedule、320 updates × 10 prompts。四个种子是 101/211/307/401，主 DEV 结果固定最终 update，不按 seed 选 checkpoint。

旧路由配方使用了不同阶段 Teacher 的监督。本轮 offline 标签来自与 AOPD 完全相同的 Teacher320，完整 6,400 个训练 ID、6,160 个唯一 Teacher prompt，不按正确率过滤。其原标注执行约 2,933.9 秒，必须计入共享成本。

same updates/prompts 不意味着 same tokens/time。oracle/offline 和 AOPD 的具体工作不同；不能仅根据分数和更新次数宣称同成本优势。

## 4. 保持开发和确认边界

新 worker 只打包 train/dev、初始 adapter 和训练标签。主 DEV 使用 1,100 个独立输入；训练内小 DEV 和最终完整 BF16 DEV 分开保存。旧 test/replication 已观察，不能成为修订候选的未见确认集。

候选需要全部四 seed 通过绝对指标/schema/类别 recall 门，并对两个 SFT 各有正均值和至少三对正 seed，以及受保护域非劣，才考虑新确认。新确认还需要未见的 query、新的数学回归来源或明确的未见子集，以及预先声明的连续检验规则。既有 SVAMP 全部原题已用完，不能重新取相同题目声称独立复验。

## 5. 监控证据按问题选择工具

无 profiler 的固定轨迹重复计时判断实际速度；nsys 定位阶段与 launch/同步间隙；PyTorch profiler 关联 operator/shape；NCU 检查代表 kernel 的 SM/DRAM/occupancy/launch counters。低频 nvidia-smi 和可选 DCGM 记录设备状态，`/proc` 和 perf 补充 host 证据，Compute Sanitizer 单独核查执行访问。

采集从 checkpoint-0160 冷恢复，warmup update161，捕获 update162。各工具独立进程，保存源文件/data/state 指纹与输出哈希。NCU replay 的时间不进入无 profiler 的速度主表。

新增 NCU 仍需真实 report/CSV、counter 支持与权限、目标 NVTX range 和实际 kernel 核验。锁定官方包、`--version` 成功、任务 running，都不等于采集通过。

## 6. 当前可报告的事实

完整代码与文档已发布到独立 main；doc 保留原 manuscript 历史并加入本轮记录。最新 main `015b10c` 已包含采集内容检查、数值状态对比、串行 NCU 兼容性入口、冻结采集源码和公开证据。CPU 136 项通过。

首次 profiler 套件已终态失败：none/nsys/PyTorch 应用退出 0，冷恢复与 update162 运行门通过；NCU 2026.1.1 在 driver 535.261.03 上报告不兼容，应用退出 11，套件 Pod 退出 1。实际 nsys CUDA/API/NVTX 和 PyTorch GPU kernel 事件已核查；91 个原始输出文件的 bytes/SHA256 全部一致。

全部 300 个 adapter 和 900 个 optimizer tensor 均完成与 none 的逐 tensor 比较。nsys/PyTorch 的 adapter 最大绝对差为 `3.73e-9` / `1.86e-9`，optimizer 为 `4.75e-9` / `5.76e-9`；RNG、update、token 数一致。预设数值容差通过，但精确哈希不同，不能宣称相同生成文本或质量。

首次套件的 schedule-to-stop 为 1,898 秒、1 GPU，即 `0.52722 H20·h`；container start-to-stop 为 1,895 秒。它包含资产准备、失败工具、导出和容器关闭；这是观测分配区间代理，不是财务账单。新 NCU 兼容任务另计。

兼容任务按官方锁定的 2025.2.1 → 2024.1.1 → 2023.3.1 串行执行真实训练，首个完整 runtime/counter 通过者才入选。它已提交，尚无有效新 counter。四个 DEV seed 仍在执行，seed101 已从 1,100 条初始 DEV 进入 anchor 训练；最终四方法模型尚未收齐。质量、时间、成本和规模结论分别等待对应证据，稳定质量优势仍未建立。

## 7. 2026-10-28 的扩展与新证据

用户取消全局 8 卡限制后，新增权重 0.1、1.0 各四个种子，与已有 0.5 候选和两个强 SFT 比较。只改变 gold CE 系数，保留原训练代码、配对 prompt 日程及最终 update320。全部候选/对照完成后，在通过原门的候选中选择平均 DEV business macro 最大者，同分选较小权重。原四方法仍在运行；不能根据中间 step 的有利分数提前挑选。

新增八个单卡任务与原四个单卡任务均进入运行态。另一个单卡 profiler recovery 独立运行。这一资源扩展改善并行进度，不构成质量证据，也不会把新增 gold CE 计算从成本里抹去。

三版旧 NCU 已全部失败，实际异常位置是 warmup 的 Teacher SDPA：`CUDNN_STATUS_NOT_INITIALIZED`。45 个索引产物通过 SHA256 校验，Pod 调度到结束为 1,376 秒，即 0.38222 H20·h。该异常尚未定位到具体库/驱动原因。新的 recovery 对照默认 dispatch 与显式禁用 cuDNN SDPA 的诊断副本，并对历史热点 FP32 GEMM 家族采集详细 section；结果仍待收集。

输入审计显示：small-version ESCI 只剩 1,273 个未用 test query，GSM8K test 只剩 119 题，SVAMP 为零。改查完整源后有 17,626 个未用 ESCI test query、4,573 道未用 GSM8K train 原题。新确认必须声明抽样总体变化、按原题/查询组排除历史输入，并在任何新模型预测前冻结数据和显著性预算。公开基准是否进入基础模型预训练仍未知。

main `b361316` 已发布以上协议、源码快照、收集器、数据审计、失败成本与文档，当前 CPU 回归 140 项通过。GPU 终态、各权重最终质量和新计数器仍分别等待实证。

## 8. 恢复采集与统一 DEV 验收的核验

main `198271e` 补齐三档权重的统一收集和选择入口，待全部训练结束后重评 30,800 条固定最终 DEV 输出；完整评分尚未执行。实际 rollout ID、版本、学习率、完整更新数、参数变化、token 状态和最终 adapter 都是审计检查项。所有权重、种子和两个强 SFT 齐备后，才按原门槛和同分规则选择；没有合格者则返回 null。CPU 回归 156 项通过，GPU 质量仍待终态产物。

恢复采集使用 NCU 2025.2.1、显式 no_cudnn，实际获得八个基础 kernel 及一个指定 FP32 GEMM。旧解析器只识别长 CSV，误报零 kernel；当前支持真实宽表和单位行，核验了六项捕获的全部 156 个索引文件。原始 false verdict 不覆盖，复核结果另存 `PROFILE_RECOVERY_V1_REVALIDATION.json`。

指定 GEMM 为 1.473088 ms、SM throughput 88.902048%、DRAM throughput 1.975737%、achieved occupancy 12.492236%，寄存器为 255 register/thread，寄存器限制为 2 block，L2 sector hit rate 为 92.505896%。这些是一个过滤 launch 的计数器，不是全训练平均或 BF16 Tensor Core 利用率，也不能证明优化收益。

NCU 基础采集、GEMM、nsys 的 300 个 adapter 与 900 个 optimizer tensor 均通过同 dispatch 基线的预设容差；RNG、step、token 数一致，精确哈希不同。这个数值检查不证明文本或业务质量等价。恢复任务消耗 3,727 秒、1.035278 H20·h，包含准备和失败工具。

独立 host/GPU telemetry 通过，perf/DCGM 未安装。Sanitizer 日志显示 student rollout prefill 的 DeltaNet 路径 OOM，同时记录 ExpandableSegment::map 中 cuMemCreate 的 CUDA_ERROR_INVALID_VALUE，汇总 3 errors。新的 1-H20 对照保持完整模型、输入、rollout 和训练窗口，只关闭 expandable segments，并对比基线与 memcheck。已提交不等于运行成功；原失败、allocator 对照结果及成本分别保留。

稳定质量优势仍未建立。接下来仍须全部 DEV 最终模型、统一独立评分，以及合格候选在新的未观测数据上的预先冻结确认和不重叠复验。

## 9. 标量权重没有晋级：保留失败，再检验分领域监督

三种权重的四种子固定 update320 已全部终结，30,800 条 DEV 预测独立重评，428 个文件及 24 份训练后 adapter 全部绑定。0.1/0.5 的业务均值相对 offline 为 +0.3981/+0.2608 pp，均有 3/4 正种子配对；数学相对 oracle 的查询 bootstrap 95% 下界为 −8.00/−5.25 pp，均未通过 −5 pp 非劣门。1.0 数学下界为 −2.00 pp，但业务相对 offline 为 −0.2545 pp，仅 2/4 正种子配对，并有 substitute 非劣失败。没有 DEV 候选晋级，更没有新确认优势。

原四任务与八个权重任务分配成本合计 56.668611 H20·h，历史对照/Teacher/init 成本分别保留。rollout batch1 memcheck 独立诊断为超时退出124，没有完成 update161/162或形成 error summary；基线完成162。全部41个索引文件核验，51个terminal源文件导出；1.313889 H20·h 全部保留。未知超时原因不能写成 OOM 或零错误。

下一轮只在 DEV 检验业务 CE 0.1/0.5、math/instruction CE1.0。损失对每条 gold 回答先取 token 均值再乘领域系数，最后除以完整轨迹数；禁止按权重和或微批领域数归一化。新实现通过实际梯度、uniform map 对标量等价、两种聚合的微批不变性和候选执行绑定检查，全套176项 CPU 回归通过。

新源码单独封存，旧协议原始字节保留；复用强对照前重新核验全部16段更新历史和实际prompt日程。8个单卡任务通过预检后提交并启动，原门槛保持不变。终态监控将收集两组候选四种子并重评22,000条候选/强对照/initial输出。合格DEV仍须新的确认与不重叠复制，当前没有质量优势结论。
