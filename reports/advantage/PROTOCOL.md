历史协议及迭代记录（2026-10-25 起）。其中旧并发预算不适用于 2026-10-27 新轮；新轮最多 8 H20，见 anchor-development-v1.json。

# ServeDistill：质量优势研究

2026-10-25 启动。动机是扩展实验暴露的 substitute 常数预测和相关性类别偏置；已有 m4 执行收益保持独立记录。本轮目标是获得通过绝对类别质量门的模型，并验证 OPD 对公平 SFT 对照的质量优势。旧失败结果保持不变。

## 第一阶段：开发集筛选

仅使用既有扩展 train/dev。dev 已用于先前实验和诊断，所有本阶段结果只用于开发选型，不能作为确认性优势证据。

- 数据：复制真实已标注训练输入，对 relevance 四类各 400 行、substitute 两类各 800 行作重加权；search 1,600、math/instruction 各 800，总 6,400 行。重复真实输入不是新增独立样本，保留 source_sample_id、input/query hash 和实际唯一输入数。
- 提示筛选：原 2B/9B，分别评估原提示及只来自训练集的标签示例；完整业务 dev 每域 300 条，math/instruction 各 100 条。
- Teacher 修复：9B LoRA oracle SFT，BF16/FLA、320 updates、10 prompts/update、microbatch=4、lr=1e-4；每 80 步检查每域 40 条 dev。独立人工标签用于 SFT，不写进待预测 query。
- 强 SFT 对照开发：2B 从原始基座开始，采用同一重加权池和 schedule；FP32/PyTorch、320 updates，分别筛选 lr=5e-5 / 1e-4，其余更新和 dev 参数相同。它是开发筛选，不能与 Teacher 不同精度的时长直接比较方法效率。
- 使用相同 LoRA r=16、零 dropout、AdamW、clip=1 和 128 token 上限。保留实际生成长度、分类混淆、各类召回、格式错误与梯度记录。

计划 4 个独立 H20 screening slot，每个最多 3 小时，名义合计 12 GPUh；平台准备、导出和释放另计。通过最终 payload 的平台预检后提交，error 阻止提交，warn/hint 记录。资源当前态空结果不能推断无法调度，实际排队和终态单独记录。

第一阶段追加共同起点开发对照：固定第一轮 `student_sft_lr1e4/checkpoint-0160` adapter，四个 arm 均重置 Adam，以 seed=29、80 updates、每步 10 个 prompt、相同 domain schedule 和 FP32 Student 比较原始 Teacher Full-KL、原始 Teacher token OPD、oracle SFT lr=5e-5 / 1e-4。采样蒸馏每个 prompt 生成两条轨迹，SFT 每个 prompt 使用一条硬标签，不能将同 prompt 预算描述成同 response-token 预算。四卡四小时窗口名义最多 16 GPUh；这是已观察 dev 上的诊断与选型，不进行正式测试。

第二轮共同起点开发对照固定较强的 `student_sft_lr1e4/checkpoint-0320`，使用第一轮训练已选中的 Teacher `checkpoint-0240`。该教师在完整实际评分提示下业务 macro=0.720874、substitute 正类 recall=0.556522，relevance 四类 recall 均不低于 0.591837。四组为修复 Teacher Full-KL、token OPD、离线 Teacher 硬标签 SFT、人工标签 oracle SFT，均重置 Adam、seed=29、80 updates、每步 10 prompts、lr=2e-5、FP32 Student、microbatch=4；蒸馏 group=2，SFT group=1。每 40 步检查每域 40 条 dev，所选 checkpoint 再评完整业务 dev 每域 300 条。第 320 步教师资格评测仍保留，但本轮不随其输出改变已固定的第 240 步教师。四卡四小时窗口名义最多 16 GPUh，仍只用于开发选型。

离线标注覆盖全部 6,400 条训练 ID、6,160 个唯一输入，不按预测是否正确过滤。应同时报告标注全集及实际训练消费 prompt 的交集：全集标注含未消费样本，不能仅凭这部分额外预标注开销主张 OPD 成本优势。该轮同 prompt/update 预算也不等于同 token 或同 GPU 时间预算。

补充开发对照继续固定第 320 步共同 Student、重置 Adam、seed=29 和 80 updates：原始 Teacher Full-KL lr=2e-5、修复 Teacher token OPD lr=5e-5，以及 oracle / 相同修复 Teacher 离线 SFT 各 lr=5e-5、1e-4。四张 H20，两个 SFT slot 先训练 oracle，再复用第二轮已完成且通过哈希验证的全部无过滤离线标签训练 offline SFT，名义最多 16 GPUh。原始 Teacher 第 160 步起点的诊断不能代替此共同第 320 步对照。新任务在原始诊断任务终态后提交；总并发上限 12 张 H20。所有结果仍属于观察过的开发集。


低学习率开发追加（2026-10-25）：同一第 320 步 Student、修复 Teacher 和 seed=29，纯 token OPD / 相同 Teacher 无过滤离线 SFT / oracle SFT 各筛选 lr=5e-6、1e-5。仍为 80 updates、10 prompts/update、FP32 Student；每 20 步评完整业务 dev 各 300 条和实际可用通用 dev，所选 checkpoint 再按既有完整 1,100 条开发规则评分。改变开发 checkpoint 选择不构成独立优势证据。四张 H20，最多四小时，与既有任务总并发不超过 12 张。

序列等权开发追加（2026-10-25）：实际 token OPD 日志显示 search 214/800 prompts 却占 11,990/20,607 响应 token（58.18%）。固定相同第 320 步起点、Teacher、seed=29、80 updates 和 prompt schedule，追加 aggregation=trajectory 的纯 OPD lr=2e-5 / 5e-5，以及相同序列归约的 offline / oracle SFT lr=2e-5。该归约对每条响应先取 token 均值，再对序列取均值；仍是即时 token OPD 信号，不增加人工标签混合损失。offline 使用已完成的全部无过滤 Teacher 标签。四张 H20、最多四小时，在 repaired 任务终态后提交，总并发不超过 12 卡。所有新旧聚合与学习率结果进入开发选型池，不能事后只保留有利尝试。

## 正式对照的设计约束

开发筛选完成后固化候选和主对照，再从官方 ESCI 中冻结未用于旧 train/dev/test/replication 的独立 query。测试自然类别比例保持不变；训练重加权不应用于评测。新正式协议、模型/data/source hash 与验收阈值须在测试揭盲前落盘。

必须保留原 Teacher Full-KL、修复 Teacher 的 OPD、相同 Teacher 轨迹的离线 SFT，以及独立人工标签 SFT。不能把带额外标签的混合损失当成原始纯 OPD，也不能忽略 oracle SFT 这个强竞争者。

至少四对种子，固定共同 Student 起点和 prompt schedule。分别报告同更新/prompt 预算与相同 GPU 时间预算；Student 训练精度统一，Teacher 修复、离线标注、rollout、评分、训练、评测和等待均入账。重复轨迹不能被当成新增独立 query。

优先确认性主比较是预先选定的 OPD 对离线 SFT 的业务三域 macro 差异，query 配对并绑定种子，报告 CI 与种子差异。同时设置 substitute 的正类 recall 和多数类 macro-F1 对照、每域非劣效及有限通用回归。只有模型质量门通过，才讨论有效答案吞吐与成本优势。

开发集尝试可以迭代；正式 test 不能反复用于挑方法、调参或选择最高分。若新版本在正式检验失败，记录该失败，下一轮只能在开发集修改并重新冻结独立 holdout。性能优势、修复优势和 OPD 对 SFT 的方法优势分别判断。

## 独立留出集与后续执行入口

已冻结 `data_advantage_holdout/manifest.json`，seed=202610252。先按 query 排除 `data`、`data_search`、`data_expanded` 和本轮筛选数据所有分片，再从剩余 7,273 个官方 ESCI test query 中选 6,000 个。query 选择和域分配均先于标签检查，候选产品仍按无标签随机规则构造。主测试三域各 1,500 个 query；独立复验三域各 500 个 query；二者不是分布外评测。

数学回归采用官方 SVAMP 的 1,000 条原始题目，各分片 500 条，来源 commit `78e727689e1c1bebfc4be39c446898e8e10b0518`、MIT。SVAMP 按有限 Decimal 数值等价评分，允许 `1` 与 `1.0` 等价；原有 GSM8K 和 instruction 评分保持原规则。测试输出尚未生成或用于选型。

主测试 SHA256：`339073c4a3f0b44465c3ff6da7fc97749f5c9ae84454270b445e9cf6eb2fee85`；独立复验 SHA256：`0e5ea11f936a4f6a3fd40d35d87077edc85048bc62bc48552cb475e06227943d`。独立审计已验证 6,000 个业务 query、1,000 个数学输入与全部历史数据及彼此分片零重叠；未使用类别标签调整测试比例。训练和 dev 逐字继承筛选池，仍属开发证据。

下一阶段入口 `run_advantage_labels.py` 对同一修复 Teacher 生成训练集离线硬标签；重复重加权输入按 Teacher prompt hash 只生成一次，不按正确率过滤。OPD 使用 Student 自己生成的轨迹并由同一 Teacher 评分，因此不能将两者写成“完全相同轨迹”。`run_advantage_arm.py` 固定共同 Student adapter、重置相同 Adam，并分别支持 development / confirmation：正式阶段要求哈希匹配的协议，主结果固定最终 update，而非选最高 test checkpoint。

正式入口同时验证实际模型权重、配置、tokenizer、adapter 配置、每个数据分片及离线标签的训练 ID / prompt 绑定；只验证 manifest 自身不足以证明文件没有被替换。训练与评测源码哈希也进入允许 arm 的合同。离线标注按 256 个唯一 Teacher prompt 分段保存，续跑验证 Teacher、数据、源码与已完成片段哈希；未完成尝试保留，完整平台卡时与失败执行仍须计费。

Teacher 必须使用蒸馏评分时的实际提示进行开发资格验证。第一轮 9B oracle SFT 使用 Student system，原提示筛选和后续 Teacher 评分使用更长的 Teacher system；不能把前一种提示的开发分数直接视为后一种提示已通过。`qualify_advantage_teacher.py` 对相同 adapter、完整业务 dev 每域 300 条和有限通用检查，按实际 Teacher system 验证。离线标注与评分共用 `system_prompt()`，包括没有显式 system 的通用样本；提示默认值和相关源码进入续跑哈希合同。

时间预算辅助入口通过 `method_time_budget_seconds` 与 `prior_method_seconds` 限制已计费的方法时间。前置 Teacher 修复、标注及准备等共享成本必须由 `--cost-manifest` 绑定到已保存执行区间的 SHA256、实际 GPU 数和预先声明的摊销次数。训练计时包含模型初始化、训练内 dev、完整更新和 checkpoint；只在完整更新之间停止，保存实际最终步数、已花时间及超额，冷续跑不能重置计时。到达步数上限不算完成同时间实验。最终独立评测和平台全生命周期卡时另外完整入账；该入口本身不证明总 GPU 成本相等，必须在终态成本审计后判断。

`advantage_statistics.py` 提供四对种子的业务 macro 配对 query CI、独立的探索性 seed/query CI、各域非劣效及每个种子的绝对指标与类别 recall 门。最终候选、强 SFT 主对照、具体阈值和等 GPU 时间预算仍需根据开发筛选固化；当前只有数据已经冻结，尚未把未完成的方法协议称为确认性方案。质量优势不能自动转成成本优势。

`analyze_advantage_confirmation.py` 收齐协议中全部 arm 后，验证每个 seed、实际训练配置、协议/模型/data 指纹、最终完整 update、测试逐行覆盖和评分，再计算冻结的四对 seed 主比较与不相交复验。缺失控制组、评分改动、测试挑 checkpoint 或配置漂移均拒绝分析。原始 Full-KL 若触发预先列明的训练保护错误，允许保留其失败状态并评测最后保存的完整 atomic update；没有成功训练完成文件，不能把此评测写成成功的同预算控制。四对 OPD/离线 SFT 主比较及 oracle 强对照必须全部成功完成，未知失败与独立评测失败仍阻止正式分析。仅终态成本审计通过仍不足以判为同成本优势，成本结论继续独立核验。

## 证据与当前状态

此前的 2.98× learner、1.98× 完整训练收益仅适用于已报告的 microbatch 对照，不作为本轮质量优势。第一轮四个筛选任务均已完成，修复 Teacher 第 240 步实际提示资格评测也已完成；Student 蒸馏与强对照仍在开发阶段，正式方法优势尚未建立。

完整开发集上的较强 2B SFT（lr=1e-4、320 步）业务 macro=0.6299，search NDCG=0.8490、relevance macro-F1=0.5097、substitute macro-F1=0.5310。另一个 lr=5e-5 的 substitute 正类 recall 仅 0.1391，不能用较高 accuracy=0.6533 隐去低召回。二者均属开发选型结果。40 条检查和完整 300 条的差异不是方法优势；相同 40 条上的 FP32 训练/合并 BF16 推理也有少量预测变化，详见 `PRECISION_DEVELOPMENT.json`，尚不能将其全部归因于精度。

执行入口：[advantage_data.py](../../servedistill/advantage_data.py)、[run_advantage_screen.py](../../scripts/run_advantage_screen.py)。全部候选、失败尝试、完整成本、执行合同和重判证据都会保留。

原始 Full-KL 共用第 320 步起点的开发任务已失败：第 80 步所有检查输出均达到 128 token 上限且 JSON valid=0，触发已有训练保护。保存的第 40/80 步 checkpoint 将单独作完整开发集评测，失败、未完成状态和全部成本保持不变。失败控制的报告规则先于任何新 holdout 输出固化，不能以失败控制的低分替代 OPD 对强 SFT 的主比较。

序列等权学习率对照补齐（2026-10-25）：固定相同起点、Teacher、seed=29、80 updates、10 prompts/update 和 FP32/microbatch=4，追加纯 OPD lr=1e-5 / 1e-4，offline 与 oracle SFT 各 lr=5e-5 / 1e-4。与序列等权原组一起比较，确保强 SFT 同样经过学习率开发筛选；仍为完整无过滤离线 Teacher 标签，所有尝试进入开发选型池。四张 H20，最多四小时，总并发不超过 12 卡。

正式共同更新预算的开发选型：候选、offline 与 oracle 均按所冻结最终步数（40 或 80）下的完整 1,100 条 BF16 开发评分排序，再生成四对新种子合同。原始训练内 FP32 分数和各自最佳 checkpoint 都保留，但不能代替此同最终更新步数的选择。若 selected-dev 恰好评分了该步数，复用其原始逐行输出并保存来源哈希；其他 checkpoint 在已完成任务释放的空闲 GPU 上补评，全部时长继续计费。

较大预算开发追加（2026-10-25）：统一第 80 步完整评分已显示强离线 SFT 的业务 macro 为 0.677070（token lr=5e-5）和 0.678944（trajectory lr=2e-5），已完成 OPD 尚未建立优势。追加同一第 320 步 Student、修复 Teacher、fresh Adam seed=29 的 320 次更新、每步 10 prompts，共 3,200 prompts。纯 OPD 分别为 token lr=5e-5、trajectory lr=2e-5；offline SFT 为 token lr=5e-5 与 trajectory lr=2e-5/5e-5；oracle SFT 为 token lr=5e-6/1e-5 与 trajectory lr=5e-5。两类 SFT 各自独占 GPU 顺序训练，始终使用同一起点，不承接前一个 SFT arm。每 80 步检查完整业务 dev 和实际可用通用 dev，训练完成后自动补齐第 320 步完整 1,100 条 BF16 开发评分；保留各自 selected-dev 和全部第 80 步失败/成功结果。四张 H20、最多六小时，全部成本计入，在现有任务释放资源后按总并发不超过 12 卡提交。仍不生成独立 holdout 输出，后续若采用 320 更新正式预算，必须重新冻结该预算下的强对照及确认性合同。

较大预算补充：第 80 步完整 BF16 评分的 trajectory OPD lr=5e-5 为 0.677926，仍低于最强 offline 的 0.678944。将 trajectory lr=5e-5 的 group=2 和 group=4 两个纯 OPD 配置纳入 320 更新开发池，同步补齐原始/修复 Teacher 的 Full-KL lr=2e-5、group=2、token 归约控制，起点、seed 和 prompts 完全相同。group=4 提供更多即时采样轨迹，但不将其报告为同 response-token 或同 GPU 时间预算。Full-KL 开发保护失败保持失败身份，额外评分不得伪造成功训练。四卡最多六小时，所有尝试、完整评分和成本进入最终开发选型审计。


320 更新补充开发（2026-10-25）：追加 trajectory OPD lr=1e-5、token OPD lr=2e-5，及 token offline / oracle SFT lr=2e-5。继续使用相同第 320 步 Student、修复 Teacher、fresh Adam seed=29、3,200 prompts 和 FP32。全部 16 个当前 320 更新配置与此前 26 个 80 更新配置进入保留记录；正式选型仅比较当前共同预算下的完整 1,100 条 BF16 评分。四张 H20 最多六小时，总并发不超过 12 卡，平台预检成功、无告警后提交。

80 步 trajectory-tuned 平台任务进入终态时，仅三项补评已完整持久化；另三项只有启动记录，不能算完成。新补充任务先重新评分六个不可变最终 checkpoint，保留已有逐行输出和重复评分变化，随后才开始上述新训练；新增评分与准备全部计费。确认任务调度将 offline / oracle SFT 共用一个 GPU 顺序执行，三个较重的 OPD / Full-KL 各自独占一个 GPU，所有 family 每个 seed 都从共同起点重新初始化。确认协议尚未冻结，留出集预测尚未生成。


正式优势的强 SFT 约束（揭盲前补齐）：预先指定的 offline SFT 保持主比较，oracle SFT 增加为必须通过的强对照检验。四对种子、相同业务 macro / query CI / 非劣效 / 绝对类别门规则，在主测试和不相交复验上均须分别胜过两种 SFT；不能在两种比较间选较有利者。分析入口逐行核验全部 arm 后同时输出两种比较，仅二者全部通过才标记质量优势。CPU 回归验证“胜过较弱 offline、仅追平更强 oracle”不能获得优势结论；该回归不是 GPU 结果。


运行证据补齐（2026-10-25 10:54 UTC）：补充 worker 已完成六个 tuned 最终 80 步 checkpoint 的完整 BF16 DEV 评分，缺失结果不再以启动记录代替。六项重复与原始输出继续保留；序列等权 offline lr=5e-5 的完整 business macro 为 0.679209。当前 320 更新开发仅两项固定步数完整评分已收齐，候选仍在运行，尚未冻结正式确认协议。

当前原始 Teacher Full-KL（long_original_full_kl_lr2e5）于第 80 步触发既有 Joint training alarm persisted and dev confirmed degradation，最后训练内完整 FP32 DEV 的全部域分数为零。训练失败与非零退出码保留；GPU2 已实测仅 3 MiB 占用后，核对其 saved atomic checkpoint 的 adapter、optimizer/RNG 和 state 与持久化文件哈希一致，再启动唯一保存 checkpoint 的完整 BF16 DEV 补评。该额外评测不建立成功训练或同预算对照身份。

学习曲线与卡复用（2026-10-25 11:06 UTC）：`DEVELOPMENT_CURVES.json` 保存各 arm 的实际训练内 DEV 覆盖、精度、梯度更新进度和响应 token 消费。当前训练内 DEV 共 1,400 行，最终选择仍只使用既定完整 1,100 行 BF16；两者不能直接混排。trajectory OPD lr=5e-5 的训练内 business macro 从第 80 步 0.68408 回落到第 160 步 0.67677，暂未形成稳定优势。原始 Full-KL 补评完成后，只有其 evaluator 成功文件、全部评测产物持久化哈希、driver 身份、进程释放和 GPU 空闲检查都通过，才允许该卡继续评分其他不可变开发 checkpoint；训练失败身份不因 evaluator 成功而改变。


640 更新开发扩展（2026-10-25）：320 更新的预声明 45 个原始/平均候选全部评分并独立重判。纯 token OPD lr=5e-5 为 0.6893445，最强 offline trajectory lr=5e-5 的 all4 factor mean 为 0.6919865，最强 oracle trajectory lr=5e-5 的 last2 为 0.6894761。相对差异分别为 -0.0026420 和 -0.0001316，均未建立优势；完整池保存在 `DEVELOPMENT320_FINAL.json`，不揭盲留出集。

新的共同预算为 fresh Adam、seed=29、640 updates / 6,400 prompts、同一 Student320 和修复 Teacher。四个纯 OPD 配置覆盖 token / trajectory 与 lr=2e-5 / 5e-5；offline 和 oracle 各覆盖两种归约与 lr=1e-5 / 2e-5 / 5e-5，共十二个 SFT 配置。全部无过滤离线标签、FP32、microbatch=4、相同 prompt schedule；训练检查每160步。每个配置均评最终640、480/640 last2 和160/320/480/640 all4，共48个候选，须全池完成再选型。平均是FP32 LoRA factor均值，来源checkpoint及optimizer/RNG保留；不是BA增量均值或可续跑的平均optimizer。

两项四H20作业分别执行OPD与SFT，单作业八小时窗口、名义最多32 GPUh，总并发仍不超过12卡。预检error阻止提交，warn/hint展示；预检SID保留式合并到最终payload并在30分钟内提交。v1/v2/v3打包在CPU预检中分别发现规则/driver/源码完整性问题，均未提交、无GPU产出；v4绑定已恢复并核验的源码及平均规则。源码工作区漂移的原因未知，旧文件和哈希审计保留，不能凭猜测归因。30项CPU回归及执行包源码/driver/hash/Shell检查通过，不构成GPU或模型质量结果。

640 更新对照补齐（2026-10-25 13:26 UTC）：追加纯 token / trajectory OPD lr=1e-5，
及原始 / 修复 Teacher Full-KL lr=2e-5。三项作业各四张 H20、各最多八小时，
总并发上限 12 卡，第三项预检成功且无告警。OPD、offline 和 oracle 各有两种归约、
三档学习率及三种最终 adapter，共各 18 个候选；两项 Full-KL 各三个候选，完整声明池共
60 个。若原始 Full-KL 触发已观察的训练保护，只能保留其失败证据，不能删掉尝试或伪造
同更新预算完成。其余方法的未知失败仍阻止正式冻结。

`.ops/freeze_advantage_budget640.py` 仅在完整池独立重判、20 个实际训练来源的二进制
审计完成，且纯 OPD 业务 macro 高于两类最强选定 SFT 时生成正式合同。最高
`0.8 business + 0.2 general` 规则对三类方法一致；未满足条件时继续 DEV，留出集仍不可用于
选型。CPU 验证包含缺失候选、学习率网格缺失、oracle 更强或追平、未知失败、提前揭盲、
目标漂移和重复候选的拒绝测试。58 项 CPU 回归通过；四种新种子共 20 arm 的 640 步
执行包演练通过源码、配置、Shell、平均规则、原始失败保留和终态归档检查。演练产物
显式禁止提交，不是 GPU 质量结果。

三项运行作业另有有时限的 CPU 证据归档进程，等待全部声明 DEV 输出及允许的原始
Full-KL 失败证据完成后，把逐文件 SHA256 绑定的 JSON/JSONL/log 保存至 HDFS。
该进程不启动 GPU 工作或暂停 coordinator。最终模型二进制审计、平台终态成本和
独立留出集优势仍分别验证，不能由归档启动记录代替。


运行归档补充：独立、限时的协调进程只在四个训练子进程均已启动后暂停任务
coordinator，训练子进程继续运行；归档文件持久化且 SHA256 验证通过后恢复
coordinator，另有八小时硬期限。已分配的等待时间完整计费。这与上述 CPU 归档进程
是两个不同进程，不能把 coordinator 暂停解释为训练暂停或免费等待。

2026-10-25 15:57 UTC 的真实产物验证：已完成的四项低学习率 SFT 来源均通过
GPU worker 上的 CPU 二进制审计。另从 HDFS 下载实际
`deep_offline_trajectory_lr1e5` 的四个 checkpoint、Adam/RNG、配置与两种平均 adapter，
在本地 CPU 上独立通过相同审计，结果与 worker 证据一致。首次本地下载把
`servedistill-reference` 挂载别名错当成远端目录而失败；修复后直接解析提交 payload
中的 HDFS 挂载映射并拒绝未声明路径。两次尝试都保留。该验证不启动 GPU 或生成
独立留出集输出，证据为
`.ops/advantage-budget640-sft-v4/cpu-binary-audit-20261025T155602Z/completed.json`。

备用开发规则的诊断：`.ops/lora_delta_average.py` 用 QR/SVD 平均实际 BA 增量，再压回
原 rank；规则仍只用于开发准备，不纳入当前 60 个候选或既有确认协议。在实际
离线 trajectory lr=1e-5 的 150 个 LoRA 模块上，last2 / all4 的因子均值相对增量均值
误差分别为 0.0215% / 0.0543%，rank 压缩误差为 0.0210% / 0.0529%。该差异很小，
目前没有 GPU 质量评分或收益结论。完整数学验证和实际权重诊断均保留。

2026-10-25 16:40 UTC 的开发质量诊断：`DEVELOPMENT640_GATES.json` 将已完成
12 个候选的预测哈希、绝对域分数、JSON 合法率及类别 recall 绑定到开发报告。
其中 7 个通过既定绝对下限；5 个低学习率 oracle 候选未达到 relevance macro-F1
0.45，其中一个也未达到 substitute yes recall 0.30。该诊断保留全部候选，
不改变最高 `0.8 business + 0.2 general` 的选型规则，不排除或替换强 SFT 对照；
更不代表独立留出集、配对种子或非劣效检查已经通过。

2026-10-25 16:45 UTC 的 480 步训练内诊断：已完整收齐并逐行独立重判 12 个
可用来源、每项 1,400 条 FP32 预测，保存为 `INTERIM640_FP32_STEP480.json`。
按相同最高 `0.8 business + 0.2 general` 规则，当前 OPD 为 trajectory lr=2e-5，
business macro 0.6781198；该步可用 offline token lr=1e-5 为 0.6724061，
oracle token lr=1e-5 为 0.6728731。探索性差异分别 +0.0057137 / +0.0052467，
query CI 分别 [-0.0128611, 0.0237424] / [-0.0182445, 0.0288289]，都跨零。
原先 token OPD lr=5e-5 的 320 步 0.7033992 回落到 480 步 0.6704394，
substitute yes recall 仅 0.3217391，接近既定 0.30 下限。完整声明池尚未完成，
没有独立优势；不得按早期峰值停止、替换最终候选或揭盲留出集。

2026-10-25 16:56 UTC 的执行衔接：`.ops/continue_advantage640.py` 只读取完整 60 候选
开发报告和 20 来源二进制审计，直接复用既有 `freeze_advantage_budget640.select`；
任一输入未齐时不冻结或发起确认。当前真实未齐状态的 check-only 检查通过。
全部门禁满足后先执行 freeze check-only、再一次性生成正式合同，准备每作业八小时
的四对种子确认，并由既有提交器重查实际全局 12 H20 容量及最终 payload 预检。
开发候选不胜出则保持留出集未揭盲并返回继续开发。源码恢复包 v5 保存 17 个文件，
原 v4 的 16 个文件无漂移，没有修改 GPU 训练实现。

`OPD640_OBSERVED_RUNTIME.json` 保存截至 16:58 UTC 的六项纯 OPD、共 3,040 次
已收集更新的运行诊断：每次记录的 total loss 均等于 OPD loss，最大 rollout/recompute
logprob 误差 0.0001006，低于 0.15 容忍阈值。该记录只覆盖已观察更新，不代表训练
终态、最终质量或留出集优势。

2026-10-25 17:13 UTC 的低/中学习率 SFT 阶段结果：八个来源的全部 final / last2 / all4，
共 24 个候选、各 1,100 条 BF16 DEV 已完整独立重判，保存阶段指纹于
`DEVELOPMENT640_SFT_LOW_MID.json`。按原规则当前 offline 仍为 trajectory lr=1e-5
final，business 0.6899657 / general 0.505；oracle 更新为 trajectory lr=2e-5 all4，
business 0.6639265 / general 0.545。`DEVELOPMENT640_GATES.json` 同步覆盖全部24项，
仅保存诊断，不排除不合格对照。两项中学习率 offline 已通过真实二进制审计，累计6/20。
高学习率 SFT 及 OPD / Full-KL 最终候选仍未全部完成，不能冻结或宣布优势。
执行衔接的缺候选、缺审计、更强 oracle 和正向完整池四种 CPU 门禁演练均通过，
任何子进程动作都被拦截；演练不是实际冻结、提交、预测或 GPU 验证。


2026-10-25 17:48 UTC 的主 OPD 终步证据：四个主配置的完整 2,560 次更新及每项
1,400 行最终 FP32 训练内 DEV 均已逐行核验，保存为
`OPD640_MAIN_TRAINING_COMPLETE.json`。每次 total loss 都等于 OPD loss，无训练告警，
最大 rollout/recompute logprob 误差为 0.0001006，Teacher 冻结核验通过。
这仍不同于全部实际 checkpoint 二进制审计、最终 BF16 选型和平台终态成本。

首批 640 步 BF16 DEV 的 token OPD lr=5e-5 business macro 为 0.6911694，
相对当前最强 offline 的探索性差异为 +0.0012037，query CI
[-0.0236718, 0.0253669]。relevance 增益 +0.0823146 同时伴随 substitute
退步 -0.0891227；数学 success 从 offline 0.11 降为 0.06。
不能把小幅总体正差称为独立优势；平均候选、高率 SFT 和剩余对照仍在运行，
最高 `0.8 business + 0.2 general` 规则、全部强 SFT 控制与质量门保持不变。

`OPD640_CLASS_DRIFT.json` 逐行核验全部已完成 FP32 训练内 DEV 曲线并保留类别
recall、预测数及完整窗口运行指标；`OPD640_SAMPLE_MIX.json` 绑定实际保存的训练
轨迹、原标签和离线 Teacher 类别分布。Teacher 硬标签的 substitute 正类 recall
为 0.56375，平衡训练集中的正类预测占比为 602/1600。该分布只描述硬输出，
不能代替 Teacher 条件 logprob 或证明回落因果。
导出的 `figures/opd640-training-dev.png` / `.svg` 显式标注 FP32、1,400 行与 seed29，
另有来源 SHA256 元数据；图中不补不存在的后续点，不与 1,100 行 BF16 排名混排。
新增两个独立诊断入口未修改 v5 恢复包的 17 个文件，源码零漂移。


2026-10-25 18:32 UTC 的隔离开发准备：全部六项纯 OPD、低/中率 SFT 与两项 Full-KL，
共48候选、16来源二进制审计已收齐；高率 SFT 仍在运行。主 OPD 平台已进入成功终态，
当前最高原规则 OPD 为 token lr=5e-5 all4，business 0.6743257 / general 0.52，
仍低于当前最强 offline 的 business 0.6899657。其余候选、审计和终态成本未齐，
因此不冻结或揭盲独立留出集。

`.ops/decay640-source-v1` 仅为后续 DEV 的隔离 CPU 准备，不修改现有 v5 的17文件。
依据已观察的320步以后域回落与通用退步，准备640更新前320步恒定、后320步线性
衰减到初始学习率10%的假设；同一 Student320、Teacher240、fresh Adam seed29、
6400 prompts，三方法各自同时筛选 token/trajectory 与三率1e-5/2e-5/5e-5，
并保留两项 Full-KL。60个新final/last2/all4与60个既有constant候选共同排序，
不能通过删除较强旧SFT或只给OPD调参获得优势。

47项CPU验证通过，包括20个实际旧配置的prompt序列不变、学习率端点与单调性、
恒定日程真实Adam更新逐位等同，以及六个实际CPU Adam断点恢复。三个具体执行包
演练通过Python/Bash/源码hash/预检payload一致/平均driver/终态归档成员hash检查，
且演练合同明确禁止提交。允许原始Full-KL失败与未知失败归档拒绝分支均通过。
新的审计模板逐步核验learning_rate_used及保存Adam param_groups中的lr，
仅CPU源码和数值验证不能代替GPU训练、模型优势或终态成本。

新正式任务尚未准备或提交；启动门必须先满足现有完整60候选、全部20实际
二进制审计、旧选型明确不胜两种强SFT及全部旧分配区间终态成本闭合。
若旧池通过原门，继续旧确认协议；不切换到新方法。Teacher320已完成的资格
与只读权重可用性检查保留为另一个假设，本次隔离衰减准备仍固定Teacher240。


2026-10-25 19:01 UTC 的扩展确认流程准备：既有60个640步候选已全部评分并独立重判，
当前最强offline trajectory lr=5e-5 all4的business为0.7004091 / general0.535，
当前OPD token lr=5e-5 all4为0.6743257 / general0.52。二进制审计仍为19/20，
SFT平台作业终态与完整成本尚未闭合，不生成确认协议或留出集预测。

新`.ops/freeze_advantage_combined640.py`保留全部constant与declared-decay共120候选，
对三类方法各36项执行相同最高0.8business+0.2general规则；必须同时胜过两类最强
选定SFT。全部40来源需实际saved checkpoint/Adam/RNG二进制审计与终态成本。
确认执行包使用同一隔离源码，旧constant配置显式继承constant日程；源码CPU逐位
兼容不能写成GPU验证。四pairedseed与两disjointsplit、绝对指标/recall/非劣效门
复用原始确认规则，分析仅替换被冻结源码的文件解析，统计实现保持不变。

27项CPU检查通过，包括拒绝缺旧强SFT、缺候选、缺日程网格、更强或追平oracle、
提前揭盲、未知失败、预算与评分权重漂移；四真实训练prompt序列跨20配置配对一致，
原绝对/recall/非劣效阈值逐字核验相同。四具体种子执行包经Python/Bash/hash/预检
payload一致检查，全部演练合同禁止提交。旧缺质量门的轻量夹具已更换为完整阈值
夹具，失败尝试保留；最终27项验证为`.ops/combined640-confirmation-cpu-v2/verified.json`。
`.ops/continue_advantage_combined640.py`仅在实际全120候选/20新来源审计/全部成本
终态后允许冻结；当前前台控制器等待，未提交确认。全部旧v5的17文件无漂移。


2026-10-25 19:09 UTC 旧640开发收齐：60/60候选全部独立重判，20/20真实二进制来源
审计已通过，原选择门明确拒绝冻结。最终按既定规则OPD token lr=5e-5 all4 business
0.6743257 / general0.52；offline trajectory lr=5e-5 all4为0.7004091 / 0.535；
oracle trajectory lr=5e-5 all4为0.6836209 / 0.565。原始Teacher Full-KL本轮成功
完成640步并有全部三个adapter评分，与此前80/320轮失败分别记录。

SFT四个GPU子任务实际退出码均为0，coordinator已按完成归档hash验证恢复。
运行态只读探测显示四GPU利用率0，worker正在发布约10.6GB完整产物；尚无worker
完成标记及平台终态。此前失败与空闲等待也保留计费，成本快照不能算最终完整卡时。
扩展确认的本地监控窗口已延长为36小时，覆盖开发与两批确认等待；每GPU作业仍
八小时硬限制且实际全局并发不超过12H20。没有冻结正式协议或生成holdout预测。


2026-10-25 19:40 UTC 旧轮终态闭合与衰减正式提交：旧 OPD、SFT、controls 均为
平台 done，归档全齐，60/60候选和20/20实际二进制审计通过。所有旧尝试累计
152.8355556 H20·h，历史失败、停止、评估和上传等待全部保留；峰值12H20。
父终态成本快照保存于`.ops/decay640-source-v1/prior-terminal-COST_AUDIT.json`，
SHA256 `9c6839dd2cdbb6968e8039f2249925b3330bf112dbbfbb9cc2613632c9b28a2a`。
旧本地watch已在确认无子进程后退出，未停止平台任务或发送训练信号。

三组衰减正式作业 OPD `7afe9b96b2aab611`、SFT `394cbd4a0c1f43b0`、
controls `d1d04aa49a2b15a5` 均为4H20、8小时硬上限。最终业务payload预检全部
success、无warn/hint，request_context保留式合并precheck_sid后于30分钟内创建。
任务拉起不构成GPU训练正确性、模型优势或最终卡时证据。联合池目前只收齐旧60项，
原120候选/40来源实际二进制/四pairedseed/双强SFT/双留出集质量门保持不变。

`.ops/teacher320-cpu-stage-v1/verified.json`保存后续独立假设的只读CPU权重核验：
Teacher320 adapter file/config SHA256与真实teacher prompt资格评测一致，400个
FP32 LoRA张量布局完整，rank16。其资格DEV business0.7178789/general0.65，
相对Teacher240 business略低、general略高；该CPU记录不证明模型运行或OPD收益。
本轮衰减作业仍固定Teacher240，未导出新Teacher、未提交Teacher320训练或揭盲。


2026-10-25 19:55 UTC 新任务运行态核验：三组衰减任务均running，首批12个来源的
实际保存config逐字段匹配提交计划，衰减起点320/final ratio0.10一致。训练前DEV
仍在评分，OPD已收808/1400行、SFT824/1400行、controls104/1400行；尚无原子
更新收集，不把GPU利用率或配置正确性写成优化器执行验证。阶段指纹保存于
`.ops/decay640-source-v1/observed-startup.json`。新增任务运行成本是未闭合快照，
旧152.8355556 H20·h终态快照单独保留。

新增`.ops/audit_decay640_live.py`复用原实际二进制审计源码及日程校验，完整来源
优先在其仍存活worker上执行CPU核验，非running则只走原本地CPU/HDFS下载审计。
31项CPU检查覆盖20个正式arm模板绑定、非运行态禁止执行、非法arm拒绝和旧源码
完整性；检查不是实际checkpoint/Adam/RNG审计。`.ops/run_decay640_live_first.py`
只替换原控制器审计dispatch，训练/候选/统计门不变；旧控制器于无子进程边界安全
退出，新控制器复用已提交3作业，没有重复提交。原v5、隔离日程和确认源码无漂移。


2026-10-25 20:11 UTC 的实际更新验证：12个已启动来源共收集468次原子更新，
其中70次纯OPD的total loss逐次等于OPD loss。实际learning_rate_used、rollout/
learner版本及累计响应token均匹配；最大重算logprob误差0.0000783205，低于0.15。
`DECAY640_OBSERVED_RUNTIME.json`逐来源绑定实际config/metrics和提交合同SHA256，
并保留与旧恒定日程前320步的数值差异；当前相同响应token数和微小浮点差异只描述
已观察前缀，不证明后半段衰减、saved Adam/RNG或最终质量。

Teacher320的CPU备用导出准备绑定已通过实际提示资格评分的checkpoint与原始9B
manifest，未启动新GPU工作。流式合并只保留现有Teacher240语言导出的427个原生
参数名；200个LoRA模块须与当前PEFT merge源码逐位匹配，不能提前把FP32 BA增量
舍入至BF16。基座下载、CPU数学与最终整包验证分开记录；CPU成功不等于GPU加载、
合并后Teacher资格或OPD优势。现有三组训练、完整120候选与留出集门禁不变。


2026-10-25 20:32 UTC 的开发与CPU备用导出核验：4个低率SFT均完整完成第160步
训练内FP32 DEV，各1400行（业务900、math200、instruction300），独立重判通过。
offline token/trajectory business分别0.6424469/0.6584389，oracle分别
0.6426518/0.6532277。16条完整训练曲线保存于`DECAY640_CLASS_DRIFT.json`；
这不是既定1100行BF16最终候选，不加入120候选的最终排名。

Teacher320 CPU导出v1在第4分片遇到48个原生F32的A_log/norm参数，过严的
all-BF16守卫停止执行；3个已完成分片、源码、错误类别和哈希均保留，未启动GPU。
通过仍running的OPD worker只读CPU探测，逐文件SHA256匹配实际Teacher240导出，
427个语言权重的真实dtype/shape均核对，全部BF16；证据为
`.ops/teacher320-cpu-export-v1/reference-tensor-schema.json`。

独立v2显式将上述48个非LoRA原生F32参数转换为实测BF16格式；LoRA增量仍按
FP32 BA、最后in-place加法转换精度。8个当前PEFT方法逐位数值检查及48个实际
原生参数转换检查通过，v2开始完整CPU重建。v1原始权重和部分导出不替换，不修改
当前GPU训练、旧17文件、隔离日程13文件或联合确认5文件。CPU整包、HDFS上传、
实际GPU加载、合并后教师资格与OPD独立优势分别验证；尚未发布或揭盲留出集。


2026-10-25 20:55 UTC：Teacher320 CPU v2完整成功，4分片、427语言权重、
200实际LoRA合并均通过保存后逐Tensor核验及当前PEFT CPU方法逐位核验。
10文件17,920,538,815字节已上传，manifest SHA256
`af152863f5905f5b94b3eede28c5642261fe840e7aff97da876795bc1360f659`
回读一致。记录为`.ops/teacher320-cpu-export-v2/staged.json`；这不证明worker
逐文件下载、GPU加载、合并后教师资格或OPD质量收益，CPU v1失败产物仍完整保留。

备用`.ops/advantage-teacher320-candidate-v1`已准备1H20、3小时硬上限的独立任务：
保持原1100行实际Teacher提示DEV，先GPU资格评测；通过既有business>=0.65、
substitute macro-F1>=0.60、yes recall>=0.45后，才生成全部6400训练ID的无过滤
离线标签。不执行Student训练或留出集预测。64个源码/开发数据文件的实际archive
提取及8项资格流程拒绝检查通过；生成器在CPU检查中为mock，不能算GPU资格。
任务尚未提交；仅当当前120候选、40实际binary审计及终态成本齐备且DEV未胜过
双强SFT时进入该后续路线，已有确认路线优先，全球上限12H20。

当前三组衰减作业仍running。12已启动来源共2285真实更新，其中636纯OPD更新的
total loss逐次等于OPD loss；最大behavior误差0.0000787973<0.15。低率SFT已
跨过320步，其实际后半段learning_rate_used匹配固定线性衰减；OPD尚在恒定前缀。
实时JSONL合同核验仍不代替最终saved Adam/RNG、完整120候选或终态成本审计。


2026-10-25 21:31 UTC 后续推进已就绪：Teacher320 DEV 控制器通过26项CPU检查并启动，
当前只等待既有120候选、40真实二进制来源与全部终态成本，未提交新GPU任务。
完整联合DEV若胜过双强SFT，保留原四种子确认路线优先；只有原选择器明确NoAdvantage
且逐来源audit_pool通过，才提交已有1H20/3小时的资格与训练标注任务。

资格独立重判绑定原1100行、实际Teacher提示、CPU模型manifest与逐文件worker核验；
通过原有门后才标注全部6400训练ID。标注审计核验完整唯一prompt分段、每段与最终
标签SHA256、原始ID/input/prompt、逐行评分及汇总，拒绝删行、错误过滤和来源漂移。
任务终态只下载持久化HDFS输出，失败与部分产物完整保留，不连接已释放worker；
全生命周期成本继续由原统一审计发现并计费。这些CPU检查和控制器等待不建立GPU
资格、Teacher320训练收益或独立优势。旧17文件、隔离13文件、联合确认5文件和
Teacher320已有8文件均无漂移。

截至21:25 UTC，共核验3619次新轮实际更新，其中1008次纯OPD总损失逐次等于OPD
损失；最大behavior误差0.0000824928<0.15。低率SFT已实测执行衰减后缀，六项OPD
仍处前320步恒定段。训练内FP32完整曲线已收齐六项OPD的第160步与四项低率SFT
的第480步；这些1400行曲线不加入1100行BF16最终120候选排名，也不改变最终步数。


2026-10-25 21:48 UTC：备用Teacher320公平训练执行包通过30项CPU检查，入口为
`.ops/prepare_teacher320_decay640.py`，验证记录为
`.ops/teacher320-decay640-source-v1/cpu-verification.json`。20个配置逐字段继承
当前衰减轮：共同Student320、fresh Adam seed29、640updates/6400prompts、
FP32、前320恒定后线性衰减到10%。纯OPD、offline和oracle均保留两种归约与
三个学习率；全部60个新增final/last2/all4候选及此前120候选须一起保留。
原始Teacher Full-KL保持原始Teacher身份；新的修复Teacher路径及终态archive
绑定Teacher320 manifest。没有修改训练损失、评分规则、当前GPU任务或确认门槛。

CPU演练使用旧Teacher240真实标签模拟尚未生成的Teacher320标签，所有演练合同
标记never-submit及mock_teacher_labels；这不是Teacher320标注、GPU训练或质量
结果。首次演练识别出宽泛字符串替换误改learning_rate_decay_start，错误配置
留在`cpu-rehearsal-20261025T214725843894Z/opd`，已将替换限制为带引号的arm
前缀。最终30项检查核验全部20配置与原合同完全一致，以及archive成员、Bash、
预检payload身份、纯OPD源码、已揭盲拒绝、失败Teacher拒绝和确认路线优先。

真实check-only返回waiting all120 prior candidates，未准备实际训练包或提交GPU。
入口仅在全部120候选/40真实二进制来源/所有终态成本闭合、联合DEV明确
NoAdvantage、Teacher320资格与全部6400无过滤标签独立重判通过且平台成功终态
之后准备真实包。旧17文件、隔离13文件、联合确认5文件、Teacher320既有8文件
及控制器8文件逐SHA256核验均无漂移。尚未实现或验证新增180池的正式分析路线，
因此不能把本执行包准备描述成完成了后续训练或确认。

当前实际训练进度：四个低率SFT均有640步completed；六个OPD约238次更新。
完整1100行BF16最终和平均adapter评分尚未收齐，联合选择池仍为60/120。
三个CPU控制器仍存活，原有确认和备用资格路线继续自动等待，无重复提交。


2026-10-25 22:20 UTC：Teacher320独立DEV证据工作流通过34项CPU检查，入口为
`.ops/teacher320_decay640_workflow.py`，记录为
`.ops/teacher320-decay640-source-v1/workflow-cpu-verification.json`。完整180候选
选择保留此前120候选和两类最强SFT；完整二进制门禁覆盖40个旧来源及20个新来源。
新流程分别核验实际Teacher manifest、无过滤标签、共享Student320、声明配置、
逐步学习率、保存Adam/RNG/FP32因子和final/last2/all4绑定，拒绝错误Teacher、
标签漂移、缺失来源、预测绑定变化和已揭盲。终态archive验证使用独立Teacher320
合同和实际成员SHA256，下载尝试目录用微秒避免重复审计覆盖。旧17、隔离13和
联合确认5文件仍逐SHA256不变。

CPU演练重放旧Teacher240的60份实际1100行BF16 DEV预测，明确替换为mock
Teacher320 provenance，另使用小型合成FP32权重/Adam/RNG和模拟HDFS IO。
这些检查不证明实际Teacher320资格、标签、训练或质量。两次失败夹具及哈希已
保留：第一份误写部分配置而非runtime完整默认值，第二份测试根目录替换同时
改变了模板读取路径；修正后通过34项。实际新Teacher320训练包尚未准备或提交。
正式的Teacher专属四seed确认、完整训练生命周期与自动衔接仍需实现并独立核验，
完成之前不能启动新增20分支训练。原有联合640确认路线保持优先。

截至22:19 UTC，六条纯OPD均已实测320次更新，实际学习率仍为前320步恒定值，
总loss逐步等于纯OPD loss，最大behavior误差0.0000886917<0.15。联合最终候选
已完成72/120，新增实际二进制审计4/20，四条低率SFT已完成训练和全部3份候选。
训练内1400行FP32第320步开发评测正在运行；不能与1100行BF16最终候选混排。
三个CPU控制器存活，当前三个4H20作业仍running，终态成本和独立优势尚未成立。


2026-10-25 22:51 UTC：Teacher320专属确认入口`.ops/teacher320_confirmation.py`完成15项
CPU检查，记录为`.ops/teacher320-decay640-source-v1/confirmation-cpu-verification.json`。
四seed、20arm的具体包逐一核验教师路径、标签全集字节、源码/配置/archive哈希及Bash/Python。
新OPD可保留Teacher320，最强旧offline仍使用其实际Teacher240；每个所选来源分别绑定
Teacher manifest、提交合同、实际provenance和完整无过滤标签。原absolute/recall/非劣效/
positive-seed质量阈值的AST字面值完全一致；统计源码未修改。新remote root与归档名独立，
不覆盖已有confirmation-v1。首次CPU运行遗漏overlay命名空间，失败源码哈希与记录保留；
补齐后具体四包通过。IDE apply_patch持续timeout，确认文件未生成/未变后用精确本地写入。

演练为显式synthetic180排名、mockTeacher320身份及重放Teacher240标签，实际binary与
terminalcost仅在隔离freezer夹具内mock；包的rehearsal标记禁止提交。这些不是GPU质量结果
或真实冻结。完整分析运行拒绝测试及训练lifecycle/自动衔接仍未完成，因此仍禁止新增DEV
训练启动。当前既有三个控制器继续等待120候选、40真实binary和终态成本；留出集未揭盲。


2026-10-25 23:24 UTC：Teacher320确认分析完成24项CPU检查，实际运行原评分、统计和
平均adapter核验，覆盖20arm、四seed和两个合成不相交分片。评分/缺行/重复ID/input、
summary/config/checkpoint/protocol/adapter、缺oracle控制与缺复验输出均拒绝；
仅胜offline但追平oracle、任一复验失败也不成立优势。新offline选中分支同时核验
其实际教师身份和完整标签路径。合成数据与小型FP32保存权重均明确never-submit，
没有生成或使用真实holdout预测。统计与原absolute/recall/非劣效门未修改。

新`.ops/run_teacher320_decay640.py`与`.ops/continue_teacher320_advantage.py`完成30项
CPU生命周期检查；五份记录合计133项、45个源码SHA256。具体三包准备先于任何submit，
12H20容量暂不足只等待重试；缺180候选/20新binary/终态成本不冻结，平台未知失败
保持失败证据且不连接退役worker；真实freeze或独立预测存在时禁止继续DEV。
冻结后恢复直接进入独立`.ops/advantage-confirmation-teacher320-v1`，四seed20arm、
双强SFT及双split分析继续复用原门。所有CLI/GPU/HDFS/freezer/runner动作在CPU演练
中mock，不能把合成优势当成GPU结果。一次容量夹具期限错误及实际入口导入顺序
错误均保留；后者先加载verified decay overlay再导入教师auditor，直接prepare入口
三包30项重新通过，三个独立新进程均核验实际Config来源。旧17/13/5源码不变。

三个实际check-only入口全部rc0。PID 2944423的前台后续控制器已启动，仍等待真实
120候选/40来源与成本终态；新增Teacher320训练未准备或提交，Teacher320资格任务
也仍等前置门。原三控制器继续运行，无重复submit。当前联合DEV仍72/120、新binary
4/20，三个既有4H20作业running；未冻结、未揭盲、独立质量优势仍未成立。
