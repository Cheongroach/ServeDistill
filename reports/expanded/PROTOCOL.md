# ServeDistill 扩大实验：固定轨迹性能与训练质量回归

状态：2026-10-25 已完成终态验收，结果见 [REPORT.md](REPORT.md)。以下执行设计和主统计判据保持冻结；m8/m4 比较及逐类退化审计为标注的探索性追加分析。

目的：验证 microbatch 优化在多批次、重复计时和较长训练中是否稳定，并区分执行性能、数值一致性、被测任务质量。先前单个 update 的 283 response tokens 和百万级 kernel 数不足以支持稳态加速或质量保持结论。

## 固定条件与资源

- Qwen3.5-9B Teacher → Qwen3.5-2B Student；原有固定模型文件、Transformers 5.2.0、PyTorch 2.9.1、FP32/PyTorch DeltaNet Student、BF16/FLA Teacher、LoRA r=16。
- 共同父状态为原 Full-KL seed17 的 update40，adapter/Adam 指纹相同。性能比较固定该状态与轨迹；较长训练从该父状态显式 fork，重置实验步号和监控历史，以新的种子生成 RNG，不冒充原任务的无缝续训。
- 继续使用原 H20 GPU 队列。拟用 2 卡性能任务和 8 卡训练任务，峰值 10 卡；性能脚本最长 4 小时、训练脚本最长 6 小时。名义上限 56 GPUh，平台初始化和资源释放时间另记；实际按 pod schedule_at→stop_at 核算。
- 启动前执行实际预检；error 拦截，warn/hint 保留。任务完成后收齐终态、产物和成本，释放全部本实验资源。

## 更大的独立数据

从原固定 ESCI parquet 重新抽样，排除旧 train/dev/test/ood 的全部 query group；新 split 之间按归一化 query 隔离。GSM8K 和生成指令排除旧样本/input hash，并保持新 split 间隔离。

| Split | 搜索/相关性/替代品（每域） | 数学 | 指令 | 总量 |
|---|---:|---:|---:|---:|
| train | 3,000 | 2,000 | 3,000 | 14,000 |
| dev | 300 | 200 | 300 | 1,400 |
| test | 1,500 | 600 | 1,500 | 6,600 |
| replication | 500 | 200 | 500 | 2,200 |

具体可用数量以生成后的 hash/leakage 验收为准。ESCI 使用官方独立标签，自然类别比例保留；replication 是独立官方 test query 复测，不声称真实分布外场景。数学与指令仍是有限任务族，不代表完整通用能力。

## 固定轨迹性能实验

- 两个 H20 worker，各选 128 个不同 prompt，按用户 prompt token 长度和领域覆盖组成 16 个 bank；每个 bank 为 8 prompts×4 completions，共 32 条轨迹。长度不含 chat template 与 system 文本。
- Student 在共同父状态生成轨迹，冻结 token、EOS mask、behavior logprob 和 sample IDs。比较 microbatch=1/2/4/8。
- 每个 bank 和 microbatch 先预热完整一步；正式重复 3 次，交替、平衡执行顺序。每次在计时之外恢复相同 adapter/Adam/RNG，避免更新状态、warmup、生成随机数混杂。
- CUDA 同步界定完整一步；性能主结果来自无 profiler。记录有效 response token/s、完整 learner update 时间、各阶段范围、每步 allocated/reserved 峰值、梯度及 update delta；准备、生成、编译另记。
- 32 bank×4 microbatch×3 repetitions=384 次正式一步计时，另有 128 次预热；256 个独立 prompt/1,024 条固定轨迹。重复执行次数不冒充独立数据数量。
- 对少量已预热 bank 追加 bounded Nsight CUDA/NVTX capture，用于解释发射结构；不将 profiler 时间并入性能主结果。

数值拆分：原生 Teacher batch 路径记录 post-clip 梯度 cosine/相对 L2 与参数 update 差；对少量 bank 固定同一份 Teacher logits，再比较 Student 的 post-clip 梯度。behavior logprob 最大误差由每步概率门禁单独记录。固定 Teacher 模式用于分离 Teacher batch/padding 与 Student 累积因素，不作为原生整步吞吐基线；未保存跨 microbatch 的逐 token logprob 差，也不把 post-clip 比较冒充完整 pre-clip 梯度一致性。

统计单位为 bank，先取 bank 内重复的中位数，再配对 bootstrap bank；报告 95% CI 与长度分桶结果。只有无 profiler 的吞吐改进 CI 下界超过 1，才称为本 workload 的稳定改善；不外推 serving P95。

## 较长的 on-policy 训练

- seeds=17/29/43/71，每种子成对比较 m1/m4，共 8 arms。
- 每 arm 240 optimizer updates、8 prompts/update、group_size=4、rollout_batch=4、max_new_tokens=128；相同初始 adapter/Adam、prompt schedule 和种子。总计划 1,920 updates、61,440 条生成轨迹，实际有效 token 数落盘。
- Full reverse KL、token aggregation、学习率 2e-5、gradient clipping 1.0；不同时改精度或融合 kernel。概率和有限值门禁保留。
- 每 60 updates 用固定的 dev 子集（每域 40 条，共 200 条）评估并保存。主质量终点固定 update240；dev-selected checkpoint 只作次级结果，test 不用于重选或调参。
- 父模型在新冻结评测集上另测一次；每 arm 在 6,600 test 与 2,200 replication 上以 BF16 部署形态评估。输出按 sample ID/input hash 配对；主要比较 m4 与同种子的 m1。

任务指标：搜索 NDCG/MRR、相关性/替代品 macro-F1、数学/指令 success、JSON 合法率、截断率。并报告训练阶段、评估、编译、准备与驻留的完整时间和有效 GPU 成本。

质量非劣效边界预设为每业务域 −5 pp：按 query 成对重采样、保持同一 query 的跨 arm/seed 输出绑定；报告各域 CI 与各 seed 结果。均值接近、p>0.05 或单种子通过都不等于质量等价；CI 不够窄时结论为未确定。四种子仍不足以声称跨所有训练随机性的普适结论。

## 产物与验收

保存数据 manifest、源码/model/parent hashes、固定轨迹合同、逐次性能记录、逐步训练指标、冻结评测输出、bootstrap 脚本、代表 trace、终态和成本。原始内部路径/资源 ID 不进入公开仓库。

CPU 验证覆盖数据隔离、bank/顺序、总体 token loss 权重、fork 状态和成对统计。GPU 验证独立报告；提交、Running、环境 probe 或 CPU 测试通过不作为最终性能/质量验收。
