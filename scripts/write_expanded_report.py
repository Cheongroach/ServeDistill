"""Write the expanded study report only from validated, terminal evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAINS = ("search", "relevance", "substitute", "math", "instruction")
METRICS = ("NDCG", "macro-F1", "macro-F1", "success", "success")


def read(path):
    return json.loads(path.read_text())


def percent(value):
    return f"{value * 100:.2f}"


def ci(values, scale=1):
    return f"[{values[0] * scale:.2f}, {values[1] * scale:.2f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    output = args.statistics
    validation = read(output / "validation.json")
    costs = read(output / "cost-audit.json")
    if not validation["completed"] or not costs["all_terminal"]:
        raise ValueError("Final reporting requires validated artifacts and released allocations")
    perf = read(output / "performance.json")
    test = read(output / "quality-test.json")
    replication = read(output / "quality-ood.json")
    training = read(output / "training-summary.json")
    parent = read(output / "parent-scores.json")
    classification = read(output / "classification-audit.json")["splits"]
    secondary = read(output / "secondary-dev-selected.json")
    manifest = read(args.data / "manifest.json")
    m4 = perf["comparisons"]["4"]
    m8_vs_m4 = perf["exploratory_m8_vs_m4"]
    joint = test["all_business_domains_noninferior"] and replication["all_business_domains_noninferior"]
    lines = [
        "# ServeDistill：扩大实验结果",
        "",
        f"Qwen3.5-9B → Qwen3.5-2B Full reverse-KL OPD，H20，"
        f"microbatch 1/2/4/8 的固定轨迹比较，以及 m1/m4 四组配对种子的较长训练。",
        "",
        f"无 profiler 的 m4 learner 吞吐为 m1 的 **{m4['paired_throughput_ratio']:.2f}×**，"
        f"32 bank 配对 bootstrap 95% CI **{ci(m4['paired_bank_ci95'])}**。"
        f"新测试集和独立 query 复测上的三业务域 −5 pp 非劣效联合判据"
        f"**{'全部通过' if joint else '没有全部通过'}**。"
        f"这两个结论分别来自固定轨迹计时和实际训练后评测。",
        "",
        "非劣效结论只比较 m4 与 m1。逐类复核发现 8 个最终模型在两个 split 的"
        "替代品任务上都只预测 `no`，正类召回率为 0；该域保留的是退化的分类器。"
        "相关性分类也明显偏向 E/I。执行优化成功不能作为业务模型可用性的证明。",
        "",
        f"已验收 **8 arms × 240 updates**、**{validation['response_tokens']:,} 个有效训练 response token**。"
        f"本次扩展按 Pod 调度到释放计 **{costs['scheduled_gpu_hours']:.4f} H20 GPUh**，"
        f"峰值 {costs['peak_scheduled_gpus']} 卡，全部释放；"
        f"加上前期质量研究和诊断，项目合计 **{costs['project_total_gpu_hours']:.4f} GPUh**。",
        "",
        "## 为什么此前的小数据不够",
        "",
        "此前 283 个 response token、一次捕获 update 的百万级 kernel，"
        "能帮助定位细碎发射，不能给出稳态性能置信区间，也不能证明训练质量保持。"
        "本实验把实际计时重复、独立 workload 数、训练种子和独立评测 query 分开计数。",
        "",
        "| 证据 | 本次规模 | 统计单位 |",
        "|---|---:|---|",
        f"| 固定轨迹 | 256 prompts、1,024 trajectories、{perf['n_unique_trajectory_response_tokens']:,} response tokens | 32 个独立 bank |",
        "| 正式计时 | 384 updates，正式对照前 128 个预热 update | 每 bank 内 3 次取中位数 |",
        "| Nsight 追加 | 4 个预热 update、4 个捕获 update | 代表 bank 的机制解释，排除出性能 CI |",
        f"| 较长训练 | 4 seeds × 2 microbatch × 240 updates；"
        f"{validation['generated_training_trajectories']:,} 条验收生成轨迹 | 4 对训练种子 |",
        "| Test / replication | 6,600 / 2,200 queries，各 checkpoint 完整评测 | query 配对，绑定四个种子 |",
        "",
        "## 固定轨迹性能",
        "",
        "每次在计时外恢复同一 adapter、Adam 和 RNG，固定 token、EOS mask、behavior logprob。"
        "计时包含完整 learner update 与 CUDA 同步，排除 rollout、状态恢复、CPU 向量导出和 profiler。"
        "按 bank 取重复中位数，吞吐估计为总有效 token 除以 bank 中位耗时之和。",
        "",
        "| microbatch | response token/s | 相对 m1 / 95% CI | bank update 中位 / P95，s | allocated / reserved 最大值，GiB |",
        "|---|---:|---|---|---|",
    ]
    for m in (1, 2, 4, 8):
        row = perf["comparisons"][str(m)]
        lines.append(
            f"| {m} | {row['response_tokens_per_second']:.2f} | "
            f"{row['paired_throughput_ratio']:.2f} / {ci(row['paired_bank_ci95'])} | "
            f"{row['median_bank_update_seconds']:.2f} / {row['p95_bank_update_seconds']:.2f} | "
            f"{row['maximum_peak_allocated_gib']:.2f} / {row['maximum_peak_reserved_gib']:.2f} |")
    lines += [
        "",
        "![吞吐与显存](performance-memory.png)",
        "",
        f"探索性追加比较：m8 相对 m4 的吞吐比为 "
        f"{m8_vs_m4['paired_throughput_ratio']:.3f}×，32 bank 配对 95% CI "
        f"{ci(m8_vs_m4['paired_bank_ci95'])}；allocated 峰值额外增加 "
        f"{perf['comparisons']['8']['maximum_peak_allocated_gib'] - m4['maximum_peak_allocated_gib']:.2f} GiB。"
        "该比较不是预设主终点；区间包含 1 时，不能确立 m8 有额外稳定收益。",
        "",
        "P95 是 bank update 的分布统计，不是 serving 请求 P95。"
        "显存是本步骤的 PyTorch 高水位，未包含非 PyTorch 分配。"
        "置信区间条件于当前父状态、两个 worker 与该 workload；不代表跨所有硬件和模型。",
        "",
        "| prompt 长度四分位 | bank 数 | 用户 prompt 平均长度范围，token | m4/m1 / 95% CI |",
        "|---|---:|---|---|",
    ]
    for name, row in perf["prompt_length_quartiles"].items():
        lines.append(
            f"| {name} | {row['n_independent_banks']} | "
            f"{ci(row['mean_prompt_token_length_range'])} | "
            f"{row['m4_throughput_ratio']:.2f} / {ci(row['paired_bank_ci95'])} |")
    lines += [
        "",
        "长度按用户 prompt token 数统计，不包含 system 和 chat template；"
        "分桶属于较小的条件分析，不能推导所有长上下文场景。",
        "",
        "## 四种子训练后的质量",
        "",
        "主终点固定 update240；相同种子的 m1/m4 从共同父 adapter/Adam fork，"
        "使用相同 prompt schedule。测试集没有用于 checkpoint 选择或调参。"
        "下表每个区间保持同一 query 的四个种子输出绑定，再对 query 配对重采样。"
        "CI 条件于已观察的四个训练种子，不把 4×query 输出视为更多独立 query。",
        "",
        f"次级 dev-selected checkpoint 的 8 arm 摘要另存 "
        f"[secondary-dev-selected.json](secondary-dev-selected.json)。"
        f"其中 3 个较早 checkpoint 另完成 {secondary['additional_evaluation_outputs']:,} 条输出，"
        "经过完整覆盖和独立重判；其余复用最终 checkpoint。"
        "主表和非劣效 CI 始终使用固定 update240，不按 test 结果重选。",
        "",
        "共同父 checkpoint 是已经训练过的 update40，并非未训练的 2B 基座。"
        "下面的父状态→最终状态变化是描述性的延长训练结果；"
        "m4 对 m1 的配对比较才是本次 microbatch 修改的主对照。",
        "",
        "| 领域 | Test 父状态，% | Test m1 / m4，% | Replication 父状态，% | Replication m1 / m4，% |",
        "|---|---:|---|---:|---|",
    ]
    for domain in DOMAINS:
        a, b = test["slices"][domain], replication["slices"][domain]
        lines.append(
            f"| {domain} | {percent(parent['test'][domain])} | "
            f"{percent(a['m1_mean'])} / {percent(a['m4_mean'])} | "
            f"{percent(parent['ood'][domain])} | "
            f"{percent(b['m1_mean'])} / {percent(b['m4_mean'])} |")
    lines += [""]
    for title, result in (("Test", test), ("独立 query replication", replication)):
        lines += [
            f"### {title}",
            "",
            "| 领域 / 指标 | 独立 query / 指标有效 query | m1 均值，% | m4 均值，% | m4−m1，pp / 95% CI | −5 pp 非劣效 |",
            "|---|---:|---:|---:|---|---|",
        ]
        for domain, name in zip(DOMAINS, METRICS):
            row = result["slices"][domain]
            lines.append(
                f"| {domain} / {name} | {row['n_independent_queries']} / "
                f"{row['n_metric_eligible_queries']} | "
                f"{percent(row['m1_mean'])} | {percent(row['m4_mean'])} | "
                f"{percent(row['delta'])} / {ci(row['paired_query_ci95'], 100)} | "
                f"{row['noninferiority']} |")
        lines += [
            "",
            "| 领域 | seed17 / 29 / 43 / 71 的配对差，pp | 探索性 seed bootstrap 95% CI，pp |",
            "|---|---|---|",
        ]
        for domain in DOMAINS:
            row = result["slices"][domain]
            lines.append(
                f"| {domain} | {' / '.join(percent(v) for v in row['paired_seed_deltas'])} | "
                f"{ci(row['exploratory_paired_seed_ci95'], 100)} |")
        lines += [
            "",
            "| 领域 / 辅助指标 | m1 / m4，% | 配对差，pp / 95% CI |",
            "|---|---|---|",
        ]
        for domain in DOMAINS:
            for key, row in result["slices"][domain]["auxiliary"].items():
                lines.append(
                    f"| {domain} / {key} | {percent(row['m1'])} / {percent(row['m4'])} | "
                    f"{percent(row['delta'])} / {ci(row['paired_query_ci95'], 100)} |")
        lines += [""]
    lines += [
        "![配对质量差异](quality-deltas.png)",
        "",
        "这些是 pointwise CI；三业务域联合结论要求每域分别通过。"
        "−5 pp 是预先声明的可接受退化界限，通过并不表示逐位等价、零退化或 OPD 优于 SFT。"
        "四种子经验 bootstrap 只作探索，不能确立跨训练随机性的普适结论。"
        "replication 使用官方 test 的独立 query，未人为制造分布偏移，不声称真实 OOD。",
        "",
        "### 类别退化与绝对质量",
        "",
        "替代品域的 [逐类复核](classification-audit.json)显示，"
        "8 个最终 arm 的预测均为常数 `no`。下表每个 query 数是单个 split 的独立数量，"
        "没有把重复 checkpoint 输出增加到独立样本量中。",
        "",
        "| Split | 独立 query / 正类支持 | 每 arm 预测 yes | 正类召回率，% | m1 / m4 macro-F1，% | 常数多数类 macro-F1，% |",
        "|---|---|---:|---:|---|---:|",
    ]
    for split, result in (("test", test), ("ood", replication)):
        arms = [classification[split][label]["substitute"] for label in training]
        if not all(row["prediction_counts"] == {"no": row["queries"]} for row in arms):
            raise ValueError("Reported substitute class-collapse finding no longer matches evidence")
        row, quality_row = arms[0], result["slices"]["substitute"]
        lines.append(
            f"| {'Test' if split == 'test' else 'Replication'} | "
            f"{row['queries']} / {row['per_class']['yes']['support']} | 0 | 0.00 | "
            f"{percent(quality_row['m1_mean'])} / {percent(quality_row['m4_mean'])} | "
            f"{percent(row['majority_constant_macro_f1'])} |")
    lines += [
        "",
        "因此 substitute 的配对差和 bootstrap CI 都为零，"
        "来自相同的常数预测；不能解释成有用能力得到保持。"
        "相关性分类的逐类召回也显示 S/C 类薄弱，下面是四种子均值：",
        "",
        "| Split / 类别 | 标签支持 | m1 recall，% | m4 recall，% |",
        "|---|---:|---:|---:|",
    ]
    for split in ("test", "ood"):
        for label in ("E", "S", "C", "I"):
            values = {
                m: [classification[split][f"full_kl_m{m}_seed{seed}"]["relevance"]["per_class"][label]
                    for seed in validation["seeds"]]
                for m in (1, 4)}
            lines.append(
                f"| {'Test' if split == 'test' else 'Replication'} / {label} | "
                f"{values[1][0]['support']} | "
                f"{percent(sum(row['recall'] for row in values[1]) / len(values[1]))} | "
                f"{percent(sum(row['recall'] for row in values[4]) / len(values[4]))} |")
    lines += [
        "",
        "该追加分析是描述性诊断，冻结的质量终点与 −5 pp 判据保持不变。"
        "当前证据足以支持本 workload 的 microbatch 执行选择，"
        "不足以支持 OPD 优于 SFT、完整业务能力保持或生产上线。",
        "",
        "## 实际训练工作与端到端时间",
        "",
        "| arm | prompt visits / unique prompts | 有效 response tokens | 平均 response 长度 / 截断，% | train / final eval，min |",
        "|---|---|---:|---|---|",
    ]
    training_seconds = {}
    for label, value in training.items():
        counts = validation["training_schedule_counts"][label]
        durations = {
            stage: (row["end_unix_seconds"] - row["start_unix_seconds"]) / 60
            for stage, row in value["execution"].items()}
        training_seconds[label] = durations["train"] * 60
        lines.append(
            f"| {label} | {counts['prompt_visits']:,} / {counts['unique_training_prompts']:,} | "
            f"{value['completed']['response_tokens']:,} | "
            f"{value['mean_response_tokens_per_trajectory']:.2f} / "
            f"{percent(value['mean_trajectory_truncation_rate'])} | "
            f"{durations['train']:.2f} / {durations['final']:.2f} |")
    seed_time_ratios = [
        training_seconds[f"full_kl_m1_seed{seed}"]
        / training_seconds[f"full_kl_m4_seed{seed}"]
        for seed in validation["seeds"]]
    summed_training_seconds = {
        m: sum(training_seconds[f"full_kl_m{m}_seed{seed}"]
               for seed in validation["seeds"])
        for m in (1, 4)}
    lines += [
        "",
        f"14,000 条是训练池规模；实际四对种子共访问 "
        f"{validation['unique_training_prompts_across_seed_pairs']:,} 个不同 prompt。"
        "每 arm 1,920 次 prompt visit，成对 arm 共享 schedule。"
        "实际 response 长度可随 on-policy 参数轨迹改变，端到端时间比包含 rollout、"
        "形状编译、dev 评估和 checkpoint，不能与固定轨迹 learner 计时混用。",
        "",
        f"相同 240 updates 下，四个种子 train 阶段的 m1/m4 时间比分别为 "
        f"{' / '.join(f'{ratio:.2f}' for ratio in seed_time_ratios)}；"
        f"四 arm 时间总和之比为 "
        f"{summed_training_seconds[1] / summed_training_seconds[4]:.2f}×。"
        "这是观察到的完整训练时长比，排除 final eval 和外层环境准备；"
        "没有为四种子声明稳态执行 CI，也不假设实际生成 token 完全相同。",
        "",
        "![训练进度与有效 token](training-work.png)",
        "",
        "## 数值与运行验收",
        "",
        "8 arms 的概率门禁、240 次非零更新、Teacher 冻结检查、完整 step 序列通过；"
        "父 adapter/Adam/RNG 指纹一致，评测覆盖和 input hash 通过本地独立重判。",
        "执行关键源码、配置与数据的 worker hash 与公开文件逐项一致；"
        "后续分析、绘图、测试和协议说明的差异保留在 source-comparison.json。",
        f"概率 tolerance 沿用父实验的 {validation['probability_tolerance']}；实际性能 / 训练"
        f"最大误差为 {validation['maximum_observed_performance_probability_error']:.3g} / "
        f"{validation['maximum_observed_training_probability_error']:.3g}，"
        "报告实际误差而非只报告通过。Teacher 为 requires_grad=False，"
        "冻结指纹检查每个命名参数的前 32 个元素，未逐位 hash 完整 Teacher。",
        "原生 Teacher 的 batch/padding 路径和累积顺序发生变化，"
        "post-clip 梯度相对 L2 与参数 delta 差异保留在 validation.json。",
        "",
        "原始 FP32 长向量 cosine 对相同向量出现 >1 的非法值；该字段不用于数值等价结论。"
        "独立固定 Teacher、四序列的 FP64 归约与原始 FP32 字段分开记录。"
        "这是 post-clip 梯度检查，未保存完整逐 token 跨 microbatch logprob 差，"
        "也没有覆盖所有 bank 的 pre-clip 向量一致性。",
        "",
    ]
    lines += [
        "| microbatch 对 m1 | 原生 post-clip 梯度相对 L2，中位 / 最大，% | 参数 update 相对 L2，中位 / 最大，% |",
        "|---|---|---|",
    ]
    for m, row in validation["native_numerics_summary"].items():
        gradient, update = row["gradient_relative_l2_postclip"], row["update_relative_l2"]
        lines.append(
            f"| {m} | {percent(gradient['median'])} / {percent(gradient['maximum'])} | "
            f"{percent(update['median'])} / {percent(update['maximum'])} |")
    lines += [
        "",
        "上表每个 microbatch 含 32 bank；相对 L2 仍是原生 FP32 归约记录，"
        "仅描述实际差异，不是 FP64 精度证书，也不推出训练轨迹等价。",
        "",
    ]
    numerics_path = output / "numerics.json"
    if numerics_path.exists():
        numerical = read(numerics_path)
        if numerical.get("completed"):
            lines += [
                f"独立四序列 FP64 检查：cosine="
                f"{numerical['gradient_cosine_fp64']:.12f}，post-clip 梯度相对 L2="
                f"{numerical['gradient_relative_l2_fp64']:.3g}，"
                f"参数 update 相对 L2={numerical['update_relative_l2_fp64']:.3g}。"
                "仅对应保存的固定 Teacher 小片段，不能替代原生全 bank 数值差。",
                "",
            ]
        else:
            lines += ["独立 FP64 GPU 检查未完成，不能用 CPU 归约测试替代。", ""]
    else:
        raise ValueError("Numerical audit status was not harvested")
    lines += [
        "## 代表 Nsight 窗口",
        "",
    ]
    profiles = read(output / "profiles/profiles.json")
    lines += [
        "| 捕获 | 实际 kernel 数 | 带观测完整 update，s | CUDA 活动区间并集，s |",
        "|---|---:|---:|---:|",
    ]
    for label, case in profiles["cases"].items():
        if not case["completed"]:
            lines.append(f"| {label} | 未验收 | 未验收 | 未验收 |")
            continue
        row = case["cuda"]
        lines.append(
            f"| {label} | {row['cuda_kernel_count']:,} | {row['host_update_seconds']:.2f} | "
            f"{row['kernel_copy_memset_union_seconds_in_update']:.2f} |")
    for label in ("worker0-m1", "worker0-m4"):
        case = profiles["cases"][label]
        if not case["completed"]:
            continue
        row = case["cuda"]
        window = row["representative_window"]
        phases = [
            s for s in row["host_scopes"]
            if window["start_seconds"] <= s["start_seconds"] < window["end_seconds"]]
        lines += [
            "",
            f"### {label}：第一个 learner microbatch",
            "",
            f"捕获 `fixed_bank_update` 的 {window['start_seconds']:.3f}–"
            f"{window['end_seconds']:.3f} s，"
            "下表是主机 NVTX 范围，CUDA 完成时刻另见实际 kernel 时间线。",
            "",
            "| 阶段 | 主机范围，ms |",
            "|---|---:|",
        ]
        for phase in phases:
            lines.append(
                f"| {phase['name']} | "
                f"{(phase['end_seconds'] - phase['start_seconds']) * 1000:.2f} |")
        lines += ["", f"![实际时间线](profiles/{label}-timeline.png)"]
    lines += [
        "",
        "Student FP32/PyTorch fallback 和 checkpoint 重算产生的细碎发射，"
        "应结合 CUDA launch correlation 与 NVTX 观察；"
        "Teacher BF16/FLA 与 Student 的路径不同。"
        "显式 Full-KL 范围不包含全部 vocabulary projection / entropy 成本，"
        "这些还位于 forward 中。Reference 是关闭 adapter 后的监控 forward。",
        "",
        "Kernel 通过 runtime correlation ID 同时约束进程关联，"
        "autograd 跨线程发射保留 cross_thread_temporal 标记。"
        "CUDA 区间并集是活动覆盖，不能当作 SM occupancy 或 HBM 利用率。"
        "带观测窗口时间不参与无观测吞吐 CI；"
        "单个 bank 的 trace 也不能冒充 32 bank 的分布。",
        "",
        "microbatch 从 1 提至 4，使 32 条序列的 learner 循环从 32 次降到 8 次，"
        "optimizer 更新次数保持一次。活跃 logits/activation 和 padding 随 batch 增长；"
        "单份 FP32 response logits 按 4×B×Tresponse×248,320 bytes 估算；"
        "这是 response 范围的张量规模，完整 forward 还投影 prompt 位置，"
        "不能当成总 activation 或进程显存。"
        "group_size、prompts_per_step 控制轨迹预算，rollout_batch 控制生成批次；"
        "max_new_tokens 只是上限，实际 EOS 长度和截断率必须同时查看。"
        "本次只变 microbatch，没有从这些机制推断其他参数的未测收益。",
        "",
        "观测方法与复算练习：[OBSERVATION_GUIDE.md](OBSERVATION_GUIDE.md)。"
        "包含代表窗口的阅读顺序、CUDA correlation 边界、token loss 权重、"
        "response logits 规模公式和五项已保存证据练习。",
        "",
        "## 数据、执行与复现",
        "",
        f"ESCI revision `{manifest['source']['revision']}`；排除旧 "
        f"{manifest['pilot_query_groups_excluded']:,} 个 query group 和 "
        f"{manifest['pilot_inputs_excluded']:,} 个 input。新 split 之间 query/input overlap=0，"
        f"train/test 仍共享 {manifest['product_overlap_train_test']} 个商品 ID。",
        "搜索与分类使用独立官方标签，GSM8K 和指令是有限任务族；"
        "数据规模增加不自动扩大业务外推边界。",
        "",
        "执行路径保持 Student FP32/PyTorch DeltaNet、Teacher BF16/FLA、LoRA r=16、"
        "Full reverse KL、token aggregation、lr=2e-5、clip=1、"
        "8 prompts×4 completions/update、rollout batch=4、max_new_tokens=128。"
        "BF16 部署评测与 FP32 训练分别记录；没有同时修改精度或融合 kernel。",
        "",
        "CPU 验证：完整 suite 日志中的 50 项测试通过，但执行工具因 "
        "`/proc/.../task/.../comm` 的 sandbox 写限制返回非零，"
        "不计作完整 shell 验证成功。后续终态、评测覆盖、生成轨迹、token ledger "
        "与 runtime 合同的 5 项检查独立退出 0；语法与 diff 检查通过。"
        "这些 CPU 检查与本报告的实际 GPU 产物验收分别记录。",
        "",
        "协议：[PROTOCOL.md](PROTOCOL.md)。统计入口："
        "[analyze_expanded.py](../../scripts/analyze_expanded.py)、"
        "[expanded_statistics.py](../../servedistill/expanded_statistics.py)。",
        "",
        "```bash",
        "python scripts/analyze_expanded.py \\",
        "  --performance reports/expanded/raw/performance-v1 \\",
        "  --training reports/expanded/raw/training-v1 \\",
        "  --data data_expanded --output reports/expanded",
        "python scripts/plot_expanded.py --statistics reports/expanded \\",
        "  --training reports/expanded/raw/training-v1",
        "python scripts/write_expanded_report.py --statistics reports/expanded --data data_expanded",
        "```",
        "",
        "原始输出、checkpoint、trace 在本地 raw/ 和持久化存储保留；"
        "公开证据排除生成文本、内部资源标识与绝对路径。"
        "性能原始计时、训练 steps、task metrics 和文件 hashes 支持独立复核。",
        "",
        "## 成本边界",
        "",
        "| 分配 | 调度至释放 GPUh |",
        "|---|---:|",
    ]
    for row in costs["jobs"]:
        lines.append(f"| {row['experiment']}（{row['gpu_count']} 卡） | {row['scheduled_gpu_hours']:.4f} |")
    lines += [
        f"| 扩展合计 | {costs['scheduled_gpu_hours']:.4f} |",
        f"| 原研究 + 原观测 + 扩展 | {costs['project_total_gpu_hours']:.4f} |",
        "",
        "计入环境准备、形状编译、评估、等待和释放；脚本名义上限 56 GPUh 不含平台开销。"
        "未提供单价，货币成本为 GPUh × 实际单价。"
        "没有从训练 trace 推导线上 P95、SM occupancy、HBM 带宽或生产收益。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(output / "REPORT.md"), "joint_noninferiority": joint}))


if __name__ == "__main__":
    main()
