"""Write final business and interview documents from frozen measured results."""
import argparse
import json
from pathlib import Path

import numpy as np


def percentage(value):
    return f"{100 * value:.2f}%"


def average_scores(result, method, split="test"):
    rows = [result["runs"][f"{method}_seed{seed}"]["scores"][split] for seed in (17, 29)]
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ("business_macro", "general_macro", "json_valid", "truncated")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    args = parser.parse_args()
    root = args.report
    result = json.loads((root / "results.json").read_text())
    cost = json.loads(args.cost.read_text())
    if not cost["all_terminal"]:
        raise ValueError("Final cost report requires terminal platform evidence")
    selected = result["deployment_selection"]
    best, opd = selected["method"], selected["opd_candidate"]["method"]
    baseline = result["baselines"]["baseline_2b"]["test"]
    teacher = result["baselines"]["baseline_9b"]["test"]
    methods = sorted(result["methods"], key=lambda name: -average_scores(result, name)["business_macro"])
    lines = [
        "# ServeDistill：业务 OPD 最终实测报告",
        "",
        "公开搜索业务拟真验证；Qwen3.5-9B → Qwen3.5-2B，4B 和专用 reranker 为参考。"
        "数据来自 Amazon ESCI 的真实公开 query、商品文本和独立人工标签。"
        "结果用于业务选型判断，不等于真实流量、成交、收入或完整通用能力收益。",
        "",
        "**本轮结论：工程与实验闭环完成，候选模型未获得直接上线资格。** "
        "dev 选出的最优路径是离线 SFT；OPD 家族候选是 full_kl。两者在最终 test 的替代品域"
        "均触发 5pp 保留门，综合提升不能覆盖该域回退。没有证明 OPD 比 SFT 更适合本业务。",
        "",
        f"26 个一阶段对照和 2 个通用恢复阶段完成，两种子、每阶段 80 次 optimizer update。"
        f"dev 冻结的综合候选为 **{selected['label']}**；OPD 候选为 **{selected['opd_candidate']['label']}**。"
        f"累计平台分配 **{cost['scheduled_gpu_hours']:.2f} H20 小时**，峰值 32 卡，包含失败、加载、编译、评测和闲置驻留。",
        "",
        f"初始 2B 的业务综合分为 {percentage(baseline['business_macro'])}，"
        f"通用子集为 {percentage(baseline['general_macro'])}；"
        f"9B 为 {percentage(teacher['business_macro'])} / {percentage(teacher['general_macro'])}。"
        "业务综合分是搜索 NDCG、相关性 macro-F1、替代品 macro-F1 的算术均值，"
        "不是在线成功率；通用是 GSM8K 数值答案与确定性指令任务的均值。",
        "",
        "## 全方法对照",
        "",
        "| 方法 | 业务综合 | 对初始变化 pp | 通用子集 | 对初始变化 pp | JSON 有效 | 训练 response tokens 两种子 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        score = average_scores(result, method)
        tokens = [result["runs"][f"{method}_seed{seed}"]["response_tokens"] for seed in (17, 29)]
        lines.append(f"| {method} | {percentage(score['business_macro'])} | "
                     f"{100*(score['business_macro']-baseline['business_macro']):+.2f} | "
                     f"{percentage(score['general_macro'])} | "
                     f"{100*(score['general_macro']-baseline['general_macro']):+.2f} | "
                     f"{percentage(score['json_valid'])} | {tokens[0]:,} / {tokens[1]:,} |")
    lines += [
        "",
        "统一匹配 80 更新与同一 prompt 序列；SFT 每 prompt 使用一条轨迹，"
        "OPD/GRPO 每 prompt 生成 4 条，生成长度也不同。因此 tokens、teacher 评分和 GPU 成本并不严格匹配，"
        "不能把固定更新数结果解释成固定算力的因果优势。动态和交替路径仍执行并计费 teacher 评分。",
        "",
        "## OPD 分域收益与不确定性",
        "",
        f"以下是 {opd} 两种子平均对初始 2B 的配对变化，95% CI 以独立任务抽样；"
        "训练随机性的覆盖限于实际两个种子。各业务/通用域 test 为 300 题，"
        "搜索全零 gain 的 query 排除 NDCG；每语言切片详见 paired/，小切片标注 small_slice。",
        "",
        "| 域 | 初始 | 最终两种子 | 变化 pp | 任务配对 95% CI pp |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain, row in result["methods"][opd]["test"].items():
        lower, upper = row["paired_task_ci95"]
        seeds = " / ".join(percentage(value) for value in row["candidate_seed_scores"])
        lines.append(f"| {domain} | {percentage(row['baseline'])} | {seeds} | "
                     f"{100*row['delta']:+.2f} | [{100*lower:+.2f}, {100*upper:+.2f}] |")
    lines += [
        "",
        "替代品资格门中 teacher 低于 student，是保留的负对照；"
        "最终回退必须结合这个 teacher 资格失败理解；不能声称所有域均有正向迁移。"
        "HTTP 使用 30题/域，与全 test 质量统计分开。公开数据可能存在预训练污染。",
        "",
        "## 方法消融",
        "",
        "| 前方法 → 后方法 | 搜索 Δpp | 相关性 Δpp | 替代品 Δpp | 数学 Δpp | 指令 Δpp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, comparison in result["method_comparisons"].items():
        values = " | ".join(f"{100*comparison[d]['mean_delta']:+.2f}"
                            for d in ("search", "relevance", "substitute", "math", "instruction"))
        lines.append(f"| {name} | {values} |")
    lines += [
        "",
        "每项完整 CI、种子原始分数保留在 results.json。比较为探索性，无多重检验修正。"
        "不能用表中最高 test 分数重新选部署配置。",
        "",
        "## 通用能力恢复",
        "",
        "先完成业务单独 SFT，用其 dev 选择的 adapter 初始化第二阶段通用 OPD；"
        "第二阶段新建 Adam，不等同于同阶段冷恢复。保留两个不同 business 起点的种子差异。",
        "",
        "| 域 | SFT 后两种子 | 通用 OPD 后两种子 | 平均变化 pp | 95% CI pp |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain, row in result["restoration"]["test"].items():
        before = " / ".join(percentage(value) for value in row["before_seed_scores"])
        after = " / ".join(percentage(value) for value in row["after_seed_scores"])
        lower, upper = row["paired_task_ci95"]
        lines.append(f"| {domain} | {before} | {after} | {100*row['mean_delta']:+.2f} | "
                     f"[{100*lower:+.2f}, {100*upper:+.2f}] |")
    lines += [
        "",
        "ood 是额外独立官方 test query 的复验，不能称为证明真实分布漂移。"
        "恢复结论必须同时查看业务损失与两个通用域，不能只挑指令正确率。"
        "业务SFT后通用均分约19.08%，初始为19.67%，遗忘幅度很小；"
        "第二阶段主要证明指令子集适配改善，没有证明严重遗忘后的全面恢复。"
        "恢复数学相对业务SFT的95%CI跨0，且缺少同预算二阶段SFT/继续训练对照，"
        "不能把额外80更新收益全部归因于OPD机制。",
        "",
        "## 更新、损失和训练验收",
        "",
        r"\(D_{\mathrm{KL}}(p_\theta\Vert q)=\sum_v p_\theta(v\mid x,y_{<t})"
        r"\log[p_\theta(v\mid x,y_{<t})/q(v\mid x,y_{<t})]\)。"
        "OPD 由 student 生成，teacher 在 student 实际前缀上评分；"
        "离线蒸馏由固定 teacher 轨迹做 CE。sampled-token 信号为 "
        r"\(\mathrm{stopgrad}[\log q(y_t\mid s_t)-\log p_{\rm old}(y_t\mid s_t)]\)，"
        "不做 group 标准化，负的单 token logratio 保留。固定状态的梯度验算通过，"
        "不宣称包含前缀分布导数的完整轨迹无偏梯度。",
        "",
        "每批 rollout/评分后更新一次参数；OPD 的信号是 token 级，GRPO 的 reward 是轨迹级。"
        "主版本是同步、single owner、lag=0，一条轨迹中不切换权重；"
        "没有异步吞吐收益证据。混合为 L_GRPO+λL_OPD；交替为更新级二选一。"
        "动态 λ 只使用 dev，限制 [0.25,4]，通用下降 3pp 时增大，业务停滞时减小。"
        "batch 混合/分桶复用相同 prompt 序列和总域预算，分桶在 10 个更新窗口内重排。"
        "token 平均与每轨迹平均分别消融。",
        "",
        f"32 项 CPU 回归通过（远程完整测试进程 exit=0）。28 个训练阶段最大 cached/recomputed 概率误差为 "
        f"{max(run['max_behavior_error'] for run in result['runs'].values()):.6g}，"
        "student FP32+PyTorch DeltaNet，teacher BF16；部署另以 BF16 合并。"
        "BF16 早期概率门失败触发拒绝更新，并未为了通过验收放宽该门。"
        "每阶段记录有限 gradient/参数、冻结 teacher、版本、response mask、有效 tokens、"
        "Adam/RNG checkpoint。teacher 结束指纹覆盖各参数前32元素，不能称为完整参数哈希。",
        "",
        "监控同时查看独立 dev、reward、entropy、reference logratio、ratio/ESS/clip、"
        "梯度、JSON/截断、组内有效率与分域 tokens。非有限值或概率契约失配立即停；"
        "联合趋势报警需要 dev 复验。单独 loss 增大或 sampled KL 为负不等于崩溃。"
        "梯度 cosine 只测末层一个参数投影，不是全模型梯度冲突。",
        "",
        "两次离线 SFT 受到监控字段缺失异常影响；seed17 从 checkpoint40 恢复 Adam/RNG，"
        "seed29 保留失败尝试后从头重跑。恢复进程 completed 的时间不能冒充80更新全程耗时。"
        "失败尝试和为 sidecar 保留的驻留时间均入总账。",
        "",
        "## 原生服务与质量成本",
        "",
        "vLLM 0.19.1 / torch 2.10 / Transformers 5.6.2；H20、BF16、原生 Qwen3.5 language-only、"
        "默认编译和 graph capture、max_model_len4096、max_new_tokens128、thinking关闭。"
        "localhost 流式 HTTP 从发请求到 DONE；TTFT 是首个非空 content，检索与外网排除。"
        "并发为持续补充的闭合客户端，150题/扫，5题 warmup。"
        "数字不代表线上端到端 SLA，单次短扫尤其不能据此承诺容量。",
        "",
        "| 路径 | 并发 | P95 ms | TTFT P95 ms | 请求/s | 正确答案/s | 成功率 | GPUh/百万有效答案 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in result["serving"].items():
        if name.endswith("supplement"):
            continue
        for concurrency, row in report["concurrency"].items():
            ttft = row.get("ttft_seconds", row.get("first_visible_answer_seconds"))
            lines.append(f"| {name} | {concurrency} | {row['latency_seconds']['p95']*1000:.1f} | "
                         f"{ttft['p95']*1000:.1f} | {row['requests_per_second']:.2f} | "
                         f"{row['effective_answers_per_second']:.2f} | {percentage(row['success_fraction'])} | "
                         f"{row['gpu_hours_per_million_effective_answers']:.2f} |")
    lines += ["", "### 跨业务补充与加载驻留", "",
              "| 路径 | 餐饮成功率 | 酒店成功率 | 到店成功率 | JSON有效 |", "|---|---:|---:|---:|---:|"]
    for kind in ("baseline", "teacher", "best", "opd"):
        report = result["serving"][kind + "/supplement"]
        domains = report["domain_scores"]["1"]
        lines.append("| " + kind + " | " + " | ".join(percentage(domains[d]) for d in ("dining", "hotel", "service")) + f" | {percentage(report['concurrency']['1']['json_valid_fraction'])} |")
    lines += ["", "这是每域30题的固定补充集，未训练和选型。ESCI要求 ranking/label/substitute，补充任务要求 action/choice。跨输出schema能力需独立校验，格式失配的0分应归因于迁移失败，不能据此判断模型算术能力。",
              "", "| 路径 | 到健康端点秒 | 引擎热态显存 (含预留KV/Graph) |", "|---|---:|---|"]
    for kind in ("baseline", "teacher", "best", "opd"):
        runtime = result["deployment_runtime"][kind]
        memory = runtime.get("gpu-memory", {}).get("after_warm_benchmark", "未记录")
        lines.append(f"| {kind} | {runtime['startup']['cold_start_seconds']:.2f} | {memory} |")
    lines += [
        "",
        "cascade 仅在协议错误、候选ID错误或截断时请求 teacher；正确标签不参与路由，"
        "合法但错误的决策不会被自动发现。JSON 验证前缓冲，表中 cascade 的首个可见答案为完成时刻。"
        "成本包含 student+teacher 两张专用 GPU 的全部测量驻留，包含闲置 teacher。"
        "人工团购/酒店/到店约束补充结果记录于 serving 的 supplement，未用于选择或训练，"
        "不能外推真实团购收益。",
        "",
        f"GPU 单价未知，用 c 元/GPUh：项目总成本约 {cost['scheduled_gpu_hours']:.2f}c 元。"
        "百万有效答案成本 = 表内 GPUh × c；不能以较短错误答案的高吞吐宣称业务经济性。"
        "摊销请求量应以完整训练/标注成本除以相同质量约束下的每请求成本节省；"
        "若 student 达不到质量门或回退后单请求成本不降，则没有正的摊销点。",
        "",
        "专用 Qwen3-Reranker-0.6B 的冻结搜索 test NDCG 0.8807，进程内候选评分 P95 102.9ms。"
        "它不提供三个域统一 JSON 决策，计时边界也不同。单一排序业务应首先验证该类任务模型，"
        "不能因为使用了 OPD 就排除更简单的可落地路线。",
        "",
        "## 业务迁移边界",
        "",
        "teacher 应来自业务资格验证通过的强模型、业务 SFT/RL checkpoint 或历史通用模型。"
        "本项目使用公开 post-trained9B与版本化提示，不是独立权重专家合版。"
        "仅有文本 API 时先做离线序列蒸馏，不能假定 logits 可用。"
        "可靠 teacher 与密集行为迁移适合 OPD；可信可验证 outcome 且 teacher 不够强时优先 GRPO；"
        "高质量标签充分时保留 SFT。新商品事实通过检索与数据建设解决。",
        "",
        "复现证据：results.json、paired/、training-curves.csv、cost-audit.json、"
        "deployment-selection.json、模型/数据契约与哈希。可视图见 domain-deltas.png 和 training-signals.png。"
        "main 保存工程与报告，manuscript 保存脱敏设计、取舍和对话摘要。",
    ]
    lines += ["", "## 部署一致性复验", "",
              "同150题子集比较BF16 HF与原生vLLM，输入hash逐例一致。逐字输出与解析后决策分开，尾部换行不会被当作语义变化。",
              "", "| 候选 | 去首尾空白文本一致 | JSON决策一致 | HF→vLLM业务综合 | HF→vLLM通用子集 |", "|---|---:|---:|---:|---:|"]
    for kind, row in result["deployment_parity"].items():
        before, after = row["hf_scores_same_subset"], row["vllm_scores_same_subset"]
        lines.append(f"| {kind} | {percentage(row['trimmed_text_agreement'])} | {percentage(row['parsed_json_decision_agreement'])} | {percentage(before['business_macro'])}→{percentage(after['business_macro'])} | {percentage(before['general_macro'])}→{percentage(after['general_macro'])} |")
    lines += ["", "这验证输入与质量的一致性边界，不证明不同kernel逐token完全一致；生产仍需更大固定评测集。"]
    teacher_cost = result["serving"]["baseline_9b"]["concurrency"]["16"]["gpu_hours_per_million_requests"]
    lines += ["", "## 实测摊销边界", "",
              r"若质量门通过，\(N_{break}=C_{train,GPUh}/(h_T-h_S)\)；h 是每请求GPUh，整个项目总账用于保守的研发成本摊销。",
              "下面保留同并发16的纯算术结果，质量不等价且最终保留门失败，生产摊销点均为不可用。",
              "", "| 路径 | GPUh/百万请求 | 忽略质量门的算术请求量 | 生产摊销点 |", "|---|---:|---:|---|"]
    for kind in ("best/http", "opd/http", "best/cascade", "opd/cascade"):
        value = result["serving"][kind]["concurrency"]["16"]["gpu_hours_per_million_requests"]
        amount = f"{cost['scheduled_gpu_hours']*1e6/(teacher_cost-value):,.0f}" if value < teacher_cost else "无正的成本节省"
        lines.append(f"| {kind} | {value:.3f} | {amount} | 不可用：质量门失败 |")
    lines += ["", f"12个平台任务均已终态，实际并发峰值{cost['peak_scheduled_gpus']}卡，首个GPU分配到最后释放窗口{cost['scheduled_window_hours']:.2f}小时。",
              f"终态标记failed/stopped的任务驻留合计{cost['failed_stopped_scheduled_gpu_hours']:.2f}GPUh；其中包含已成功的子实验与主动释放的host，不能当作失败计算浪费。",
              "对照与图像是探索性结果，数据/模型预训练污染、两个训练种子、短HTTP扫与非等token预算均是外推边界。"]
    lines += ["", "## 排序格式、OOD 与监控补充", "",
              "排序主指标把无效输出计为0，端到端提升同时包含格式修复与顺序质量变化。以下共同合法任务子集为事后分析，不能替代主指标。",
              "", "| 方法 | 排序JSON有效两种子 | 共同合法任务 | 共同合法排序 Δpp [95%CI] |", "|---|---:|---:|---:|"]
    for method in ("offline_sft", "opd", "full_kl"):
        valid = " / ".join(percentage(result["runs"][f"{method}_seed{seed}"]["scores"]["test"]["json_valid_by_domain"]["search"]) for seed in (17, 29))
        row = result["search_common_valid"][method]["slices"]["search"]
        lower, upper = row["paired_task_ci95"]
        lines.append(f"| {method} | {valid} | {row['n_independent_tasks']} | {100*row['delta']:+.2f} [{100*lower:+.2f}, {100*upper:+.2f}] |")
    lines += ["", "| 方法 | OOD业务综合 | OOD通用子集 |", "|---|---:|---:|"]
    for method in ("offline_sft", "opd", "full_kl", "grpo", "restore_general_opd"):
        score = average_scores(result, method, "ood")
        lines.append(f"| {method} | {percentage(score['business_macro'])} | {percentage(score['general_macro'])} |")
    for kind, row in result["holdout_retention_gate"].items():
        deltas = ", ".join(f"{d} {100*row['domain_delta'][d]:+.2f}pp" for d in row["regressed_domains"])
        lines.append(f"\n冻结候选 {row['label']}：最终5pp门={row['passes_point_estimate_5pp_gate']}；触发域：{deltas or '无'}。压测完成不提升其上线资格。")
    lines += ["", "纯sampled-token OPD的test业务综合低于离线SFT；full_kl略高，但dev选型仍为SFT。探索性混合损失比纯GRPO改善多个域，动态λ没有显示对固定λ=1一致的优势，交替更新在本配置下多数域更差。分桶与轨迹聚合的收益不一致，不能给出通用默认最优配置。",
              "", "| 路径 seed17 | 非零参数更新 | 概率误差最大 | gradient P95 | ESS P05 | 告警更新数 | rollout/teacher/learner秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("grpo", "opd", "full_kl", "mixed_100", "dynamic", "alternating"):
        run = result["runs"][method + "_seed17"]
        phase = run["timing_this_process_only"]
        times = "/".join(f"{phase[k]:.1f}" for k in ("rollout", "teacher", "learner"))
        lines.append(f"| {method} | {run['nonzero_updates']} | {run['max_behavior_error']:.6g} | {run['signals']['gradient_norm']['p95']:.3f} | {run['signals']['ess_fraction']['p5']:.6f} | {len(run['alarms'])} | {times} |")
    lines += ["", "同步实现ratio≈1、clip≈0是行为概率契约结果，不构成陈旧轨迹校正验证。分相为进程内时间，排除加载和闲置，不能替代平台GPUh。checkpoint恢复只验收已存adapter/Adam/RNG指纹与后续有效更新，未做整个80更新轨迹与未中断对照逐元素一致性验收。",
              "", "teacher有效headroom的回收比例=(最终两种子均分-初始)/(teacher-初始)。搜索114%，相关性39%，数学15%，指令75%；替代品teacher headroom非正，比例不适用。排序超过100%包含输出格式改善，不能直接解释成语义能力超过teacher。"]
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    opd_scores = average_scores(result, opd)
    interview = [
        "# ServeDistill 面试讲法",
        "",
        f"负责公开搜索业务的大模型到小模型蒸馏闭环，用 Qwen3.5-9B→2B，"
        "以独立 ESCI 标签校验 teacher 资格、业务提升与通用回退；实现"
        "OPD/GRPO/混合λ/交替/动态/分桶/全KL等两种子对照，28阶段各80更新。",
        f"最终 {opd} 业务综合分 {percentage(opd_scores['business_macro'])}，"
        f"相对初始变化 {100*(opd_scores['business_macro']-baseline['business_macro']):+.2f}pp；"
        f"通用子集变化 {100*(opd_scores['general_macro']-baseline['general_macro']):+.2f}pp。"
        "各域置信区间和负结果必须同时披露，不能称为真实业务收入增益。",
        "最终替代品回退触发5pp质量门，保留拒绝上线结论；离线SFT赢得dev选型，未证明OPD优于它。",
        "",
        "工程难点是行为概率契约：BF16 rollout/recompute 不一致被验收门拒绝，"
        "通过 FP32 student+PyTorch DeltaNet 保住精度，并单独核算成本；"
        "不是靠调大容忍阈值。checkpoint 保存 adapter/Adam/RNG/controller，"
        "冷恢复指纹通过；第二阶段恢复能力另开 optimizer，避免混淆两个实验。",
        "",
        f"完成原生 vLLM 合并模型、流式请求ID/usage/截断校验及无标签协议回退压测；"
        f"累计 {cost['scheduled_gpu_hours']:.2f} H20 小时。"
        "按每百万有效答案计算 GPU 成本，明确本地HTTP与线上SLA的差异。",
        "",
        "面对“为什么不用SFT/GRPO”：先看 teacher 是否有该域 headroom、标签是否够好、"
        "reward是否可靠，再引用实测消融；不用固定流程代替业务选择。"
        "面对“token级还是轨迹级”：信号与聚合是token/trajectory，optimizer每批更新，"
        "不在生成中切换权重。面对“多专家”：本轮是提示路由，独立权重专家融合尚未验证。",
        "",
        "不要声称：异步OPD吞吐提升、完整通用能力恢复、真实团购转化率、"
        "teacher全参数逐元素哈希、固定tokens/算力公平最优或已满足线上SLA。",
    ]
    (root / "INTERVIEW_NOTES.md").write_text("\n".join(interview) + "\n")


if __name__ == "__main__":
    main()
