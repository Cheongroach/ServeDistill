"""Render the audited DEV rule comparison and its evidence-bounded report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from servedistill.advantage_contracts import file_digest
from servedistill.runtime import write_json


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = REPO / "reports/advantage"
    result_path = directory / "ROUTER_STABILITY_DEV.json"
    result = json.loads(result_path.read_text())
    cost = json.loads((directory / "ROUTER_STABILITY_DEV_COST.json").read_text())
    if (not result["independent_regrading_completed"]
            or result["scope"] != "exploratory_development_only" or not cost["verified"]):
        raise ValueError("Only independently audited DEV outputs can be reported")
    seeds = result["seeds"]
    labels = [*result["rules"], "offline_sft", "oracle_sft"]
    names = ["Step 320", "Step 640", "Mean 480/640", "Mean 160/320/480/640",
             "Offline SFT", "Oracle SFT"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    summaries = {
        label: [result["per_seed"][str(seed)][label]["summary"] if label in result["rules"]
                else result["controls"][str(seed)][label]["summary"] for seed in seeds]
        for label in labels}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(labels))
    offsets = np.linspace(-.16, .16, len(seeds))
    for index, (seed, color) in enumerate(zip(seeds, colors)):
        for axis, metric in zip(axes, ("macro_f1", "json_valid")):
            y = [100 * summaries[label][index]["domains"]["substitute"][metric] for label in labels]
            axis.scatter(x + offsets[index], y, s=45, color=color, label=f"seed {seed}",
                         zorder=3, edgecolor="white", linewidth=.5)
    axes[0].scatter(x, [100 * np.mean([s["domains"]["substitute"]["macro_f1"]
                                     for s in summaries[label]]) for label in labels],
                    marker="_", s=240, color="#222222", linewidth=2, label="four-seed mean")
    axes[0].set_ylabel("Substitute macro-F1 (%)")
    axes[0].set_title("Quality varies across checkpoints and seeds", loc="left", fontsize=11)
    threshold = axes[1].axhline(99, color="#555555", linestyle="--",
                               linewidth=1.2, label="99% format gate")
    axes[1].set_ylabel("Schema-valid JSON (%)")
    axes[1].set_title("Every seed must meet the format gate", loc="left", fontsize=11)
    minimum_valid = min(s["domains"]["substitute"]["json_valid"]
                        for group in summaries.values() for s in group) * 100
    axes[1].set_ylim(min(98.5, minimum_valid - .4), 100.35)
    for axis in axes:
        axis.set_xticks(x, names, rotation=28, ha="right")
        axis.set_xlim(-.5, len(labels) - .5)
        axis.grid(axis="y", color="#dddddd", linewidth=.6, zorder=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False,
                  bbox_to_anchor=(.5, 1.0))
    axes[1].legend(handles=[threshold], loc="lower right", frameon=False)
    figure.subplots_adjust(left=.065, right=.98, bottom=.27, top=.85, wspace=.25)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(figures / ("router-stability-dev." + suffix))
    plt.close(figure)
    write_json(figures / "router-stability-dev.metadata.json", {
        "source_sha256": file_digest(result_path), "scope": "DEV only",
        "points": "Observed seed scores; not confidence intervals",
        "independent_substitute_queries": 300, "paired_training_seeds": seeds,
        "format_threshold_percent": 99, "visual_inspection_completed": False,
    })
    chosen = result["selected_dev_rule"]
    eligible = result["eligible_rules_ranked_by_declared_dev_metric"]
    if chosen:
        summary = result["rules"][chosen]
        opening = (
            f"四种预先规定的 checkpoint 规则中，{len(eligible)} 种通过全部四个 seed 的绝对质量门禁。"
            f"按冻结的 DEV 业务宏指标排序，{chosen} 排名第一；相对 offline SFT 的均值差为 "
            f"{summary['mean_delta_business_vs_offline'] * 100:+.3f} 个百分点，"
            f"只有 {summary['positive_seed_pairs_vs_offline']}/4 个 seed 取得正差值。"
            "这是 DEV 选择结果，尚未建立独立确认或成本优势。"
        )
    else:
        opening = (
            "四种预先规定的 checkpoint 规则均未通过全部四个 seed 的绝对质量门禁，"
            "本轮没有可选候选。结果不能支持用 checkpoint 后处理修复质量稳定性，"
            "也不能据此断言所有后处理方法无效。"
        )
    lines = [
        "# Checkpoint 后处理的 BF16 DEV 稳定性实验", "", opening, "",
        "![同一 DEV 的质量与格式合法率](figures/router-stability-dev.png)", "",
        "图 1。每个彩色点是一个训练 seed 在同一 300 个 substitute query 上的结果；"
        "黑短线是四 seed 均值，虚线是预设格式门禁。点间差异不是置信区间，重复评测没有增加独立 query 数。", "",
        "## 比较问题与方法", "",
        "本轮检验：复用已完成的 AOPD 训练，固定 step320、固定 step640、最后两点因子均值、"
        "四点因子均值能否在所有 seed 上同时保持格式和分类质量？全部规则均预先声明，"
        "统一应用于 101、211、307、401；不存在按 seed 选 checkpoint、丢弃 seed 或新增输出 fallback。", "",
        "学生为 Qwen3.5-2B 文本组件。模型评测使用 BF16；平均产物使用已保存 FP32 LoRA 因子的算术均值，"
        "不是 BA 权重增量的均值。每个平均产物已从原始 checkpoint 独立重建并逐张量精确比较，"
        "固定产物与两个 SFT 对照的 adapter 文件与配置 SHA256 也已核对。", "",
        f"冻结 DEV 文件有 1400 行。本轮 workload 使用 search/relevance/substitute 各 300 行，"
        f"math/instruction 各前 100 行，共 {result['source_rows']} 个输入。"
        "每种 AOPD 规则只评估 substitute；其余域复用同 seed 的 frozen offline SFT 输出。"
        f"两个 SFT 对照各评估完整 workload；总计 {result['prediction_rows']:,} 条回答，"
        "来自相同 DEV 输入，不能计为新增独立样本。", "",
        "验收先检查每个 seed：search NDCG ≥80%，relevance macro-F1 ≥45%，"
        "substitute macro-F1 ≥55%，各业务域合法 JSON ≥99%；"
        "substitute 两类 recall ≥30%，relevance 四类 recall ≥20%。"
        "合法 JSON 同时要求语法、字段集合和值域正确；substitute 仅允许 yes/no。"
        "全部 seed 通过后才按三业务指标等权均值排序。DEV 多规则选择不产生确认性显著结论。", "",
        "## 所有规则与对照", "",
        "Δ 的单位为百分点；业务宏指标为 search NDCG、relevance macro-F1、substitute macro-F1 的等权均值。", "",
        "| 全局规则 | 业务宏指标均值 | Δ vs offline | Δ vs oracle | 正差值 seed vs offline | 全 seed 门禁 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, values in result["rules"].items():
        lines.append(
            f"| {label} | {100 * values['mean_business_macro']:.3f}% | "
            f"{100 * values['mean_delta_business_vs_offline']:+.3f} | "
            f"{100 * values['mean_delta_business_vs_oracle']:+.3f} | "
            f"{values['positive_seed_pairs_vs_offline']}/4 | "
            f"{'通过' if values['all_seed_absolute_gates_passed'] else '未通过'} |")
    lines += ["", "| substitute macro-F1 / JSON 合法率 | seed101 | seed211 | seed307 | seed401 |",
              "|---|---:|---:|---:|---:|"]
    for label in labels:
        cells = [f"{100*s['domains']['substitute']['macro_f1']:.3f}% / "
                 f"{100*s['domains']['substitute']['json_valid']:.3f}%" for s in summaries[label]]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    failures = [
        f"{label}/seed{seed} 的 {check['criterion']} 为 {check['value']:.4f}，"
        f"低于 {check['minimum']:.4f}"
        for label, values in result["rules"].items()
        for seed, checks in values["failed_checks_by_seed"].items() for check in checks
    ]
    lines += ["", "门禁失败项：" + ("；".join(failures) + "。" if failures else "无。"), "",
              "逐 seed 的类别 support、precision、recall、F1 与全部门禁判断保留在机读结果中。", "",
              "## 证据允许的结论", "",
              "在这批 DEV 输入上，final640、last2、all4 均未出现 substitute 值域异常，"
              "四个 seed 的 schema 合法率均为 100%。"
              f"last2 的业务宏指标仅比 final640 高 "
              f"{100 * (result['rules']['last2']['mean_business_macro'] - result['rules']['final640']['mean_business_macro']):.3f} "
              "个百分点；相对 offline SFT 的提升也没有在所有 seed 上保持同一方向。"
              "本轮没有形成值得立即投入新确认实验的稳定质量改善证据，下一步应继续在 DEV 上定位质量差异。", "",
              "本次是失败确认之后的探索性 DEV 消融。原 test/OOD 已观察，第一次确认的负结果保留；"
              "本轮输出不能替代该结果，也不能把原 test/OOD 再用作未见确认集。"
              "若要提出修订候选的优势，必须先冻结统一规则，再在未见的独立 query 上完成配对确认。", "",
              f"本轮新增分配成本为 {cost['allocated_gpu_hours']:.3f} H20·h，"
              f"保留 {cost['trial_count']} 个 trial。按 Pod start_at 至 stop_at × 分配卡数计算，"
              "不等同于付费账单或有效计算工时。没有新增训练；此前确认的 46.548 H20·h "
              "及历史开发成本单独保留，不能把本轮成本称为全项目总成本。", "",
              "- [完整独立重判结果](ROUTER_STABILITY_DEV.json)。",
              "- [所有 trial 的分配成本](ROUTER_STABILITY_DEV_COST.json)。",
              "- [精确 adapter 来源与结果采集审计](raw/router-stability-dev-v1/collection-audit.json)。",
              "- [冻结 DEV 规则](../../.ops/router-stability-dev-v1/contract.json)。",
              "- [原失败确认](CONFIRMATION_AOPD_DOMAIN_ROUTER.md)。",
              "- [重判入口](../../scripts/analyze_router_stability_dev.py)与"
              "[图表入口](../../scripts/write_router_stability_dev_report.py)。", ""]
    (directory / "ROUTER_STABILITY_DEV.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
