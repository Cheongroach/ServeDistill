"""Publication figures from validated expansion statistics and actual training rows."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from servedistill.data import load_rows


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    output = args.statistics
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    perf = json.loads((output / "performance.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, m in enumerate((1, 2, 4, 8)):
        row = perf["comparisons"][str(m)]
        value, ci = row["paired_throughput_ratio"], row["paired_bank_ci95"]
        axes[0].plot([index, index], ci, color="#166534")
        axes[0].scatter([index], [value], color="#166534", s=35)
        axes[1].bar(index, row["maximum_peak_allocated_gib"], color="#2563eb", width=.55)
    axes[0].axhline(1, color="#9ca3af", linestyle="--")
    axes[0].set(ylabel="Paired response-token throughput / m1",
                title=f"{perf['n_independent_banks']} frozen banks; 95% bank bootstrap CI")
    axes[1].set(ylabel="Maximum per-step allocated memory (GiB)",
                title="Observed memory; excludes non-PyTorch allocation")
    for ax in axes:
        ax.set(xticks=range(4), xticklabels=["m1", "m2", "m4", "m8"], xlabel="Microbatch")
        ax.grid(axis="y", alpha=.2)
    save(fig, output, "performance-memory")
    domains = ("search", "relevance", "substitute", "math", "instruction")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split, title in zip(axes, ("test", "ood"), ("Test", "Independent query replication")):
        quality = json.loads((output / ("quality-" + split + ".json")).read_text())
        for index, domain in enumerate(domains):
            row = quality["slices"][domain]
            value = row["delta"] * 100
            ci = np.array(row["paired_query_ci95"]) * 100
            ax.plot(ci, [index, index], color="#166534")
            ax.scatter([value], [index], color="#166534", s=35)
            ax.scatter(np.array(row["paired_seed_deltas"]) * 100, [index + .14] * 4,
                       s=13, alpha=.6, color="#475569")
        ax.axvline(0, color="#9ca3af", linestyle="--")
        ax.axvline(-5, color="#b91c1c", linestyle=":")
        ax.set(title=title, xlabel="m4 - paired m1 (percentage points)")
        ax.grid(axis="x", alpha=.2)
    axes[0].set(yticks=range(5), yticklabels=domains)
    axes[0].invert_yaxis()
    fig.text(.1, -.02, "Intervals: paired queries conditional on four seed pairs. "
             "Small dots: individual seeds. Red line: predeclared -5 pp margin.", fontsize=9)
    save(fig, output, "quality-deltas")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for seed in (17, 29, 43, 71):
        for m, color in ((1, "#2563eb"), (4, "#166534")):
            rows = load_rows(args.training / f"full_kl_m{m}_seed{seed}/train/metrics.jsonl")
            steps = [r["step"] for r in rows]
            axes[0].plot(steps, [r["elapsed_seconds"] / 60 for r in rows],
                         color=color, alpha=.6, label=f"m{m}" if seed == 17 else None)
            axes[1].plot(steps, [r["cumulative_response_tokens"] for r in rows],
                         color=color, alpha=.6)
    axes[0].set(ylabel="Elapsed training minutes incl. dev/compilation", xlabel="Optimizer update")
    axes[1].set(ylabel="Actual cumulative response tokens", xlabel="Optimizer update")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend()
    save(fig, output, "training-work")
    print(json.dumps({"figures": ["performance-memory", "quality-deltas", "training-work"]}))


if __name__ == "__main__":
    main()
