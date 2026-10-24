"""Standalone research figures from measured task and training evidence."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.report
    result = json.loads((root / "results.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    names = [name for name in result["methods"] if name not in ("business_sft", "restore_general_opd")]
    figure, axes = plt.subplots(1, 5, figsize=(16, 5), constrained_layout=True)
    for axis, domain in zip(axes, ("search", "relevance", "substitute", "math", "instruction")):
        rows = [result["methods"][name]["test"][domain] for name in names]
        values = np.array([row["delta"] for row in rows]) * 100
        intervals = np.array([row["paired_task_ci95"] for row in rows]) * 100
        # Draw intervals explicitly; the point can lie outside percentile CIs.
        axis.hlines(np.arange(len(names)), intervals[:, 0], intervals[:, 1], color="#718096")
        axis.scatter(values, np.arange(len(names)), s=25, color="#166e83")
        axis.axvline(0, color="#aaa", linewidth=0.8)
        axis.set_title(domain)
        axis.set_xlabel("Change vs 2B baseline (pp)")
        axis.invert_yaxis()
        axis.set_yticks(np.arange(len(names)), names if domain == "search" else [])
    figure.suptitle("Frozen test: two seeds, paired task 95% CI")
    for extension in ("png", "pdf"):
        figure.savefig(root / ("domain-deltas." + extension), dpi=180)
    plt.close(figure)

    with (root / "training-curves.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for method in ("grpo", "opd", "mixed_025", "mixed_100", "mixed_400", "dynamic"):
        rows = [row for row in curves if row["run"] == method + "_seed17"]
        for axis, key, label in zip(axes.flat,
                ("reward_mean", "entropy", "teacher_sampled_kl", "gradient_norm"),
                ("Training reward", "Entropy", "Sampled teacher log-ratio", "Gradient norm")):
            valid = [row for row in rows if row[key] not in ("", "None")]
            if valid:
                axis.plot([int(r["step"]) for r in valid], [float(r[key]) for r in valid],
                          label=method, linewidth=1, alpha=0.8)
            axis.set_xlabel("Optimizer update")
            axis.set_ylabel(label)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("Observed training signals, seed 17; sampled KL may be negative")
    for extension in ("png", "pdf"):
        figure.savefig(root / ("training-signals." + extension), dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
