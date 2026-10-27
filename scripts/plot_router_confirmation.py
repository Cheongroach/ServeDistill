"""Render the frozen confirmation result and separately scoped DEV diagnosis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    reports = repo / "reports/advantage"
    result_path = reports / "CONFIRMATION_AOPD_DOMAIN_ROUTER.json"
    audit_path = reports / "CONFIRMATION_AOPD_DOMAIN_ROUTER_AUDIT.json"
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    output = reports / "figures"
    colors = {"test": "#245E9B", "ood": "#C66B28"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    comparisons = [
        ("primary_test", "Offline SFT · test", "test"),
        ("disjoint_replication", "Offline SFT · OOD", "ood"),
        ("oracle_primary_test", "Oracle SFT · test", "test"),
        ("oracle_disjoint_replication", "Oracle SFT · OOD", "ood"),
    ]
    for y, (key, label, split) in enumerate(comparisons):
        macro = result[key]["business_macro"]
        point = 100 * macro["delta"]
        low, high = np.array(macro["paired_query_ci95"]) * 100
        axes[0].plot([low, high], [y, y], color=colors[split], lw=2)
        axes[0].scatter(point, y, color=colors[split], s=38, zorder=3)
        axes[0].text(high + 0.05, y, f"{point:+.2f}", va="center", fontsize=9)
    axes[0].set_yticks(range(4), [x[1] for x in comparisons])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#666666", lw=1, ls="--")
    axes[0].set_xlim(-1.25, 2.9)
    axes[0].set_xlabel("Business macro difference (percentage points)")
    axes[0].set_title("a  Frozen paired-query 95% intervals", loc="left")
    for key, split, offset in (
        ("primary_test", "test", -0.07),
        ("disjoint_replication", "ood", 0.07),
    ):
        deltas = np.array(result[key]["domains"]["substitute"]["paired_seed_deltas"]) * 100
        axes[1].scatter(np.arange(4) + offset, deltas, color=colors[split],
                        label=split, s=38)
    axes[1].axhline(0, color="#666666", lw=1, ls="--")
    axes[1].set_xticks(range(4), result["paired_seeds"])
    axes[1].set_xlabel("Paired training seed")
    axes[1].set_ylabel("Substitute macro-F1 difference (pp)")
    axes[1].set_title("b  Router minus offline SFT by seed", loc="left")
    axes[1].legend(frameon=False)
    fig.tight_layout(pad=1.4)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"router-confirmation.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    palette = ["#245E9B", "#568548", "#C66B28", "#9E3B64"]
    for seed, color in zip(result["paired_seeds"], palette):
        rows = [r for r in audit["dev_diagnostic"] if r["seed"] == seed]
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [100 * r["substitute_macro_f1"] for r in rows],
                     marker="o", color=color, label=str(seed))
        axes[1].plot(steps, [100 * r["substitute_json_valid"] for r in rows],
                     marker="o", color=color, label=str(seed))
    axes[0].set_ylim(59, 75)
    axes[0].set_ylabel("Substitute macro-F1 (%)")
    axes[0].set_title("a  Checkpoint behavior varies across seeds", loc="left")
    axes[0].legend(title="Training seed", frameon=False, ncol=2, fontsize=9)
    axes[1].axhline(99, color="#666666", ls="--", label="99% format threshold")
    axes[1].set_ylim(97.6, 100.4)
    axes[1].set_ylabel("Schema-valid JSON (%)")
    axes[1].set_title("b  Seed 401 format failure was visible on DEV", loc="left")
    axes[1].legend(frameon=False, loc="lower left", fontsize=9)
    for ax in axes:
        ax.axvline(320, color="#999999", lw=1, ls=":")
        ax.set_xticks([0, 160, 320, 480, 640])
        ax.set_xlabel("Completed training updates")
    fig.tight_layout(pad=1.4)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"router-confirmation-dev.{suffix}", dpi=180)
    plt.close(fig)
    metadata = {
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "primary_interval_scope": "Independent queries, conditional on four observed paired seeds",
        "dev_scope": "FP32 training DEV; 300 substitute queries, not BF16 final selection",
        "frozen_quality_advantage_supported": result["quality_advantage_supported"],
    }
    (output / "router-confirmation.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
