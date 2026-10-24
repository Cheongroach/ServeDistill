"""Freeze update-matched ablations before reading final test results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/matrix"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    args = parser.parse_args()
    if args.steps < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive updates and distinct seeds required")
    args.output.mkdir(parents=True, exist_ok=True)
    methods = [
        ("oracle_sft", {"objective": "oracle_sft"}),
        ("offline_sft", {"objective": "offline_sft"}),
        ("grpo", {"objective": "grpo"}),
        ("opd", {"objective": "opd"}),
        ("mixed_025", {"objective": "mixed", "opd_weight": 0.25}),
        ("mixed_100", {"objective": "mixed", "opd_weight": 1.0}),
        ("mixed_400", {"objective": "mixed", "opd_weight": 4.0}),
        ("alternating", {"objective": "alternating"}),
        ("dynamic", {"objective": "dynamic"}),
        ("opd_bucketed", {"objective": "opd", "bucketed": True}),
        ("opd_trajectory", {"objective": "opd", "aggregation": "trajectory"}),
        ("full_kl", {"objective": "full_kl"}),
        ("business_sft", {"objective": "oracle_sft", "business_only": True}),
        ("restore_general_opd", {"objective": "opd", "general_only": True}),
    ]
    rows = []
    for seed in args.seeds:
        for name, override in methods:
            config = {
                "business_domains": ["search", "relevance", "substitute"],
                "seed": seed, "steps": args.steps, "prompts_per_step": 5,
                "group_size": 4, "microbatch": 1, "rollout_batch": 4,
                "max_new_tokens": 128, "learning_rate": 2e-5,
                "opd_weight": 1.0, "dev_every": 20, "dev_per_domain": 20,
                "behavior_tolerance": 0.15, **override,
            }
            label = f"{name}_seed{seed}"
            path = args.output / (label + ".json")
            if path.exists():
                raise FileExistsError("Frozen experiment configuration already exists: " + str(path))
            path.write_text(json.dumps(config, indent=2) + "\n")
            rows.append({"label": label, "config": path.name,
                         "requires_teacher_labels": name == "offline_sft",
                         "requires_initial_adapter": name == "restore_general_opd"})
    manifest = {
        "models": {"student": "Qwen/Qwen3.5-2B", "teacher": "Qwen/Qwen3.5-9B"},
        "matching": "fixed optimizer updates and prompt schedule; tokens and GPUh reported separately",
        "seeds": args.seeds, "steps": args.steps, "experiments": rows,
        "final_test_used_for_selection": False,
    }
    (args.output / "matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"experiments": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
