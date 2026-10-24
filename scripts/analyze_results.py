"""Build public task statistics and training curves from collected evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from servedistill.data import load_rows
from servedistill.statistics import bootstrap_metric, compare, metric

DOMAINS = ("search", "relevance", "substitute", "math", "instruction")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def scores(rows):
    domains = {
        domain: metric([row for row in rows if row["domain"] == domain])
        for domain in DOMAINS
    }
    return {
        "domains": domains, "business_macro": float(np.mean([domains[d] for d in DOMAINS[:3]])),
        "general_macro": float(np.mean([domains[d] for d in DOMAINS[3:]])),
        "json_valid": float(np.mean([r["json_valid"] for r in rows])),
        "truncated": float(np.mean([r["truncated"] for r in rows])),
        "independent_tasks": len(rows),
    }


def paired_methods(before, after):
    pairs = []
    for previous, current in zip(before, after):
        compare(previous, [current], repetitions=1)
        pairs.append((previous, {row["sample_id"]: row for row in current}))
    rng = np.random.default_rng(20261024)
    result = {}
    for domain in DOMAINS:
        before_rows = [[row for row in previous if row["domain"] == domain]
                       for previous, _ in pairs]
        after_rows = [[current[row["sample_id"]] for row in before_rows[index]]
                      for index, (_, current) in enumerate(pairs)]
        ids = [[row["sample_id"] for row in rows] for rows in before_rows]
        if ids[0] != ids[1]:
            raise ValueError("Training-seed task order differs")
        indices = rng.integers(0, len(ids[0]), (2000, len(ids[0])))
        deltas = [bootstrap_metric(current, indices) - bootstrap_metric(previous, indices)
                  for previous, current in zip(before_rows, after_rows)]
        result[domain] = {
            "before_seed_scores": [metric(rows) for rows in before_rows],
            "after_seed_scores": [metric(rows) for rows in after_rows],
            "mean_delta": float(np.mean([metric(a) - metric(b)
                                         for b, a in zip(before_rows, after_rows)])),
            "paired_task_ci95": np.percentile(np.mean(deltas, axis=0), [2.5, 97.5]).tolist(),
            "n_independent_tasks": len(ids[0]), "n_training_seeds": len(pairs),
        }
    return result


def analyze(evidence, output, selections):
    runs = {}
    for host in sorted(evidence.glob("host*")):
        for directory in host.iterdir():
            if directory.is_dir() and (directory / "train/completed.json").exists():
                runs[directory.name] = directory
    if len(runs) != 28:
        raise ValueError(f"Expected 26 primary and 2 restoration runs; found {len(runs)}")
    baseline_root = evidence / "host3"
    baseline = {}
    for name in ("baseline_2b", "baseline_4b", "baseline_9b"):
        baseline[name] = {
            split: load_rows(baseline_root / name / "final" / f"{split}.jsonl")
            for split in ("test", "ood")
        }
    result = {
        "deployment_selection": selections,
        "baselines": {name: {split: scores(rows) for split, rows in splits.items()}
                      for name, splits in baseline.items()},
        "methods": {}, "runs": {}, "serving": {},
        "statistics_scope": "2000 paired task bootstrap draws, conditional on two observed training seeds",
    }
    curves = []
    for label, path in sorted(runs.items()):
        if not (path / "pipeline-completed.json").exists():
            raise ValueError("Final evaluation or deployment merge incomplete: " + label)
        train = path / "train"
        completed = json.loads((train / "completed.json").read_text())
        metrics = load_rows(train / "metrics.jsonl")
        if len(metrics) != 80 or len({r["step"] for r in metrics}) != 80:
            raise ValueError("Incomplete or duplicated optimizer updates: " + label)
        if not completed["teacher_unchanged"] or completed["nonzero_updates"] < 1:
            raise ValueError("Training acceptance failed: " + label)
        domains = {}
        for record in metrics:
            for domain, value in record["domains"].items():
                aggregate = domains.setdefault(domain, {"prompts": 0, "response_tokens": 0})
                aggregate["prompts"] += value["prompts"]
                aggregate["response_tokens"] += value["response_tokens"]
            curves.append({"run": label, **{key: record.get(key) for key in (
                "step", "reward_mean", "loss", "rl_loss", "opd_loss", "entropy",
                "teacher_sampled_kl", "reference_sampled_kl", "effective_group_fraction",
                "gradient_norm", "parameter_delta_l2", "opd_weight", "behavior_max_error",
                "response_tokens", "json_valid", "truncation_rate", "elapsed_seconds",
                "gradient_projection_cosine")}})
        result["runs"][label] = {
            "selection": json.loads((train / "selected.json").read_text()),
            "updates": len(metrics), "nonzero_updates": completed["nonzero_updates"],
            "response_tokens": completed["response_tokens"],
            "max_behavior_error": completed["max_behavior_error"],
            "teacher_unchanged": completed["teacher_unchanged"],
            "cuda_peak_bytes": max(r["cuda_peak_bytes"] for r in metrics),
            "timing_this_process_only": completed["phase_seconds"],
            "domain_budget": domains,
            "alarms": [{"step": r["step"], "alarms": r["alarms"]} for r in metrics if r["alarms"]],
            "cold_restoration": [
                json.loads(p.read_text()) for p in train.glob("restoration-*.json")],
            "scores": {split: scores(load_rows(path / "final" / f"{split}.jsonl"))
                       for split in ("test", "ood")},
            "signals": {
                key: {f"p{q}": float(np.percentile([r[key] for r in metrics if key in r], q))
                      for q in (5, 50, 95)}
                for key in ("gradient_norm", "entropy", "effective_group_fraction",
                            "teacher_sampled_kl", "reference_sampled_kl",
                            "gradient_projection_cosine")
                if any(key in r for r in metrics)
            },
            "dynamic_events": load_rows(train / "weight_events.jsonl")
            if (train / "weight_events.jsonl").exists() else [],
        }
    methods = sorted({label.rsplit("_seed", 1)[0] for label in runs})
    for method in methods:
        labels = [f"{method}_seed{seed}" for seed in (17, 29)]
        result["methods"][method] = {}
        for split in ("test", "ood"):
            candidates = [load_rows(runs[label] / "final" / f"{split}.jsonl") for label in labels]
            paired = compare(baseline["baseline_2b"][split], candidates)
            write(output / "paired" / f"{method}-{split}.json", paired)
            result["methods"][method][split] = {
                domain: paired["slices"][domain] for domain in DOMAINS}
    result["restoration"] = {}
    for split in ("test", "ood"):
        before = [load_rows(runs[f"business_sft_seed{seed}"] / "final" / f"{split}.jsonl")
                  for seed in (17, 29)]
        after = [load_rows(runs[f"restore_general_opd_seed{seed}"] / "final" / f"{split}.jsonl")
                 for seed in (17, 29)]
        result["restoration"][split] = paired_methods(before, after)
    result["method_comparisons"] = {}
    for previous, current in (("offline_sft", "opd"), ("oracle_sft", "opd"),
                              ("grpo", "mixed_025"), ("grpo", "mixed_100"),
                              ("grpo", "mixed_400"), ("opd", "opd_bucketed"),
                              ("opd", "opd_trajectory"), ("opd", "full_kl"),
                              ("mixed_100", "dynamic"), ("mixed_100", "alternating")):
        before = [load_rows(runs[f"{previous}_seed{seed}"] / "final/test.jsonl") for seed in (17, 29)]
        after = [load_rows(runs[f"{current}_seed{seed}"] / "final/test.jsonl") for seed in (17, 29)]
        result["method_comparisons"][previous + "->" + current] = paired_methods(before, after)
    for path in (baseline_root / "serving_native_v1").glob("*/http/benchmark.json"):
        result["serving"][path.parent.parent.name] = json.loads(path.read_text())
    deployment = baseline_root / "deployment_v1"
    if not (deployment / "completed.json").exists():
        raise ValueError("Frozen deployment and cascade did not finish")
    for kind in ("best", "opd"):
        target = deployment / kind
        if not target.exists():
            continue
        for mode in ("http", "cascade", "supplement"):
            report = json.loads((target / mode / "benchmark.json").read_text())
            result["serving"][kind + "/" + mode] = report
            records = load_rows(target / mode / "requests.jsonl")
            result["serving"][kind + "/" + mode]["domain_scores"] = {
                str(concurrency): {
                    domain: metric([row for row in records if row["concurrency"] == concurrency
                                    and row["domain"] == domain])
                    for domain in sorted({row["domain"] for row in records})
                } for concurrency in (1, 4, 16)
            }
    write(output / "results.json", result)
    with (output / "training-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.evidence, args.output, json.loads(args.selection.read_text()))


if __name__ == "__main__":
    main()
