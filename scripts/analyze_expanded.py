"""Validate terminal expansion artifacts and compute frozen paired estimators."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from servedistill.data import grade, load_rows
from servedistill.expanded_statistics import performance, quality
from servedistill.runtime import write_json
from servedistill.statistics import metric


SEEDS = (17, 29, 43, 71)


def classification_metrics(rows):
    """Describe class collapse separately from the frozen paired endpoint."""
    task = rows[0]["task"]
    labels = ("E", "S", "C", "I") if task == "esci_classification" else ("yes", "no")
    support = Counter(row["target_class"] for row in rows)
    predictions = Counter(row["predicted_class"] for row in rows)
    classes = {}
    for label in labels:
        tp = sum(row["target_class"] == label and row["predicted_class"] == label for row in rows)
        fp = predictions[label] - tp
        fn = support[label] - tp
        classes[label] = {
            "support": support[label], "predicted": predictions[label], "true_positive": tp,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        }
    majority = support.most_common(1)[0][0]
    return {
        "queries": len(rows), "target_counts": dict(support), "prediction_counts": dict(predictions),
        "per_class": classes, "macro_f1": metric(rows),
        "accuracy": sum(row["target_class"] == row["predicted_class"] for row in rows) / len(rows),
        "majority_class": majority,
        "majority_constant_macro_f1": metric([{**row, "predicted_class": majority} for row in rows]),
        "constant_prediction": len(predictions) == 1,
    }


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_completed(path):
    value = read(path)
    if not value.get("completed"):
        raise ValueError("Incomplete terminal artifact: " + str(path))
    return value


def require_execution(path):
    value = read(path)
    if value["returncode"] != 0 or value["end_unix_seconds"] <= value["start_unix_seconds"]:
        raise ValueError("Failed execution artifact: " + str(path))
    return value


def export_rows(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return {
        "file": path.name, "records": len(rows),
        "sha256": digest(path),
    }


def attach_source(rows, source):
    group = {row["sample_id"]: row for row in source}
    if len(rows) != len(group) or {r["sample_id"] for r in rows} != set(group):
        raise ValueError("Frozen evaluation does not cover its full split")
    for row in rows:
        original = group[row["sample_id"]]
        if row["input_sha256"] != original["input_sha256"]:
            raise ValueError("Frozen input hash changed")
        expected = grade(original, row["text"])
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Independent grading does not reproduce evaluation record")
        if "query_group_sha256" in original:
            row["query_group_sha256"] = original["query_group_sha256"]
    return rows


def validate_training_work(metrics, rollouts, schedule, completed, source_ids,
                           updates=240, prompts=8, group_size=4):
    """Reconcile actual generated work with the frozen schedule and token ledger."""
    steps = list(range(1, updates + 1))
    if ([row["step"] for row in metrics] != steps
            or [row["step"] for row in rollouts] != steps
            or len(schedule) != updates):
        raise ValueError("Incomplete training work ledger")
    cumulative, trajectories = 0, 0
    for step, batch, metric_row, rollout in zip(steps, schedule, metrics, rollouts):
        if len(batch) != prompts or not set(batch).issubset(source_ids):
            raise ValueError("Training schedule differs from the frozen prompt pool")
        expected_ids = [identity for identity in batch for _ in range(group_size)]
        if (rollout["sample_ids"] != expected_ids
                or len(rollout["texts"]) != len(expected_ids)
                or len(rollout["rewards"]) != len(expected_ids)
                or rollout["version"] != step - 1):
            raise ValueError("Generated trajectories differ from the scheduled work")
        tokens = metric_row["response_tokens"]
        if not isinstance(tokens, int) or tokens <= 0 or metric_row["parameter_delta_l2"] <= 0:
            raise ValueError("Training work has no effective tokens or parameter update")
        cumulative += tokens
        if metric_row["cumulative_response_tokens"] != cumulative:
            raise ValueError("Cumulative response token ledger changed")
        trajectories += len(expected_ids)
    if completed["response_tokens"] != cumulative:
        raise ValueError("Completed response tokens differ from the work ledger")
    return {"generated_trajectories": trajectories, "response_tokens": cumulative}


def require_runtime(path, precision):
    """Keep execution paths and token events fixed while excluding load duration."""
    value = read(path)
    if (value["precision"] != precision or value["enable_thinking"]
            or value["architecture"] != "qwen3_5_text"
            or value["student_vocab"] != 248320
            or any(value["loading"].get(key) for key in
                   ("missing_keys", "unexpected_keys", "mismatched_keys"))):
        raise ValueError("Runtime/model contract changed: " + str(path))
    return {key: item for key, item in value.items() if key != "load_seconds"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read(args.data / "manifest.json")
    source = {}
    for split in ("train", "dev", "test", "ood"):
        path = args.data / (split + ".jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["splits"][split]["sha256"]:
            raise ValueError("Data split digest changed: " + split)
        source[split] = load_rows(path)
    for root in (args.performance, args.training):
        finished = require_completed(root / "worker-completed.json")
        if not finished["assets_verified"] or any(t["returncode"] != 0 for t in finished["tasks"]):
            raise ValueError("Outer worker tasks or verified asset contract failed")
    executed_source = read(args.performance / "expansion-source-hashes.json")
    if executed_source != read(args.training / "expansion-source-hashes.json"):
        raise ValueError("Performance and training used different staged sources")
    write_json(args.output / "execution-source-hashes.json", executed_source)
    published_root = Path(__file__).resolve().parents[1]
    source_comparison = []
    for name, executed_hash in sorted(executed_source.items()):
        published_hash = digest(published_root / name)
        critical = (
            name.startswith(("servedistill/", "configs/expanded/", "data_expanded/"))
            or name in {
                "scripts/worker.py", "scripts/profile_worker.py",
                "scripts/setup_expanded_worker.py", "scripts/run_expanded_arm.py",
                "scripts/run_expanded_performance.py", "scripts/benchmark_microbatch.py",
            })
        if critical and published_hash != executed_hash:
            raise ValueError("Published execution-critical source differs: " + name)
        source_comparison.append({
            "file": name, "execution_sha256": executed_hash, "published_sha256": published_hash,
            "execution_critical": critical, "identical": published_hash == executed_hash,
        })
    write_json(args.output / "source-comparison.json", {
        "files": source_comparison, "all_execution_critical_sources_identical": True,
        "boundary": "Post-execution analysis/plot, test and protocol edits may differ; hashes retained",
    })
    timings, contracts, parents, native, frozen = [], [], [], [], []
    training_runtime, final_runtime = None, require_runtime(
        args.performance / "parent-final/contract.json", "bf16")
    for worker in range(2):
        root = args.performance / ("performance" + str(worker))
        require_completed(root / "worker-completed.json")
        require_execution(root / "benchmark-execution.json")
        parent = read(root / "parent-restoration.json")
        if not parent["passed"] or parent["step"] != 40:
            raise ValueError("Performance parent was not restored")
        parents.append(parent)
        current = require_runtime(root / "runtime-contract.json", "float32_torch")
        if training_runtime is not None and current != training_runtime:
            raise ValueError("Performance runtime paths differ")
        training_runtime = current
        timings.extend(load_rows(root / "measurements.jsonl"))
        native.extend({**r, "worker_index": worker}
                      for r in load_rows(root / "native-numerics.jsonl"))
        frozen.extend({**r, "worker_index": worker}
                      for r in load_rows(root / "frozen-teacher-numerics.jsonl"))
        for bank in range(16):
            contracts.append({**read(root / f"bank-{bank:02d}-contract.json"),
                              "worker_index": worker, "bank": bank})
    expected_native = {(w, b, m) for w in range(2) for b in range(16) for m in (1, 2, 4, 8)}
    if (len(native) != len(expected_native)
            or {(r["worker_index"], r["bank"], r["microbatch"]) for r in native} != expected_native
            or any(not math.isfinite(r[key]) or r[key] < 0 for r in native for key in
                   ("gradient_relative_l2_postclip", "update_relative_l2"))):
        raise ValueError("Incomplete or nonfinite native numerical comparisons")
    if (len(frozen) != 4 or {(r["worker_index"], r["bank"]) for r in frozen}
            != {(w, b) for w in range(2) for b in range(2)}
            or any(r["sequences"] != 4 or not math.isfinite(r["gradient_relative_l2"])
                   for r in frozen)):
        raise ValueError("Incomplete fixed-Teacher numerical comparisons")
    native_summary = {
        str(m): {
            key: {"median": median(r[key] for r in native if r["microbatch"] == m),
                  "maximum": max(r[key] for r in native if r["microbatch"] == m)}
            for key in ("gradient_relative_l2_postclip", "update_relative_l2")
        } for m in (2, 4, 8)
    }
    prompt_ids = set()
    for bank in contracts:
        ids = bank["sample_ids"]
        if len(ids) != 32 or len(set(ids)) != 8:
            raise ValueError("Each bank must have eight prompts and four completions")
        if prompt_ids.intersection(ids):
            raise ValueError("Performance banks share prompts")
        prompt_ids.update(ids)
    bank_lengths = {
        (b["worker_index"], b["bank"]): sum(b["prompt_token_lengths"]) / len(b["prompt_token_lengths"])
        for b in contracts}
    bank_contracts = {(b["worker_index"], b["bank"]): b for b in contracts}
    probability_tolerance = read(args.training / "full_kl_m1_seed17/train/config.json")["behavior_tolerance"]
    for row in timings:
        key = (row["worker_index"], row["bank"])
        row["mean_prompt_token_length"] = bank_lengths[key]
        bank = bank_contracts[key]
        if (row["effective_response_tokens"] != bank["effective_tokens"]
                or row["response_width"] != bank["response_width"]):
            raise ValueError("Timing record differs from frozen bank contract")
        if (not math.isfinite(row["behavior_max_error"])
                or row["behavior_max_error"] > probability_tolerance
                or not math.isfinite(row["loss"])
                or row["parameter_delta_l2"] <= 0):
            raise ValueError("Performance numerical/probability contract failed")
    parent_keys = ("adapter_sha256", "optimizer_sha256", "rng_sha256")
    signature = {key: parents[0][key] for key in parent_keys}
    if any({key: parent[key] for key in parent_keys} != signature for parent in parents):
        raise ValueError("Performance parents differ")
    records, secondary = {}, {}
    scheduled_ids = set()
    schedule_counts = {}
    work_counts = {}
    training_ids = {row["sample_id"] for row in source["train"]}
    for seed in SEEDS:
        schedules = []
        for m in (1, 4):
            label = f"full_kl_m{m}_seed{seed}"
            root = args.training / label
            completed = require_completed(root / "completed.json")
            if completed["primary_checkpoint"] != "checkpoint-0240" or completed["test_used_for_selection"]:
                raise ValueError("Primary evaluation checkpoint changed")
            train = require_completed(root / "train/completed.json")
            if (require_runtime(root / "train/contract.json", "float32_torch") != training_runtime
                    or require_runtime(root / "final/contract.json", "bf16") != final_runtime):
                raise ValueError("Training or final evaluation runtime paths differ")
            metrics = load_rows(root / "train/metrics.jsonl")
            if train["steps"] != 240 or [r["step"] for r in metrics] != list(range(1, 241)):
                raise ValueError("Incomplete 240-update training arm")
            if not train["teacher_unchanged"] or train["nonzero_updates"] != 240:
                raise ValueError("Training parameter/freeze contract failed")
            numerical_fields = ("loss", "behavior_max_error", "gradient_norm", "parameter_delta_l2")
            if any(not math.isfinite(r[key]) for r in metrics for key in numerical_fields):
                raise ValueError("Nonfinite training numerical record")
            if any(r["behavior_max_error"] > r["behavior_tolerance"] for r in metrics):
                raise ValueError("Training probability contract failed")
            fork = read(root / "train/fork-provenance.json")
            if not fork["parent_restoration"]["passed"] or fork["rng_seed"] != seed:
                raise ValueError("Fork seed/restoration mismatch")
            if {key: fork["parent_restoration"][key] for key in parent_keys} != signature:
                raise ValueError("Training parent differs from performance parent")
            schedules.append(read(root / "train/schedule.json"))
            work_counts[label] = validate_training_work(
                metrics, load_rows(root / "train/rollouts.jsonl"),
                schedules[-1], train, training_ids)
            ids = [i for batch in schedules[-1] for i in batch]
            if len(ids) != 1920:
                raise ValueError("Prompt schedule budget changed")
            scheduled_ids.update(ids)
            schedule_counts[label] = {
                "prompt_visits": len(ids), "unique_training_prompts": len(set(ids)),
                "schedule_sha256": hashlib.sha256(
                    (root / "train/schedule.json").read_bytes()).hexdigest()}
            records[label] = {
                "completed": train,
                "execution": {stage: require_execution(root / (stage + "-execution.json"))
                              for stage in ("train", "final")},
                "metrics": metrics,
                "evaluation": {
                    split: attach_source(load_rows(root / "final" / (split + ".jsonl")), source[split])
                    for split in ("test", "ood")},
            }
            selected = train["selected"]["path"]
            if completed["dev_selected_secondary"] != selected:
                raise ValueError("Dev-selected checkpoint provenance differs")
            selected_rows = records[label]["evaluation"]
            additional = 0
            if selected != completed["primary_checkpoint"]:
                require_execution(root / "dev-selected-final-execution.json")
                if require_runtime(root / "selected-final/contract.json", "bf16") != final_runtime:
                    raise ValueError("Secondary evaluation runtime differs")
                selected_rows = {
                    split: attach_source(load_rows(root / "selected-final" / (split + ".jsonl")), source[split])
                    for split in ("test", "ood")}
                additional = sum(len(rows) for rows in selected_rows.values())
            secondary[label] = {
                "checkpoint": selected, "step": train["selected"]["step"],
                "additional_evaluation_outputs": additional,
                "scores": {
                    split: {
                        domain: metric([r for r in rows if r["domain"] == domain])
                        for domain in sorted({r["domain"] for r in rows})}
                    for split, rows in selected_rows.items()},
            }
        if schedules[0] != schedules[1]:
            raise ValueError("Seed-paired prompt schedules differ")
    timing_result = performance(timings)
    bank_times = {
        m: np.array([bank["median_seconds"][m] for bank in timing_result["banks"]])
        for m in (4, 8)}
    sampled_banks = np.random.default_rng(20261025).integers(
        0, len(timing_result["banks"]),
        (timing_result["bootstrap_repetitions"], len(timing_result["banks"])))
    m8_vs_m4_samples = (bank_times[4][sampled_banks].sum(axis=1)
                       / bank_times[8][sampled_banks].sum(axis=1))
    timing_result["exploratory_m8_vs_m4"] = {
        "paired_throughput_ratio": float(bank_times[4].sum() / bank_times[8].sum()),
        "paired_bank_ci95": np.percentile(m8_vs_m4_samples, [2.5, 97.5]).tolist(),
        "boundary": "Exploratory additional comparison; primary m4 versus m1 remains frozen",
    }
    write_json(args.output / "performance.json", timing_result)
    for split in ("test", "ood"):
        result = quality(
            [records[f"full_kl_m1_seed{s}"]["evaluation"][split] for s in SEEDS],
            [records[f"full_kl_m4_seed{s}"]["evaluation"][split] for s in SEEDS])
        write_json(args.output / ("quality-" + split + ".json"), result)
    parent_scores, classification = {}, {}
    for split in ("test", "ood"):
        rows = attach_source(load_rows(args.performance / "parent-final" / (split + ".jsonl")), source[split])
        parent_scores[split] = {
            domain: metric([r for r in rows if r["domain"] == domain])
            for domain in sorted({r["domain"] for r in rows})}
        groups = {"parent_update40": rows, **{
            label: value["evaluation"][split] for label, value in records.items()}}
        classification[split] = {
            label: {
                domain: classification_metrics([r for r in group if r["domain"] == domain])
                for domain in ("relevance", "substitute")}
            for label, group in groups.items()}
    write_json(args.output / "parent-scores.json", parent_scores)
    write_json(args.output / "classification-audit.json", {
        "splits": classification,
        "boundary": "Descriptive class support/predictions; frozen noninferiority criteria unchanged",
    })
    write_json(args.output / "secondary-dev-selected.json", {
        "arms": secondary,
        "additional_evaluation_outputs": sum(
            row["additional_evaluation_outputs"] for row in secondary.values()),
        "boundary": "Dev-only selection, descriptive secondary results; primary remains fixed update240",
    })
    write_json(args.output / "runtime-contracts.json", {
        "training_and_performance": training_runtime, "final_evaluation": final_runtime,
        "boundary": "Student path recorded by runtime; Teacher batch kernels are not fingerprinted here",
    })
    # Public machine-readable task evidence excludes generated text and internal paths.
    evidence = args.output / "task-metrics.jsonl.gz"
    with gzip.open(evidence, "wt") as stream:
        for label, value in records.items():
            for split, rows in value["evaluation"].items():
                for row in rows:
                    stream.write(json.dumps({"arm": label, "split": split,
                                             **{k: v for k, v in row.items() if k != "text"}}) + "\n")
    write_json(args.output / "validation.json", {
        "completed": True, "seeds": list(SEEDS), "arms": len(records),
        "optimizer_updates": 1920,
        "response_tokens": sum(value["completed"]["response_tokens"] for value in records.values()),
        "generated_training_trajectories": sum(
            value["generated_trajectories"] for value in work_counts.values()),
        "unique_training_prompts_across_seed_pairs": len(scheduled_ids),
        "training_schedule_counts": schedule_counts,
        "independent_performance_prompts": len(prompt_ids),
        "frozen_trajectory_contracts": contracts, "common_parent": signature,
        "native_numerics": native, "frozen_teacher_numerics": frozen,
        "native_numerics_summary": native_summary,
        "probability_tolerance": probability_tolerance,
        "maximum_observed_performance_probability_error": max(r["behavior_max_error"] for r in timings),
        "maximum_observed_training_probability_error": max(
            r["behavior_max_error"] for value in records.values() for r in value["metrics"]),
        "teacher_freeze_fingerprint_scope": "First 32 flattened elements of every named Teacher parameter",
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
    })
    write_json(args.output / "training-summary.json", {
        label: {
            "completed": value["completed"], "execution": value["execution"],
            "generated_trajectories": work_counts[label]["generated_trajectories"],
            "mean_response_tokens_per_trajectory": (
                work_counts[label]["response_tokens"]
                / work_counts[label]["generated_trajectories"]),
            "mean_trajectory_truncation_rate": sum(
                row["truncation_rate"] for row in value["metrics"]) / len(value["metrics"]),
        }
        for label, value in records.items()})
    public_files = [
        export_rows(args.output / "performance-measurements.jsonl.gz", timings),
        export_rows(args.output / "training-steps.jsonl.gz", [
            {"arm": label, **row} for label, value in records.items() for row in value["metrics"]]),
        {
            "file": evidence.name, "records": sum(
                len(rows) for value in records.values() for rows in value["evaluation"].values()),
            "sha256": digest(evidence),
        },
    ]
    write_json(args.output / "evidence-manifest.json", {
        "files": public_files,
        "boundary": "Repetitions and seeds do not increase independent bank/query counts",
        "original_fp32_cosine_valid": False,
        "original_fp32_cosine_reason": "Self-comparison exceeded 1; raw fields retained but excluded from claims",
    })
    print(json.dumps({"completed": True, "arms": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
