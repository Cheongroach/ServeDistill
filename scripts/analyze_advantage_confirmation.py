"""Regrade frozen independent predictions before computing paired quality claims."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from servedistill.advantage_contracts import file_digest, verify_data
from servedistill.advantage_statistics import paired_sft_quality
from servedistill.advantage_failures import control_failure_checkpoint
from servedistill.adapter_average import verify_average, validate_postprocessing
from servedistill.data import grade, load_rows
from servedistill.runtime import summarize, write_json
from servedistill.scheduling import Config
from servedistill.time_budget import completed_checkpoint


def verified_arm(root: Path, label: str, expected: dict, protocol_hash: str,
                 planned_config: dict, source: dict, split: str,
                 failure_policy: dict | None = None,
                 postprocessing: dict | None = None) -> tuple[list[dict] | None, dict]:
    path = root / label
    provenance = json.loads((path / "start-provenance.json").read_text())
    if provenance["phase"] != "confirmation" or provenance["label"] != label:
        raise ValueError("Evidence is not the frozen confirmation arm: " + label)
    if provenance.get("test_used_for_selection") or provenance["protocol_sha256"] != protocol_hash:
        raise ValueError("Arm used a different protocol or test selection: " + label)
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Arm fingerprints differ from the frozen contract: " + label)
    config_path = path / "train/config.json"
    config = json.loads(config_path.read_text())
    if config != dataclasses.asdict(Config(**planned_config)) or config["seed"] != provenance["seed"]:
        raise ValueError("Actual training configuration differs from frozen input: " + label)
    outcome_path = path / "control-outcome.json"
    failure_record = None
    if outcome_path.exists():
        if failure_policy is None:
            raise ValueError("Failure is not allowed for this frozen arm: " + label)
        outcome = json.loads(outcome_path.read_text())
        checkpoint, failure_record = control_failure_checkpoint(
            path, config, failure_policy, require_weights=False)
        if (outcome["phase"] != "confirmation" or outcome["label"] != label
                or outcome["protocol_sha256"] != protocol_hash
                or outcome.get("test_used_for_selection")
                or outcome["status"] != "training_failed"
                or outcome["checkpoint_evaluated"] != (checkpoint is not None)):
            raise ValueError("Failed-control outcome differs from its frozen policy: " + label)
        for key, value in failure_record.items():
            if key == "checkpoint_files_sha256":
                if any(outcome[key].get(name) != digest for name, digest in value.items()):
                    raise ValueError("Failed-control saved checkpoint evidence changed: " + label)
            elif outcome.get(key) != value:
                raise ValueError("Failed-control evidence changed: " + label + "/" + key)
        if checkpoint is None:
            if (path / "final").exists():
                raise ValueError("Unscored failed control contains predictions: " + label)
            return None, {**failure_record, "seed": provenance["seed"], "rows": 0,
                          "control_outcome_sha256": file_digest(outcome_path)}
    else:
        complete = json.loads((path / "completed.json").read_text())
        train_complete = json.loads((path / "train/completed.json").read_text())
        checkpoint = completed_checkpoint(config, train_complete)
        if (not complete.get("completed") or complete["phase"] != "confirmation"
                or complete["checkpoint"] != checkpoint or complete.get("test_used_for_selection")):
            raise ValueError("Confirmation did not evaluate the final completed update: " + label)
    if postprocessing is not None:
        validate_postprocessing(postprocessing, config["steps"])
        evaluated = path / "train" / checkpoint
        if failure_record is None:
            if complete.get("postprocessing") != postprocessing:
                raise ValueError("Evaluated postprocessing differs from frozen rule: " + label)
            if postprocessing["kind"] == "lora_factor_mean_fp32":
                evaluated = path / "averaged-adapter"
                verify_average(path / "train", evaluated, postprocessing, config["steps"])
            if complete["evaluated_adapter_file_sha256"] != file_digest(evaluated / "adapter_model.safetensors"):
                raise ValueError("Completed evaluated adapter identity changed: " + label)
        identity = json.loads((path / "final/evaluation-adapter.json").read_text())
        if (identity["adapter_file_sha256"] != file_digest(evaluated / "adapter_model.safetensors")
                or identity["adapter_config_sha256"] != file_digest(evaluated / "adapter_config.json")):
            raise ValueError("Final predictions used a different adapter: " + label)
    predictions_path = path / "final" / (split + ".jsonl")
    rows = load_rows(predictions_path)
    if len(rows) != len(source) or {row["sample_id"] for row in rows} != set(source):
        raise ValueError("Confirmation predictions do not cover the frozen split: " + label)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated confirmation predictions: " + label)
    for record in rows:
        sample = source[record["sample_id"]]
        if sample["input_sha256"] != record["input_sha256"] or sample["domain"] != record["domain"]:
            raise ValueError("Confirmation input changed: " + label + "/" + record["sample_id"])
        actual = grade(sample, record["text"])
        if any(record.get(key) != value for key, value in actual.items()):
            raise ValueError("Independent regrading differs: " + label + "/" + record["sample_id"])
    summary = summarize(rows)
    saved_summary = json.loads((path / "final" / (split + ".summary.json")).read_text())
    if summary != saved_summary:
        raise ValueError("Independent summary differs from saved predictions: " + label)
    return rows, {
        "predictions_sha256": file_digest(predictions_path),
        "provenance_sha256": file_digest(path / "start-provenance.json"),
        "completion_sha256": file_digest(path / "completed.json") if failure_record is None else None,
        "control_outcome_sha256": file_digest(outcome_path) if failure_record else None,
        "training_failure": failure_record,
        "training_config_sha256": file_digest(config_path),
        "final_checkpoint": checkpoint, "seed": provenance["seed"],
        "rows": len(rows), "summary": summary,
    }


def analyze(root: Path, data: Path, protocol_path: Path, cost_path: Path | None = None):
    protocol = json.loads(protocol_path.read_text())
    repository = Path(__file__).resolve().parents[1]
    for key, relative in (("analysis_source_sha256", "scripts/analyze_advantage_confirmation.py"),
                          ("statistics_source_sha256", "servedistill/advantage_statistics.py")):
        if protocol.get(key) != file_digest(repository / relative):
            raise ValueError("Frozen confirmation analysis source changed: " + relative)
    for relative, expected in protocol["execution_source_files"].items():
        if file_digest(repository / relative) != expected:
            raise ValueError("Frozen execution/grading source changed: " + relative)
    if protocol.get("test_observed_before_freeze") is not False:
        raise ValueError("Protocol does not establish a pre-test freeze")
    if protocol["data_manifest_sha256"] != file_digest(data / "manifest.json"):
        raise ValueError("Confirmation data manifest differs from its protocol")
    verification = verify_data(data, confirmation=True)
    primary = protocol["primary_comparison"]
    seeds = protocol["paired_seeds"]
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("Protocol must freeze four distinct paired training seeds")
    left, right = primary["baseline_arms"], primary["candidate_arms"]
    if len(left) != 4 or len(right) != 4 or set(left) & set(right):
        raise ValueError("Protocol primary comparison does not contain four distinct pairs")
    oracle = protocol["strong_sft_guard"]["baseline_arms"]
    if (len(oracle) != 4 or len(set(oracle)) != 4 or set(oracle) & set(left + right)
            or [protocol["arm_configs"][label]["seed"] for label in oracle] != seeds
            or any(protocol["arm_configs"][label]["objective"] != "oracle_sft" for label in oracle)):
        raise ValueError("Strong SFT guard must bind four distinct paired oracle controls")
    policies = protocol.get("control_failure_policy", {})
    if set(policies) & set(left + right):
        raise ValueError("Primary candidates and SFT baselines must all complete successfully")
    if any(label not in protocol["allowed_arms"]
           or not label.startswith("original_full_kl_seed") for label in policies):
        raise ValueError("Only original Full-KL controls may use failure reporting")
    protocol_hash = file_digest(protocol_path)
    results, evidence, oracle_guards, comparison_support = {}, {}, {}, {}
    for split in ("test", "ood"):
        source = {row["sample_id"]: row for row in load_rows(data / (split + ".jsonl"))}
        predictions = {}
        # Retain all controls, including failed superiority comparisons.
        for label, expected in protocol["allowed_arms"].items():
            rows, record = verified_arm(
                root, label, expected, protocol_hash, protocol["arm_configs"][label], source, split, policies.get(label),
                protocol["arm_postprocessing"][label])
            if rows is not None:
                predictions[label] = rows
            evidence.setdefault(label, {})[split] = record
        for labels in (left, right):
            observed = [evidence[label][split]["seed"] for label in labels]
            if observed != seeds:
                raise ValueError("Primary pairing differs from the frozen seed order")
        compared = paired_sft_quality(
            [predictions[label] for label in left],
            [predictions[label] for label in oracle],
            [predictions[label] for label in right], protocol["quality_gates"],
            repetitions=protocol["bootstrap_repetitions"],
            seed=protocol["bootstrap_seed"] + (split == "ood"))
        results[split] = compared["primary"]
        oracle_guards[split] = compared["oracle_guard"]
        comparison_support[split] = compared["quality_advantage_supported"]
    costs = json.loads(cost_path.read_text()) if cost_path else None
    cost_verified = bool(costs and costs["all_terminal_costs_verified"])
    return {
        "protocol_sha256": protocol_hash, "data_verification": verification,
        "primary_comparison": primary, "paired_seeds": seeds,
        "strong_sft_guard": protocol["strong_sft_guard"],
        "oracle_primary_test": oracle_guards["test"],
        "oracle_disjoint_replication": oracle_guards["ood"],
        "independent_regrading_completed": True, "arm_evidence": evidence,
        "primary_test": results["test"], "disjoint_replication": results["ood"],
        "quality_advantage_supported": (
            comparison_support["test"] and comparison_support["ood"]),
        "terminal_cost_audit_verified": cost_verified,
        "cost_audit_sha256": file_digest(cost_path) if cost_path else None,
        "cost_advantage_supported": False,
        "boundary": "Quality comparison is conditional on four frozen training seeds. "
                    "The second split is a disjoint query replication, not distribution shift. "
                    "Terminal accounting alone does not establish matched-cost superiority.",
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "data", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cost-audit", type=Path)
    args = parser.parse_args()
    report = analyze(args.root, args.data, args.protocol, args.cost_audit)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "independent_regrading_completed", "quality_advantage_supported",
        "terminal_cost_audit_verified", "cost_advantage_supported")}))


if __name__ == "__main__":
    main()
