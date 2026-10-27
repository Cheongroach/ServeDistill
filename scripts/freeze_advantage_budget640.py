"""Freeze the complete 640-update DEV selection before any independent predictions."""
import argparse
import dataclasses
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from servedistill.adapter_average import validate_postprocessing
from servedistill.advantage_contracts import file_digest, verify_data, verify_labels
from servedistill.advantage_failures import control_failure_checkpoint
from servedistill.data import canonical, load_rows
from servedistill.scheduling import Config, schedule


def select(development):
    """Reject incomplete pools and gains obtained only against a weaker SFT."""
    if (development["declared_candidates"] != 60
            or development["completed_candidates"] + development["failed_candidates"] != 60
            or not development["complete_selection_pool"]
            or not development["independent_regrading_completed"]
            or development["holdout_observed"]):
        raise ValueError("The entire declared 640-update DEV pool is required")
    failures = development["failed_controls"]
    if set(failures) - {"deep_original_full_kl_lr2e5"}:
        raise ValueError("Unexpected failed development family")
    pool = [r for values in development["rankings"].values() for r in values]
    if (len(pool) != development["completed_candidates"]
            or len({r["name"] for r in pool}) != len(pool)):
        raise ValueError("Development pool count or candidate identities differ")
    selected = {}
    for objective in ("opd", "offline_sft", "oracle_sft"):
        values = development["rankings"][objective]
        if len(values) != 18:
            raise ValueError("Each method needs both reductions, three rates and three adapters")
        if any(r["objective"] != objective or r["arm_config"]["objective"] != objective
               or r["arm_config"]["steps"] != 640 for r in values):
            raise ValueError("Development family or common update budget drift")
        grid = {(r["arm_config"]["aggregation"], r["arm_config"]["learning_rate"],
                 canonical(r["postprocessing"])) for r in values}
        expected_grid = {
            (aggregation, rate, canonical(spec))
            for aggregation in ("token", "trajectory")
            for rate in (1e-5, 2e-5, 5e-5)
            for spec in ({"kind": "final_checkpoint"},
                         {"kind": "lora_factor_mean_fp32", "steps": [480, 640]},
                         {"kind": "lora_factor_mean_fp32", "steps": [160, 320, 480, 640]})}
        if grid != expected_grid:
            raise ValueError("A method is missing a declared reduction/rate/adapter combination")
        selected[objective] = min(values, key=lambda r: (-r["selection_score"], r["name"]))
    for objective in ("offline_sft", "oracle_sft"):
        if selected["opd"]["business_macro"] <= selected[objective]["business_macro"]:
            raise ValueError("DEV candidate does not beat both strongest selected SFT controls")
    controls = [r for r in development["rankings"]["full_kl"]
                if r["source_arm"] == "deep_repaired_full_kl_lr2e5"]
    if len(controls) != 3:
        raise ValueError("Repaired Full-KL needs all declared adapters")
    selected["full_kl"] = min(controls, key=lambda r: (-r["selection_score"], r["name"]))
    for value in selected.values():
        validate_postprocessing(value["postprocessing"], 640)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError("Refusing to replace a pre-test protocol")
    subprocess.run([sys.executable, str(repo / ".ops/report_advantage_budget640.py")],
                   cwd=repo, stdout=subprocess.DEVNULL, check=True, timeout=120)
    reports = repo / "reports/advantage"
    report_path = reports / "DEVELOPMENT640.json"
    development = json.loads(report_path.read_text())
    selected = select(development)
    data = repo / "data_advantage_holdout"
    verify_data(data, confirmation=True)
    if any((reports / "raw").rglob("test.jsonl")) or any((reports / "raw").rglob("ood.jsonl")):
        raise ValueError("Independent predictions already exist; a new freeze is forbidden")
    labels = reports / "raw/repaired-v1/teacher_labels/teacher-labels.jsonl"
    labels_complete = json.loads(labels.with_name("completed.json").read_text())
    if (not labels_complete.get("completed")
            or file_digest(labels) != labels_complete["labels_sha256"]
            or labels_complete["teacher_manifest_sha256"] != development["repaired_teacher_manifest_sha256"]):
        raise ValueError("Complete unfiltered hard labels must use the repaired teacher")
    verify_labels(labels, data / "train.jsonl")
    original_name = "deep_original_full_kl_lr2e5"
    original = reports / "raw/budget640-controls-v4" / original_name
    original_config = json.loads((original / "train/config.json").read_text())
    original_provenance = json.loads((original / "start-provenance.json").read_text())
    controls_ops = repo / ".ops/advantage-budget640-controls-v4"
    original_plan = json.loads((controls_ops / "artifact-contract.json").read_text())["configs"][original_name]
    if original_config != dataclasses.asdict(Config(**original_plan)):
        raise ValueError("Original Full-KL actual configuration differs")
    failure_policy = {"objective": "full_kl", "training_messages": [
        "Joint training alarm persisted and dev confirmed degradation"]}
    if original_name in development["failed_controls"]:
        _, actual_failure = control_failure_checkpoint(
            original, original_config, failure_policy, require_weights=False)
        if actual_failure != development["failed_controls"][original_name]:
            raise ValueError("Original Full-KL failure evidence changed")
    elif not json.loads((original / "train/completed.json").read_text()).get("completed"):
        raise ValueError("Original Full-KL has neither completed nor verified failure evidence")
    common = development["common_start"]
    if any(original_provenance[key] != value for key, value in common.items()):
        raise ValueError("Original Full-KL does not share the actual student start")
    frozen_sources = {}
    for relative in original_provenance["execution_source_files"]:
        frozen_sources[relative] = file_digest(repo / relative)
    if frozen_sources != original_provenance["execution_source_files"]:
        raise ValueError("Executed original control source drift")
    for folder, digest in development["contract_sha256"].items():
        ops = repo / ".ops" / ("advantage-" + folder)
        if file_digest(ops / "artifact-contract.json") != digest:
            raise ValueError("Submitted development contract changed")
        prepared = json.loads((ops / "source-hashes.json").read_text())
        if any(prepared[relative] != digest for relative, digest in frozen_sources.items()):
            raise ValueError("Development methods executed different sources")
    # This is an audit of actual saved binary factors and optimizer/RNG identity.
    audit_path = reports / "BINARY_AUDIT640.json"
    audit = json.loads(audit_path.read_text())
    required_sources = {v["source_arm"] for values in development["rankings"].values() for v in values}
    required_sources.add(original_name)
    if not audit.get("verified") or set(audit["arms"]) != required_sources:
        raise ValueError("All 20 actual development source arms need a persisted binary audit")
    for values in development["rankings"].values():
        for value in values:
            audited = audit["arms"][value["source_arm"]]["candidates"][value["name"]]
            if (audited["predictions_sha256"] != value["predictions_sha256"]
                    or audited["completion_sha256"] != value["completion_sha256"]):
                raise ValueError("Audited candidate differs from the independently regraded DEV")
    families = {
        "candidate_opd": selected["opd"], "offline_sft": selected["offline_sft"],
        "oracle_sft": selected["oracle_sft"], "repaired_full_kl": selected["full_kl"]}
    configs, teachers, specifications = {}, {}, {}
    selection = {}
    for family, value in families.items():
        arm = reports / "raw" / value["source_folder"] / value["source_arm"]
        provenance = json.loads((arm / "start-provenance.json").read_text())
        if any(provenance[key] != digest for key, digest in common.items()):
            raise ValueError("Selected method does not share the actual student start")
        if provenance["teacher_manifest_sha256"] != development["repaired_teacher_manifest_sha256"]:
            raise ValueError("Selected repaired-teacher identity drift")
        configs[family] = value["arm_config"]
        teachers[family] = provenance["teacher_manifest_sha256"]
        specifications[family] = value["postprocessing"]
        selection[family] = value
    configs["original_full_kl"] = original_plan
    teachers["original_full_kl"] = original_provenance["teacher_manifest_sha256"]
    specifications["original_full_kl"] = {"kind": "final_checkpoint"}
    execution_hash = hashlib.sha256(json.dumps(frozen_sources, sort_keys=True).encode()).hexdigest()
    seeds = [101, 211, 307, 401]
    arm_configs, allowed, postprocessing, schedules, policies = {}, {}, {}, {}, {}
    train_rows = load_rows(data / "train.jsonl")
    for family, source_config in configs.items():
        for seed in seeds:
            label = family + "_seed" + str(seed)
            config = {**source_config, "seed": seed, "steps": 640,
                      "method_time_budget_seconds": None, "prior_method_seconds": 0.0}
            arm_configs[label] = config
            allowed[label] = {
                **common, "config_sha256": hashlib.sha256(
                    (json.dumps(config, indent=2) + "\n").encode()).hexdigest(),
                "data_manifest_sha256": file_digest(data / "manifest.json"),
                "teacher_manifest_sha256": teachers[family],
                "teacher_labels_sha256": file_digest(labels) if family == "offline_sft" else None,
                "cost_manifest_sha256": None, "execution_source_sha256": execution_hash}
            postprocessing[label] = specifications[family]
            plan = [[r["sample_id"] for r in batch] for batch in schedule(train_rows, Config(**config))]
            digest = hashlib.sha256(canonical(plan).encode()).hexdigest()
            if str(seed) in schedules and schedules[str(seed)] != digest:
                raise ValueError("Frozen paired prompt schedules differ")
            schedules[str(seed)] = digest
            if family == "original_full_kl":
                policies[label] = failure_policy
    protocol = {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "test_observed_before_freeze": False,
        "data_manifest_sha256": file_digest(data / "manifest.json"), "paired_seeds": seeds,
        "paired_prompt_schedule_sha256": schedules, "arm_configs": arm_configs,
        "allowed_arms": allowed, "arm_postprocessing": postprocessing,
        "control_failure_policy": policies, "execution_source_files": frozen_sources,
        "primary_comparison": {
            "name": "pure_opd_vs_strongest_development_selected_offline_sft",
            "baseline_arms": ["offline_sft_seed" + str(seed) for seed in seeds],
            "candidate_arms": ["candidate_opd_seed" + str(seed) for seed in seeds],
            "metric": "unweighted mean of search NDCG and relevance/substitute macro-F1"},
        "strong_sft_guard": {
            "baseline_arms": ["oracle_sft_seed" + str(seed) for seed in seeds],
            "rule": "Both strongest frozen SFT comparisons must pass on both disjoint splits"},
        "quality_gates": {
            "minimum_metric": {"search": 0.80, "relevance": 0.45, "substitute": 0.55},
            "minimum_recall": {"substitute": {"yes": 0.30, "no": 0.30},
                               "relevance": {"E": 0.20, "S": 0.20, "C": 0.20, "I": 0.20}},
            "minimum_json_valid": 0.99, "domain_noninferiority_margin": -0.03,
            "math_noninferiority_margin": -0.05, "minimum_positive_seed_pairs": 3},
        "bootstrap_repetitions": 5000, "bootstrap_seed": 202610254,
        "analysis_source_sha256": file_digest(repo / "scripts/analyze_advantage_confirmation.py"),
        "statistics_source_sha256": file_digest(repo / "servedistill/advantage_statistics.py"),
        "development_selection": selection,
        "full_development_selection_pool": development,
        "development_report_sha256": file_digest(report_path),
        "development_binary_audit_sha256": file_digest(audit_path),
        "prior_development_report_sha256": file_digest(reports / "DEVELOPMENT320_FINAL.json"),
        "development_selection_rule": "Highest 0.8 business + 0.2 finite general within each "
                                     "objective across the full declared common640 pool",
        "budget": {"updates": 640, "prompts_per_update": 10, "cost_claim_allowed": False,
                   "matching": "same updates and prompt schedule; response tokens and GPU time differ"},
        "selection": "Final full update and frozen factor rule; no holdout selection or revision",
        "replication": "Both disjoint splits must pass both frozen SFT comparisons and all quality gates",
        "boundary": "Four paired seeds condition the query CI. All attempts and complete allocated "
                    "GPU time remain in the cost ledger. A separate pre-test equal-time auxiliary "
                    "experiment is required for any cost superiority claim."}
    if args.check_only:
        print(json.dumps({"ready": True, "arms": len(allowed), "frozen": False,
                          "selected": {k: v["name"] for k, v in selection.items()}}))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"frozen": True, "protocol_sha256": file_digest(args.output),
                      "arms": len(allowed), "paired_seeds": seeds, "test_observed": False}))


if __name__ == "__main__":
    main()
