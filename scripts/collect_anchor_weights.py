"""Collect the complete paired weight extension before any joint DEV selection."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from scripts.collect_anchor_development import allocation_cost, collect, query, verify_arm
from scripts.run_anchor_weight_development import variant_config
from servedistill.advantage_contracts import file_digest
from servedistill.data import load_rows
from servedistill.runtime import write_json


def load_protocols(base_path: Path, extension_path: Path) -> tuple[dict, dict]:
    base = json.loads(base_path.read_text())
    extension = json.loads(extension_path.read_text())
    if (file_digest(base_path) != extension["base_protocol_sha256"]
            or base["scope"] != "development_only" or base["holdout_files_packaged"]
            or base["holdout_observed_in_this_round"]
            or extension["scope"] != "development_only" or extension["holdout_observed"]
            or extension["holdout_files_packaged"] or extension["seeds"] != base["seeds"]
            or len(set(base["seeds"])) != 4
            or extension["quality_gates"] != base["quality_gates"]
            or extension["development_candidates"] != ["anchor01", "aopd_anchor", "anchor10"]
            or extension["variants"] != {
                "anchor01": {"anchor_ce_weight": 0.1}, "anchor10": {"anchor_ce_weight": 1.0}}
            or base["methods"]["aopd_anchor"]["anchor_ce_weight"] != 0.5
            or extension["selection_checkpoint"] != "fixed_final_update_320"
            or extension["selection_precision"] != "bf16"
            or extension["other_training_changes"] is not None):
        raise ValueError("Frozen paired DEV protocol binding changed")
    return base, extension


def evidence_names(steps: int = 320) -> list[str]:
    names = ["worker-completed.json", "environment.json", "base-environment.json", "task-events.json"]
    names += ["study/" + name for name in (
        "completed.json", "binding.json", "execution.json", "config.json", "training.log",
        "evaluated-adapter.json", "final-dev.jsonl", "final-dev.summary.json",
        "train/config.json", "train/completed.json", "train/schedule.json",
        "train/contract.json", "train/metrics.jsonl", "train/rollouts.jsonl")]
    names += [f"study/train/checkpoint-{steps:04d}/" + name for name in (
        "adapter_model.safetensors", "adapter_config.json", "state.json")]
    return names


def verify_weight_seed(root: Path, base: dict, extension: dict, extension_hash: str,
                       variant: str, seed: int, training: list[dict]) -> dict:
    study = root / "study"
    completed = json.loads((study / "completed.json").read_text())
    binding = json.loads((study / "binding.json").read_text())
    execution = json.loads((study / "execution.json").read_text())
    worker = json.loads((root / "worker-completed.json").read_text())
    evaluated = json.loads((study / "evaluated-adapter.json").read_text())
    if (not worker["completed"] or not worker["assets_verified"]
            or not worker["tasks"] or any(t["returncode"] for t in worker["tasks"])
            or worker["tasks"] != json.loads((root / "task-events.json").read_text())
            or not completed["completed"] or completed["scope"] != "development_only"
            or completed["holdout_observed"] or completed["quality_advantage_supported"]
            or completed["seed"] != seed or completed["variant"] != variant
            or completed["extension_sha256"] != extension_hash
            or completed["evaluated_adapter"] != evaluated
            or binding["scope"] != "development_only" or binding["holdout_observed"]
            or binding["seed"] != seed or binding["variant"] != variant
            or binding["base_protocol_sha256"] != extension["base_protocol_sha256"]
            or binding["extension_sha256"] != extension_hash or not binding["fresh_adam"]
            or binding["common_initial_adapter"] != base["initial_adapter_sha256"]
            or execution["returncode"] != 0 or execution["ended"] < execution["started"]):
        raise ValueError("Closed weight DEV scope or execution binding changed")
    spec = dataclasses.asdict(variant_config(base, extension, variant, seed))
    return verify_arm(study, spec, binding["prompt_schedule_sha256"], training)


def verify_collection(root: Path, base: dict, extension: dict, extension_hash: str) -> dict:
    audit = json.loads((root / "collection-audit.json").read_text())
    required = {f"{variant}/seed{seed}/{name}"
                for variant in extension["variants"] for seed in base["seeds"]
                for name in evidence_names(base["methods"]["aopd_anchor"]["steps"])}
    records = audit["files"]
    if (not audit["verified"] or not audit["all_runs_terminal_successful"]
            or audit["scope"] != "development_only" or audit["holdout_observed"]
            or audit["extension_sha256"] != extension_hash
            or audit["base_protocol_sha256"] != extension["base_protocol_sha256"]
            or {r["path"] for r in records} != required or len(records) != len(required)):
        raise ValueError("Complete eight-run weight evidence is required")
    for record in records:
        path = root / record["path"]
        if file_digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Weight evidence changed after collection")
    cost = json.loads((root / "cost.json").read_text())
    if (file_digest(root / "cost.json") != audit["cost_sha256"]
            or not cost["verified"] or not cost["all_trials_retained"]):
        raise ValueError("Verified weight allocation cost is required")
    return audit


def main():
    parser = argparse.ArgumentParser()
    for key in ("protocol", "extension", "submissions", "training", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--hdfs-root", required=True)
    parser.add_argument("--cli", default=os.environ.get("MERLIN_CLI", "merlin-cli"))
    parser.add_argument("--control-plane", default="cn")
    args = parser.parse_args()
    base, extension = load_protocols(args.protocol, args.extension)
    jobs = json.loads(args.submissions.read_text())
    pairs = {(v, s) for v in extension["variants"] for s in base["seeds"]}
    if (len(jobs) != len(pairs) or len({j["sid"] for j in jobs}) != len(pairs)
            or {(j["variant"], j["seed"]) for j in jobs} != pairs
            or file_digest(args.training) != base["train_sha256"]):
        raise ValueError("Complete paired submissions or training inputs changed")
    cli = [args.cli, "--control-plane", args.control_plane, "job-v2"]
    states = {}
    for job in jobs:
        response = query(cli, ["runs", "get"], {"sid": job["sid"]})
        data = response["data"]
        roles = data["resource_config"]["arnold_resource_config"]["roles"]
        if sum(r["num"] * r["gpu"] for r in roles) != 1:
            raise ValueError("Single-GPU weight allocation changed")
        if data["status"] not in ("done", "success"):
            raise ValueError("All eight weight runs must close successfully before collection")
        states[job["sid"]] = response
    exits = query(cli, ["run-trials", "exit-info"], {
        "trials_list": {"filter": {"job_run_sid_any": list(states)}}})
    cost = allocation_cost(exits["pods"]["data"], exits["trials"]["data"], set(states))
    args.output.mkdir(parents=True, exist_ok=True)
    # Raw platform responses may contain private fields; never publish this directory.
    write_json(args.output / "terminal-runs.json", states)
    write_json(args.output / "terminal-exit-info.json", exits)
    training = load_rows(args.training)
    extension_hash = file_digest(args.extension)
    records, checks = [], {}
    for job in jobs:
        relative = f"{job['variant']}/seed{job['seed']}"
        remote = args.hdfs_root.rstrip("/") + "/anchor-weight-development-v2/" + relative
        root = args.output / relative
        for name in evidence_names(base["methods"]["aopd_anchor"]["steps"]):
            record = collect(remote + "/" + name, root / name)
            records.append({"path": relative + "/" + name, **record})
        checks[relative] = verify_weight_seed(
            root, base, extension, extension_hash, job["variant"], job["seed"], training)
        print(json.dumps({"arm": relative, "collected_and_bound": True}), flush=True)
    write_json(args.output / "cost.json", cost)
    write_json(args.output / "collection-audit.json", {
        "verified": True, "all_runs_terminal_successful": True, "files": records,
        "base_protocol_sha256": extension["base_protocol_sha256"],
        "extension_sha256": extension_hash, "scope": "development_only",
        "holdout_observed": False, "adapter_checks": checks,
        "cost_sha256": file_digest(args.output / "cost.json"),
        "boundary": "All eight final adapters collected; joint DEV selection remains separate.",
    })
    verify_collection(args.output, base, extension, extension_hash)


if __name__ == "__main__":
    main()
