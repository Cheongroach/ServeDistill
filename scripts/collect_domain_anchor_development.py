"""Bind every domain candidate to actual updates and complete allocation evidence."""
import argparse
import dataclasses
import json
import os
from pathlib import Path

from scripts.collect_anchor_development import allocation_cost, collect, query, verify_arm, verify_seed
from scripts.collect_anchor_weights import evidence_names
from scripts.run_domain_anchor_development import load_protocol, variant_config
from servedistill.advantage_contracts import file_digest
from servedistill.data import load_rows
from servedistill.runtime import write_json


def verify_seed_candidate(root, base, protocol, protocol_hash, variant, seed, training):
    study = root / "study"
    completion = json.loads((study / "completed.json").read_text())
    binding = json.loads((study / "binding.json").read_text())
    worker = json.loads((root / "worker-completed.json").read_text())
    execution = json.loads((study / "execution.json").read_text())
    evaluated = json.loads((study / "evaluated-adapter.json").read_text())
    if (not worker["completed"] or not worker["assets_verified"] or not worker["tasks"]
            or any(t["returncode"] for t in worker["tasks"])
            or worker["tasks"] != json.loads((root / "task-events.json").read_text())
            or not completion["completed"] or completion["scope"] != "development_only"
            or completion["holdout_observed"] or completion["quality_advantage_supported"]
            or completion["seed"] != seed or completion["variant"] != variant
            or completion["protocol_sha256"] != protocol_hash
            or completion["evaluated_adapter"] != evaluated
            or binding["scope"] != "development_only" or binding["holdout_observed"]
            or binding["seed"] != seed or binding["variant"] != variant
            or binding["protocol_sha256"] != protocol_hash
            or binding["base_protocol_sha256"] != protocol["base_protocol_sha256"]
            or not binding["fresh_adam"]
            or binding["common_initial_adapter"] != base["initial_adapter_sha256"]
            or execution["returncode"] != 0 or execution["ended"] < execution["started"]):
        raise ValueError("Domain candidate execution binding changed")
    return verify_arm(study, dataclasses.asdict(variant_config(base, protocol, variant, seed)),
                      binding["prompt_schedule_sha256"], training)


def verify_collection(root, base, protocol, protocol_hash):
    audit = json.loads((root / "collection-audit.json").read_text())
    required = {f"{v}/seed{s}/{n}" for v in protocol["variants"] for s in protocol["seeds"]
                for n in evidence_names()}
    if (not audit["verified"] or not audit["all_runs_terminal_successful"]
            or audit["scope"] != "development_only" or audit["holdout_observed"]
            or audit["protocol_sha256"] != protocol_hash
            or {r["path"] for r in audit["files"]} != required
            or len(audit["files"]) != len(required)):
        raise ValueError("Complete eight-run domain evidence required")
    for record in audit["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or file_digest(path) != record["sha256"]:
            raise ValueError("Domain evidence changed after collection")
    cost = json.loads((root / "cost.json").read_text())
    if (not cost["verified"] or not cost["all_trials_retained"]
            or file_digest(root / "cost.json") != audit["cost_sha256"]):
        raise ValueError("Complete domain allocation cost required")
    return audit


def verify_historical_controls(root, base, protocol, training):
    """Recheck historical controls against their original config and full update history."""
    audit_path = root / "collection-audit.json"
    if file_digest(audit_path) != protocol["reference_collection_sha256"]:
        raise ValueError("Frozen historical control audit changed")
    audit = json.loads(audit_path.read_text())
    if (not audit["verified"] or not audit["all_runs_terminal_successful"]
            or audit["protocol_sha256"] != protocol["base_protocol_sha256"]):
        raise ValueError("Historical controls are incomplete")
    for record in audit["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or file_digest(path) != record["sha256"]:
            raise ValueError("Historical source evidence changed")
    if file_digest(root / "cost.json") != audit["cost_sha256"]:
        raise ValueError("Historical cost changed")
    return {str(seed): verify_seed(root / f"seed{seed}", base,
                                  protocol["base_protocol_sha256"], seed, training)
            for seed in base["seeds"]}


def main():
    parser = argparse.ArgumentParser()
    for key in ("base-protocol", "protocol", "submissions", "training", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--hdfs-root", required=True)
    parser.add_argument("--cli", default=os.environ.get("MERLIN_CLI", "merlin-cli"))
    args = parser.parse_args()
    base, protocol = load_protocol(args.base_protocol, args.protocol)
    jobs = json.loads(args.submissions.read_text())
    required = {(v, s) for v in protocol["variants"] for s in protocol["seeds"]}
    if (len(jobs) != len(required) or len({j["sid"] for j in jobs}) != len(required)
            or {(j["variant"], j["seed"]) for j in jobs} != required
            or file_digest(args.training) != base["train_sha256"]):
        raise ValueError("Complete paired submissions or training input changed")
    cli = [args.cli, "--control-plane", "cn", "job-v2"]
    states = {}
    for job in jobs:
        response = query(cli, ["runs", "get"], {"sid": job["sid"]})
        data = response["data"]
        roles = data["resource_config"]["arnold_resource_config"]["roles"]
        if sum(r["num"] * r["gpu"] for r in roles) != 1:
            raise ValueError("Single-GPU domain allocation changed")
        states[job["sid"]] = response
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "terminal-runs.json", states)
    exits = query(cli, ["run-trials", "exit-info"], {
        "trials_list": {"filter": {"job_run_sid_any": list(states)}}})
    write_json(args.output / "terminal-exit-info.json", exits)
    cost = allocation_cost(exits["pods"]["data"], exits["trials"]["data"], set(states))
    write_json(args.output / "cost.json", cost)
    if any(r["data"]["status"] not in ("done", "success") for r in states.values()):
        raise ValueError("All eight domain runs must close successfully before selection; "
                         "terminal failures and allocation costs retained")
    training = load_rows(args.training)
    protocol_hash = file_digest(args.protocol)
    records, checks = [], {}
    for job in jobs:
        relative = f"{job['variant']}/seed{job['seed']}"
        remote = args.hdfs_root.rstrip("/") + "/domain-anchor-development-v1/" + relative
        root = args.output / relative
        for name in evidence_names():
            record = collect(remote + "/" + name, root / name)
            records.append({"path": relative + "/" + name, **record})
        checks[relative] = verify_seed_candidate(root, base, protocol, protocol_hash,
                                                job["variant"], job["seed"], training)
        print(json.dumps({"arm": relative, "collected_and_bound": True}), flush=True)
    write_json(args.output / "collection-audit.json", {
        "verified": True, "all_runs_terminal_successful": True, "scope": "development_only",
        "holdout_observed": False, "protocol_sha256": protocol_hash,
        "files": records, "adapter_checks": checks, "cost_sha256": file_digest(args.output / "cost.json"),
    })
    verify_collection(args.output, base, protocol, protocol_hash)


if __name__ == "__main__":
    main()
