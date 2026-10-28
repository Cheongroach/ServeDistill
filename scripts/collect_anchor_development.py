"""Collect every closed DEV arm with immutable adapter, schedule and allocation evidence."""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from servedistill.advantage_contracts import file_digest
from servedistill.runtime import write_json
from servedistill.scheduling import Config, schedule, update_learning_rate
from servedistill.data import load_rows


def query(cli: list[str], command: list[str], request: dict) -> dict:
    result = json.loads(subprocess.check_output(
        [*cli, *command, "--json", json.dumps(request)], timeout=90))
    if result.get("ResponseBase", {}).get("code", 0):
        raise ValueError("Platform query failed")
    return result


def collect(remote: str, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".collecting")
    with temporary.open("wb") as stream:
        subprocess.run(["hdfs", "dfs", "-cat", remote], stdout=stream, check=True,
                       timeout=240,
                       env={**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true"})
    record = {"sha256": file_digest(temporary), "bytes": temporary.stat().st_size}
    if target.exists():
        if file_digest(target) != record["sha256"]:
            raise ValueError("Existing evidence differs from the source: " + str(target))
        temporary.unlink()
    else:
        temporary.replace(target)
    return record


def allocation_cost(pods: list[dict], trials: list[dict], run_ids: set[str]) -> dict:
    if (not trials or not pods
            or {t["job_run_sid"] for t in trials} != run_ids
            or {p["job_run_sid"] for p in pods} != run_ids
            or {p["trial_sid"] for p in pods} != {t["sid"] for t in trials}
            or len({p["sid"] for p in pods}) != len(pods)):
        raise ValueError("Every run, trial and unique Pod must be retained")
    records = []
    for pod in pods:
        scheduled, stop = pod.get("schedule_at"), pod.get("stop_at")
        before_allocation = (not scheduled and not pod.get("start_at")
                             and not pod.get("container_id")
                             and pod["exit_status"] == -80002)
        seconds = 0.0
        if not before_allocation:
            if not scheduled or not stop or pod["gpuv"] != "NVIDIA-H20":
                raise ValueError("A closed H20 scheduling interval is required")
            seconds = (
                datetime.datetime.fromisoformat(stop.replace("Z", "+00:00"))
                - datetime.datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
            ).total_seconds()
            if seconds < 0:
                raise ValueError("Negative allocation interval")
        records.append({
            "job_run_sid": pod["job_run_sid"], "trial_sid": pod["trial_sid"],
            "trial_pod_sid": pod["sid"], "exit_status": pod["exit_status"],
            "schedule_at": scheduled, "start_at": pod.get("start_at"), "stop_at": stop,
            "gpu_count": 0 if before_allocation else 1,
            "allocation_seconds": seconds, "gpu_hours": seconds / 3600,
        })
    return {
        "verified": True, "all_trials_retained": True, "records": records,
        "allocated_gpu_hours": sum(r["gpu_hours"] for r in records),
        "cost_advantage_supported": False,
        "boundary": "Scheduled-to-stop Pod intervals; includes preparation, every DEV arm "
                    "and evaluation. Not billing or useful compute. Prior teacher repair, "
                    "labels and common initialization are separate historical costs.",
    }


def verify_arm(arm: Path, spec: dict, schedule_hash: str, training: list[dict]) -> dict:
    """Bind the evaluated final adapter to the declared complete training schedule."""
    config = Config(**spec)
    expected = [[r["sample_id"] for r in batch] for batch in schedule(training, config)]
    actual = json.loads((arm / "train/schedule.json").read_text())
    if (actual != expected or schedule_hash != hashlib.sha256(
            json.dumps(actual).encode()).hexdigest()
            or dataclasses.asdict(Config(**json.loads((arm / "config.json").read_text())))
               != dataclasses.asdict(config)
            or dataclasses.asdict(Config(**json.loads((arm / "train/config.json").read_text())))
               != dataclasses.asdict(config)):
        raise ValueError("Actual method schedule or configuration changed")
    final = json.loads((arm / "train/completed.json").read_text())
    checkpoint = arm / f"train/checkpoint-{config.steps:04d}"
    evaluated = json.loads((arm / "evaluated-adapter.json").read_text())
    state = json.loads((checkpoint / "state.json").read_text())
    hashes = {name: file_digest(checkpoint / name) for name in (
        "adapter_model.safetensors", "adapter_config.json", "state.json")}
    if (not final["completed"] or final["steps"] != config.steps
            or final["planned_steps"] != config.steps
            or final["stop_reason"] != "update_budget" or state["step"] != config.steps
            or final["final_checkpoint"] != checkpoint.name
            or not final["teacher_unchanged"] or final["nonzero_updates"] != config.steps
            or evaluated["precision"] != "bf16" or evaluated["step"] != config.steps
            or evaluated["source"] != "fixed_final_update" or evaluated["dev_rows"] != 1100
            or evaluated["adapter_sha256"] != hashes["adapter_model.safetensors"]
            or evaluated["adapter_config_sha256"] != hashes["adapter_config.json"]):
        raise ValueError("Evaluated BF16 adapter differs from fixed complete training")
    metrics = load_rows(arm / "train/metrics.jsonl")
    rollouts = load_rows(arm / "train/rollouts.jsonl")
    if len(metrics) != config.steps or len(rollouts) != config.steps:
        raise ValueError("Every optimizer update and rollout must be retained")
    tokens = 0
    repetitions = 1 if config.objective in ("offline_sft", "oracle_sft") else config.group_size
    for step, (metric, rollout, prompts) in enumerate(zip(metrics, rollouts, actual), 1):
        expected_ids = [identity for identity in prompts for _ in range(repetitions)]
        tokens += metric["response_tokens"]
        if (metric["step"] != step or metric["learner_version"] != step
                or metric["rollout_version"] != step - 1
                or rollout["step"] != step or rollout["version"] != step - 1
                or rollout["sample_ids"] != expected_ids
                or len(rollout["texts"]) != len(expected_ids)
                or len(rollout["rewards"]) != len(expected_ids)
                or metric["response_tokens"] < len(expected_ids)
                or metric["cumulative_response_tokens"] != tokens
                or metric["learning_rate_used"] != update_learning_rate(config, step)
                or not math.isfinite(metric["parameter_delta_l2"])
                or metric["parameter_delta_l2"] <= 0
                or not math.isfinite(metric["gradient_norm"])
                or not math.isfinite(metric["loss"])
                or not 0 <= metric["behavior_max_error"] <= config.behavior_tolerance):
            raise ValueError("Recorded optimizer update or executed prompt schedule changed")
    if final["response_tokens"] != tokens or state["tokens_seen"] != tokens:
        raise ValueError("Final checkpoint token budget differs from update history")
    return {"checkpoint": checkpoint.name, "files": hashes,
            "prompt_schedule_sha256": schedule_hash,
            "verified_optimizer_updates": len(metrics), "verified_response_tokens": tokens}


def verify_seed(root: Path, protocol: dict, protocol_hash: str, seed: int,
                training: list[dict]) -> dict:
    completion = json.loads((root / "study/completed.json").read_text())
    worker = json.loads((root / "worker-completed.json").read_text())
    binding = json.loads((root / "study/binding.json").read_text())
    if (not worker["completed"] or not worker["assets_verified"]
            or not worker["tasks"] or any(t["returncode"] for t in worker["tasks"])
            or not completion["completed"] or completion["seed"] != seed
            or completion["protocol_sha256"] != protocol_hash
            or completion["scope"] != "development_only" or completion["holdout_observed"]
            or completion["quality_advantage_supported"]
            or set(completion["summaries"]) != {"initial", *protocol["methods"]}
            or binding["protocol_sha256"] != protocol_hash or binding["seed"] != seed
            or binding["scope"] != "development_only" or binding["holdout_observed"]
            or not binding["fresh_adam_per_method"]
            or binding["common_initial_adapter"] != protocol["initial_adapter_sha256"]):
        raise ValueError("Closed DEV scope or input binding changed")
    events = completion["events"]
    if ([e["method"] for e in events] != list(protocol["methods"])
            or any(e["returncode"] or e["ended"] < e["started"] for e in events)
            or events != json.loads((root / "study/execution.json").read_text())):
        raise ValueError("Incomplete method execution evidence")
    checks = {}
    for label, spec in protocol["methods"].items():
        checks[label] = verify_arm(root / "study" / label, {**spec, "seed": seed},
                                   binding["prompt_schedule_sha256"], training)
    return checks


def main():
    parser = argparse.ArgumentParser()
    for key in ("protocol", "submissions", "training", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--hdfs-root", required=True)
    parser.add_argument("--cli", default=os.environ.get("MERLIN_CLI", "merlin-cli"))
    parser.add_argument("--control-plane", default="cn")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    jobs = json.loads(args.submissions.read_text())
    if (protocol["scope"] != "development_only" or protocol["holdout_files_packaged"]
            or protocol["holdout_observed_in_this_round"]
            or [j["seed"] for j in jobs] != protocol["seeds"]
            or len(jobs) != 4 or len({j["sid"] for j in jobs}) != 4
            or file_digest(args.training) != protocol["train_sha256"]):
        raise ValueError("Frozen four-seed DEV submission or training data changed")
    cli = [args.cli, "--control-plane", args.control_plane, "job-v2"]
    states = {}
    for job in jobs:
        response = query(cli, ["runs", "get"], {"sid": job["sid"]})
        data = response["data"]
        roles = data["resource_config"]["arnold_resource_config"]["roles"]
        if sum(r["num"] * r["gpu"] for r in roles) != 1:
            raise ValueError("Single-GPU allocation changed")
        if data["status"] not in ("done", "success"):
            raise ValueError("All four DEV runs must close successfully before collection")
        states[job["sid"]] = response
    exits = query(cli, ["run-trials", "exit-info"], {
        "trials_list": {"filter": {"job_run_sid_any": list(states)}}})
    cost = allocation_cost(exits["pods"]["data"], exits["trials"]["data"], set(states))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "terminal-runs.json", states)
    write_json(args.output / "terminal-exit-info.json", exits)
    names = ["worker-completed.json", "environment.json", "base-environment.json", "task-events.json",
             "study/completed.json", "study/binding.json", "study/execution.json",
             "study/initial-dev.jsonl", "study/initial-dev.summary.json"]
    for label, spec in protocol["methods"].items():
        names += [f"study/{label}/{name}" for name in (
            "config.json", "training.log", "evaluated-adapter.json",
            "final-dev.jsonl", "final-dev.summary.json", "train/config.json",
            "train/completed.json", "train/schedule.json", "train/contract.json",
            "train/metrics.jsonl", "train/rollouts.jsonl")]
        names += [f"study/{label}/train/checkpoint-{spec['steps']:04d}/{name}" for name in (
            "adapter_model.safetensors", "adapter_config.json", "state.json")]
    records, checks = [], {}
    protocol_hash = file_digest(args.protocol)
    for job in jobs:
        seed = job["seed"]
        root = args.output / f"seed{seed}"
        remote = args.hdfs_root.rstrip("/") + f"/anchor-development-v1/seed{seed}"
        for name in names:
            record = collect(remote + "/" + name, root / name)
            records.append({"path": f"seed{seed}/{name}", **record})
        checks[str(seed)] = verify_seed(
            root, protocol, protocol_hash, seed, load_rows(args.training))
        print(json.dumps({"seed": seed, "all_methods_collected_and_bound": True}), flush=True)
    write_json(args.output / "cost.json", cost)
    write_json(args.output / "collection-audit.json", {
        "verified": True, "all_runs_terminal_successful": True, "files": records,
        "protocol_sha256": protocol_hash, "scope": "development_only", "holdout_observed": False,
        "adapter_checks": checks, "cost_sha256": file_digest(args.output / "cost.json"),
        "boundary": "Final adapters and raw outputs collected independently; no quality claim.",
    })


if __name__ == "__main__":
    main()
