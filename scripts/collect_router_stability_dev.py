"""Collect closed DEV jobs, exact adapter sources, and every allocation interval."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess

from servedistill.adapter_average import verify_average
from servedistill.advantage_contracts import file_digest
from servedistill.runtime import write_json


REPO = Path(__file__).resolve().parents[1]
CLI = [os.environ.get("MERLIN_CLI", "merlin-cli"), "--control-plane",
       os.environ.get("MERLIN_CONTROL_PLANE", "cn"), "job-v2"]
HDFS = os.environ.get("SERVEDISTILL_HDFS_ADVANTAGE")


def query(command: list[str], request: dict) -> dict:
    response = json.loads(subprocess.check_output(
        [*CLI, *command, "--json", json.dumps(request)], timeout=90))
    if response.get("ResponseBase", {}).get("code", 0):
        raise ValueError("Platform query failed")
    return response


def collect(remote: str, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".collecting")
    env = {**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true"}
    with temporary.open("wb") as stream:
        subprocess.run(["hdfs", "dfs", "-cat", remote], stdout=stream,
                       env=env, timeout=180, check=True)
    record = {"sha256": file_digest(temporary), "bytes": temporary.stat().st_size}
    if target.exists():
        if (file_digest(target) != record["sha256"]
                or target.stat().st_size != record["bytes"]):
            raise ValueError("Existing evidence differs from its remote source: " + str(target))
        temporary.unlink()
    else:
        temporary.replace(target)
    return record


def main() -> None:
    if not HDFS:
        raise ValueError("Set SERVEDISTILL_HDFS_ADVANTAGE to your evidence root")
    ops = REPO / ".ops/router-stability-dev-v1"
    root = REPO / "reports/advantage/raw/router-stability-dev-v1"
    contract_path = ops / "contract.json"
    contract = json.loads(contract_path.read_text())
    jobs = json.loads((ops / "submissions.json").read_text())
    if (contract["scope"] != "development_only"
            or [r["seed"] for r in jobs] != contract["seeds"]
            or len({r["sid"] for r in jobs}) != len(jobs)):
        raise ValueError("DEV submission scope changed")
    states = {}
    for job in jobs:
        response = query(["runs", "get"], {"sid": job["sid"]})
        data = response["data"]
        if data["status"] not in ("done", "success"):
            raise ValueError("All DEV jobs must close successfully before collection")
        roles = data["resource_config"]["arnold_resource_config"]["roles"]
        if sum(r["num"] * r["gpu"] for r in roles) != 1:
            raise ValueError("Declared single-GPU allocation changed")
        states[job["sid"]] = response
    exit_info = query(["run-trials", "exit-info"], {
        "trials_list": {"filter": {"job_run_sid_any": list(states)}}})
    root.mkdir(parents=True, exist_ok=True)
    write_json(ops / "terminal-runs.json", states)
    write_json(ops / "terminal-exit-info.json", exit_info)
    trials = exit_info["trials"]["data"]
    pods = exit_info["pods"]["data"]
    if {p["trial_sid"] for p in pods} != {t["sid"] for t in trials}:
        raise ValueError("Allocation audit must retain every trial")
    costs = []
    for pod in pods:
        start, stop = pod.get("start_at"), pod.get("stop_at")
        preallocation = (not start and not pod.get("container_id")
                         and pod["exit_status"] == -80002)
        if preallocation:
            seconds, gpu = 0.0, 0
        else:
            if not start or not stop or pod["gpuv"] != "NVIDIA-H20":
                raise ValueError("A closed GPU allocation interval is required")
            seconds = (datetime.datetime.fromisoformat(stop.replace("Z", "+00:00"))
                       - datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
            if seconds < 0:
                raise ValueError("Negative allocation interval")
            gpu = 1
        costs.append({
            "job_run_sid": pod["job_run_sid"], "trial_sid": pod["trial_sid"],
            "trial_pod_sid": pod["sid"], "exit_status": pod["exit_status"],
            "start_at": start, "stop_at": stop, "gpu_count": gpu,
            "allocation_seconds": seconds, "gpu_hours": seconds * gpu / 3600,
        })
    audit, adapter_checks = [], {}
    for job in jobs:
        seed = job["seed"]
        output = root / f"seed{seed}"
        remote = f"{HDFS}/router-stability-dev-v1/seed{seed}"
        names = ["completed.json", "worker-completed.json", "environment.json",
                 "base-environment.json", "task-events.json"]
        names += [label + suffix
                  for label in (*contract["aopd_rules"], "offline_sft", "oracle_sft")
                  for suffix in (".jsonl", ".summary.json")]
        names += [f"adapters/{label}/{name}" for label in ("last2", "all4")
                  for name in ("adapter_model.safetensors", "adapter_config.json",
                               "averaging-provenance.json")]
        for name in names:
            record = collect(remote + "/" + name, output / name)
            audit.append({"seed": seed, "path": f"seed{seed}/{name}", **record})
        completion = json.loads((output / "completed.json").read_text())
        if (not completion["completed"] or completion["seed"] != seed
                or completion["contract_sha256"] != file_digest(contract_path)
                or completion["scope"] != "development_only"
                or completion["holdout_observed"] or completion["precision"] != "bf16"):
            raise ValueError("Evaluation completion binding changed")
        source = ops / "source-training" / f"seed{seed}"
        source_remote = f"{HDFS}/aopd-router-confirmation-v1/train/seed{seed}"
        marker_path = (REPO / ".ops/aopd-router-confirmation-v1/training-terminal-evidence"
                       / f"seed{seed}-training-seed-complete.json")
        marker = json.loads(marker_path.read_text())
        if (not marker["completed"] or marker["holdout_observed"]
                or marker["seed"] != seed
                or marker["protocol_sha256"] != contract["source_protocol_sha256"]):
            raise ValueError("Source training binding changed")
        source_names = [f"aopd_seed{seed}/train/completed.json"]
        source_names += [f"aopd_seed{seed}/train/checkpoint-{step:04d}/{name}"
                         for step in (160, 320, 480, 640)
                         for name in ("adapter_model.safetensors", "adapter_config.json",
                                      "state.json", "training.pt")]
        for family in ("offline_sft", "oracle_sft"):
            relative = f"{family}_seed{seed}/training-completed.json"
            record = collect(source_remote + "/" + relative, source / relative)
            if record != marker["files"][relative]:
                raise ValueError("Frozen SFT completion binding changed")
            arm = json.loads((source / relative).read_text())
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                source_names.append(f"{family}_seed{seed}/{arm['evaluated_adapter_path']}/{name}")
        for name in source_names:
            record = collect(source_remote + "/" + name, source / name)
            binding = marker["files"].get(name)
            if binding is not None and record != binding:
                raise ValueError("Frozen source artifact changed: " + name)
            audit.append({"seed": seed, "source_path": name,
                          "bound_to_original_manifest": binding is not None, **record})
        checks = {}
        for label, spec in contract["aopd_rules"].items():
            if spec["kind"] == "lora_factor_mean_fp32":
                path = output / "adapters" / label
                checks[label] = verify_average(source / f"aopd_seed{seed}/train", path, spec, 640)
            else:
                path = source / f"aopd_seed{seed}/train/checkpoint-{spec['step']:04d}"
                checks[label] = {"adapter_file_sha256": file_digest(path / "adapter_model.safetensors"),
                                 "adapter_config_sha256": file_digest(path / "adapter_config.json")}
        for family in ("offline_sft", "oracle_sft"):
            arm = source / f"{family}_seed{seed}"
            complete = json.loads((arm / "training-completed.json").read_text())
            path = arm / complete["evaluated_adapter_path"]
            checks[family] = {"adapter_file_sha256": file_digest(path / "adapter_model.safetensors"),
                              "adapter_config_sha256": file_digest(path / "adapter_config.json")}
        for label, evidence in checks.items():
            actual = completion["source_adapters"][label]
            if (actual["adapter_sha256"] != evidence["adapter_file_sha256"]
                    or actual["adapter_config_sha256"] != evidence["adapter_config_sha256"]):
                raise ValueError("Evaluated adapter differs from independently verified source")
        adapter_checks[str(seed)] = checks
        print(json.dumps({"seed": seed, "predictions_collected": True,
                          "exact_average_and_adapter_bindings_verified": True}), flush=True)
    write_json(root / "collection-audit.json", {
        "verified": True, "scope": "development_only", "holdout_observed": False,
        "all_runs_terminal_successful": True, "contract_sha256": file_digest(contract_path),
        "files": audit, "adapter_checks": adapter_checks,
    })
    write_json(REPO / "reports/advantage/ROUTER_STABILITY_DEV_COST.json", {
        "verified": True, "scope": "DEV stability study only", "all_trials_retained": True,
        "trial_count": len(trials), "records": costs,
        "allocated_gpu_hours": sum(r["gpu_hours"] for r in costs),
        "cost_advantage_supported": False,
        "boundary": "Pod allocation intervals, not billing or useful GPU compute time. "
                    "Prior confirmation and historical development costs are separate.",
    })


if __name__ == "__main__":
    main()
