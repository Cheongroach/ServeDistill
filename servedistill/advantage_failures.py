"""Keep failed control outcomes explicit without accepting failed primary arms."""
from __future__ import annotations

import json
from pathlib import Path

from .advantage_contracts import file_digest


def control_failure_checkpoint(path: Path, config: dict, policy: dict, *, require_weights: bool = True) -> tuple[str | None, dict]:
    """Verify a training abort and any last saved complete atomic update."""
    if config["objective"] != "full_kl" or policy.get("objective") != "full_kl":
        raise ValueError("Only predeclared Full-KL controls may report a training failure")
    if config.get("method_time_budget_seconds") is not None:
        raise ValueError("Failed controls cannot establish equal-time completion")
    if (path / "train/completed.json").exists() or (path / "completed.json").exists():
        raise ValueError("A failed control must not have successful completion evidence")
    execution_path = path / "train-execution.json"
    execution = json.loads(execution_path.read_text())
    failure_path = path / "train/failed.json"
    failure = json.loads(failure_path.read_text())
    if (not execution["returncode"] or failure.get("error") != "RuntimeError"
            or failure.get("message") not in policy["training_messages"]):
        raise ValueError("Control failure is not the predeclared observed training abort")
    metrics_path = path / "train/metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    steps = [row["step"] for row in metrics]
    if not steps or steps != list(range(1, steps[-1] + 1)) or steps[-1] > config["steps"]:
        raise ValueError("Failed control has incomplete or invalid update evidence")
    checkpoint = f"checkpoint-{steps[-1]:04d}"
    directory = path / "train" / checkpoint
    hashes = {}
    if directory.exists():
        for name in ("adapter_model.safetensors", "adapter_config.json", "state.json"):
            if name != "adapter_model.safetensors" or require_weights:
                hashes[name] = file_digest(directory / name)
        state = json.loads((directory / "state.json").read_text())
        if state["step"] != steps[-1]:
            raise ValueError("Saved failed-control checkpoint does not match the last atomic update")
    else:
        checkpoint = None
    return checkpoint, {
        "training_failed": True, "training_completed": False, "observed_updates": steps[-1],
        "planned_updates": config["steps"], "reached_update_budget": steps[-1] == config["steps"],
        "checkpoint": checkpoint, "checkpoint_files_sha256": hashes,
        "train_execution_sha256": file_digest(execution_path),
        "training_failure_sha256": file_digest(failure_path),
        "metrics_sha256": file_digest(metrics_path), "failure": failure,
        "boundary": "A saved atomic update is evaluation evidence; training still failed.",
    }
