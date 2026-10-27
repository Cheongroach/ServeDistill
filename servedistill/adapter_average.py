"""Materialize and verify a predeclared mean of saved FP32 LoRA factors."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file

from .advantage_contracts import file_digest


RULE = "Arithmetic mean of saved FP32 LoRA factors; not a mean of BA deltas"


def validate_postprocessing(spec: dict, updates: int) -> list[int]:
    if spec == {"kind": "final_checkpoint"}:
        return []
    if spec == {"kind": "fixed_checkpoint", "step": 320}:
        if updates != 640:
            raise ValueError("The frozen checkpoint rule requires 640 completed updates")
        return [320]
    if set(spec) != {"kind", "steps"} or spec["kind"] != "lora_factor_mean_fp32":
        raise ValueError("Unknown frozen adapter postprocessing")
    steps = spec["steps"]
    declared = {320: ([240, 320], [80, 160, 240, 320]),
                640: ([480, 640], [160, 320, 480, 640])}
    if (updates not in declared or steps not in declared[updates]
            or any(type(step) is not int for step in steps)):
        raise ValueError("Average must use a declared rule for its update budget")
    return steps


def source_average(train: Path, spec: dict, updates: int) -> tuple[dict, dict]:
    steps = validate_postprocessing(spec, updates)
    if not steps:
        raise ValueError("Final checkpoint does not require averaging")
    complete = json.loads((train / "completed.json").read_text())
    if not complete.get("completed") or complete["steps"] != updates:
        raise ValueError("Averaging requires complete source training")
    records, states = {}, []
    config_hash = None
    for step in steps:
        checkpoint = train / f"checkpoint-{step:04d}"
        state = json.loads((checkpoint / "state.json").read_text())
        if state["step"] != step:
            raise ValueError("Saved source checkpoint update changed")
        files = {
            name: {"sha256": file_digest(checkpoint / name),
                   "bytes": (checkpoint / name).stat().st_size}
            for name in ("adapter_model.safetensors", "adapter_config.json",
                         "state.json", "training.pt")
        }
        current_hash = files["adapter_config.json"]["sha256"]
        if config_hash is None:
            config_hash = current_hash
        elif current_hash != config_hash:
            raise ValueError("Source LoRA configurations differ")
        records[checkpoint.name] = {
            "files": files, "adapter_tensor_sha256": state["adapter_sha256"]}
        states.append(load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu"))
    if any(set(state) != set(states[0]) for state in states) or not states[0]:
        raise ValueError("Saved source tensor sets differ")
    average = {}
    for key, first in states[0].items():
        if first.dtype != torch.float32 or any(
                state[key].shape != first.shape or state[key].dtype != first.dtype
                for state in states):
            raise ValueError("Average requires matching FP32 LoRA factors")
        if not all(bool(torch.isfinite(state[key]).all()) for state in states):
            raise ValueError("Saved source factors contain nonfinite values")
        average[key] = torch.stack([state[key] for state in states]).mean(dim=0)
    provenance = {
        "postprocessing": spec, "source_checkpoints": records,
        "source_training_completion_sha256": file_digest(train / "completed.json"),
        "rule": RULE, "training_updates": updates, "postprocessing_only": True,
        "resumable_optimizer": False,
    }
    return average, provenance


def materialize_average(train: Path, adapter: Path, spec: dict, updates: int) -> dict:
    average, provenance = source_average(train, spec, updates)
    adapter.mkdir(parents=True, exist_ok=False)
    save_file(average, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(train / f"checkpoint-{updates:04d}/adapter_config.json",
                    adapter / "adapter_config.json")
    provenance.update({
        "adapter_file_sha256": file_digest(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_digest(adapter / "adapter_config.json"),
    })
    (adapter / "averaging-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return verify_average(train, adapter, spec, updates)


def verify_average(train: Path, adapter: Path, spec: dict, updates: int) -> dict:
    expected, source = source_average(train, spec, updates)
    actual = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    if set(actual) != set(expected) or any(
            actual[key].dtype != expected[key].dtype
            or not torch.equal(actual[key], expected[key]) for key in expected):
        raise ValueError("Evaluated adapter differs from the frozen source average")
    source.update({
        "adapter_file_sha256": file_digest(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_digest(adapter / "adapter_config.json"),
    })
    if json.loads((adapter / "averaging-provenance.json").read_text()) != source:
        raise ValueError("Averaged adapter provenance changed")
    if source["adapter_config_sha256"] != file_digest(
            train / f"checkpoint-{updates:04d}/adapter_config.json"):
        raise ValueError("Evaluated LoRA configuration differs from its source")
    return source
