"""Evaluate one frozen router source after every paired training arm is terminal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from servedistill.advantage_contracts import file_digest, verify_data, verify_files
from servedistill.data import load_rows
from servedistill.runtime import Runtime, write_json


def publish(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".upload")
        shutil.copyfile(path, temporary)
        temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in (
        "protocol", "training-root", "training-seed-manifest", "student",
        "student-manifest", "data", "output", "destination",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = file_digest(args.protocol)
    if (
        protocol["test_observed_before_freeze"]
        or not protocol["execution"]["holdout_evaluation_deferred"]
        or args.label not in protocol["allowed_arms"]
    ):
        raise ValueError("A frozen deferred-evaluation arm is required")
    family, seed_text = args.label.rsplit("_seed", 1)
    seed = int(seed_text)
    if seed not in protocol["paired_seeds"]:
        raise ValueError("Evaluation seed differs from the frozen protocol")
    seed_manifest = json.loads(args.training_seed_manifest.read_text())
    if (
        not seed_manifest["completed"]
        or seed_manifest["seed"] != seed
        or seed_manifest["protocol_sha256"] != protocol_hash
        or seed_manifest["holdout_observed"]
        or args.label not in seed_manifest["arms"]
    ):
        raise ValueError("Paired training completion gate is incomplete")
    for relative, binding in seed_manifest["files"].items():
        if not relative.startswith(args.label + "/"):
            continue
        path = args.training_root.parent / relative
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or file_digest(path) != binding["sha256"]
        ):
            raise ValueError("Persisted training evidence changed: " + relative)
    complete_path = args.training_root / "training-completed.json"
    complete = json.loads(complete_path.read_text())
    adapter = args.training_root / complete["evaluated_adapter_path"]
    if (
        not complete["completed"]
        or complete["holdout_observed"]
        or complete["test_used_for_selection"]
        or complete["postprocessing"] != protocol["arm_postprocessing"][args.label]
        or file_digest(adapter / "adapter_model.safetensors")
        != complete["evaluated_adapter_file_sha256"]
        or file_digest(adapter / "adapter_config.json")
        != complete["evaluated_adapter_config_sha256"]
    ):
        raise ValueError("Frozen evaluation adapter changed")
    data_verification = verify_data(args.data, confirmation=True)
    student_verification = verify_files(
        args.student_manifest, args.student, model_asset=True
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "evaluation-provenance.json", {
        "label": args.label,
        "family": family,
        "seed": seed,
        "protocol_sha256": protocol_hash,
        "training_completion_sha256": file_digest(complete_path),
        "training_seed_manifest_sha256": file_digest(
            args.training_seed_manifest
        ),
        "adapter_file_sha256": file_digest(
            adapter / "adapter_model.safetensors"
        ),
        "adapter_config_sha256": file_digest(adapter / "adapter_config.json"),
        "data_manifest_sha256": file_digest(args.data / "manifest.json"),
        "student_manifest_sha256": file_digest(args.student_manifest),
        "data_verification": data_verification,
        "student_verification": student_verification,
        "test_used_for_selection": False,
        "evaluation_domains": (
            ["substitute"]
            if family == "aopd"
            else ["search", "relevance", "substitute", "math"]
        ),
    })
    runtime = Runtime(str(args.student), adapter_path=str(adapter))
    reports = {}
    for split in ("test", "ood"):
        rows = load_rows(args.data / (split + ".jsonl"))
        if family == "aopd":
            rows = [row for row in rows if row["domain"] == "substitute"]
        reports[split] = runtime.evaluate(rows, args.output, split)
        publish(args.output, args.destination)
    write_json(args.output / "completed.json", {
        "completed": True,
        "label": args.label,
        "family": family,
        "seed": seed,
        "protocol_sha256": protocol_hash,
        "training_completion_sha256": file_digest(complete_path),
        "reports": reports,
        "test_used_for_selection": False,
        "holdout_observed": True,
        "completed_unix_seconds": time.time(),
    })
    publish(args.output, args.destination)


if __name__ == "__main__":
    main()
