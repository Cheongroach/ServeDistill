"""Verify the actual artifacts bound to a frozen experimental protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .data import canonical, load_rows, system_prompt


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(manifest_path: Path, root: Path, model_asset: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("files")
    if not records:
        raise ValueError("Pinned artifact manifest has no files")
    count, total = 0, 0
    for name, record in records.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact manifest escapes its root")
        path = root / relative
        expected = record if isinstance(record, str) else record["sha256"]
        if file_digest(path) != expected:
            raise ValueError("Pinned artifact contents changed: " + name)
        if isinstance(record, dict) and path.stat().st_size != record["bytes"]:
            raise ValueError("Pinned artifact size changed: " + name)
        count += 1
        total += path.stat().st_size
    if model_asset:
        if "config.json" not in records or not any(name.endswith(".safetensors") for name in records):
            raise ValueError("Model manifest must cover configuration and weights")
        if not any("tokenizer" in name for name in records):
            raise ValueError("Model manifest must cover its tokenizer")
    return {"manifest_sha256": file_digest(manifest_path),
            "verified_files": count, "verified_bytes": total}


def verify_data(root: Path, confirmation: bool) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "splits" not in manifest:
        if confirmation:
            raise ValueError("Confirmation requires a manifest covering both holdout splits")
        return verify_files(manifest_path, root)
    splits = ("train", "dev", "test", "ood") if confirmation else ("train", "dev")
    for split in splits:
        record = manifest["splits"][split]
        path = root / (split + ".jsonl")
        if file_digest(path) != record["sha256"]:
            raise ValueError("Pinned data split changed: " + split)
        if len(load_rows(path)) != record["rows"]:
            raise ValueError("Pinned data row count changed: " + split)
    return {"manifest_sha256": file_digest(manifest_path), "verified_splits": list(splits)}


def verify_labels(path: Path, training_path: Path) -> dict:
    source = load_rows(training_path)
    labels = load_rows(path)
    by_id = {row["sample_id"]: row for row in labels}
    if len(by_id) != len(labels) or set(by_id) != {row["sample_id"] for row in source}:
        raise ValueError("Offline labels must bind exactly once to every training ID")
    for row in source:
        prediction = by_id[row["sample_id"]]
        system = system_prompt(row, teacher=True)
        prompt_hash = hashlib.sha256(canonical([system, row["prompt"]]).encode()).hexdigest()
        if (prediction.get("split") != "train"
                or prediction["input_sha256"] != row["input_sha256"]
                or prediction.get("teacher_prompt_sha256") != prompt_hash):
            raise ValueError("Offline label provenance differs from its training prompt")
        if not isinstance(prediction.get("text"), str):
            raise ValueError("Offline hard trajectory must be a text completion")
    return {"labels_sha256": file_digest(path), "verified_training_rows": len(source)}


def verify_cost_credit(path: Path, config: dict) -> dict:
    """Bind shared repair/label/setup credits to persisted execution intervals."""
    manifest = json.loads(path.read_text())
    if manifest["method_time_budget_seconds"] != config["method_time_budget_seconds"]:
        raise ValueError("Cost ledger differs from the frozen method time budget")
    total, seen, records = 0.0, set(), []
    for component in manifest["components"]:
        relative = Path(component["evidence_file"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("Unsafe or duplicate cost evidence")
        seen.add(str(relative))
        evidence_path = path.parent / relative
        if file_digest(evidence_path) != component["evidence_sha256"]:
            raise ValueError("Measured cost evidence changed")
        evidence = json.loads(evidence_path.read_text())
        start, end = evidence["start_unix_seconds"], evidence["end_unix_seconds"]
        gpu, divisor = evidence["allocated_gpu_count"], component["amortization_runs"]
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in (start, end))
                or end < start or type(gpu) is not int or gpu < 1
                or type(divisor) is not int or divisor < 1):
            raise ValueError("Invalid measured GPU interval or amortization")
        credited = (end - start) * gpu / divisor
        total += credited
        records.append({"kind": component["kind"], "credited_gpu_seconds": credited,
                        "amortization_runs": divisor,
                        "evidence_sha256": component["evidence_sha256"]})
    if not math.isclose(total, config.get("prior_method_seconds", 0.0),
                        rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("Prior method cost does not match the measured ledger")
    return {"manifest_sha256": file_digest(path), "prior_method_seconds": total,
            "components": records,
            "boundary": "Credits bind measured intervals; completeness and full platform "
                        "GPU costs still require the final experiment cost audit."}
