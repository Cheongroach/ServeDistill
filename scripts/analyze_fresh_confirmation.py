"""Independently regrade every fresh output before applying the spent-alpha gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servedistill.advantage_contracts import file_digest
from servedistill.data import grade, load_rows
from servedistill.fresh_contracts import verify_packet
from servedistill.runtime import summarize
from servedistill.sequential_confirmation import compare


def verified_output(root: Path, split: str, source: dict, protocol: dict,
                    protocol_hash: str, arm: str, seed: int) -> tuple[list[dict], dict]:
    provenance = json.loads((root / "evaluation-provenance.json").read_text())
    complete = json.loads((root / "completed.json").read_text())
    if (provenance["protocol_sha256"] != protocol_hash
            or provenance["arm"] != arm or provenance["seed"] != seed
            or provenance["test_used_for_selection"]
            or provenance["adapter_hashes"] != protocol["adapters"][str(seed)][arm]
            or provenance["student_manifest_sha256"] != protocol["student_manifest_sha256"]
            or provenance["data_manifest_sha256"] != protocol["data_manifest_sha256"]
            or provenance["evaluation"] != {"batch_size": 8, "max_new_tokens": 128, "sample": False}
            or not complete["completed"] or not complete["holdout_observed"]
            or complete["test_used_for_selection"] or complete["arm"] != arm
            or complete["seed"] != seed or complete["protocol_sha256"] != protocol_hash
            or complete["completed_unix_seconds"] < provenance["started_unix_seconds"]):
        raise ValueError("Fresh output provenance or completion changed")
    path = root / (split + ".jsonl")
    rows = load_rows(path)
    if (len(rows) != len(source) or len({r["sample_id"] for r in rows}) != len(rows)
            or {r["sample_id"] for r in rows} != set(source)):
        raise ValueError("Complete one-to-one fresh output coverage required")
    for row in rows:
        sample = source[row["sample_id"]]
        if (row["input_sha256"] != sample["input_sha256"] or row["domain"] != sample["domain"]
                or row.get("query_group_sha256") != sample.get("query_group_sha256")
                or any(row.get(key) != value for key, value in grade(sample, row["text"]).items())):
            raise ValueError("Independent fresh output regrading differs")
    summary = summarize(rows)
    if (summary != json.loads((root / (split + ".summary.json")).read_text())
            or summary != complete["summaries"][split]):
        raise ValueError("Fresh output summary changed")
    return rows, {
        "predictions_sha256": file_digest(path), "rows": len(rows),
        "provenance_sha256": file_digest(root / "evaluation-provenance.json"),
        "completion_sha256": file_digest(root / "completed.json"),
        "summary_sha256": file_digest(root / (split + ".summary.json")),
    }


def analyze(packet: Path, family: Path, root: Path) -> dict:
    protocol, sources = verify_packet(packet, family, Path(__file__).resolve().parents[1])
    protocol_hash = file_digest(packet / "protocol.json")
    batches, evidence = {}, {}
    for split_index, split in enumerate(protocol["required_batches"]):
        predictions = {arm: [] for arm in ("candidate", *protocol["required_controls"])}
        for arm in predictions:
            for seed in protocol["paired_training_seeds"]:
                rows, record = verified_output(
                    root / f"seed{seed}" / arm, split, sources[split], protocol,
                    protocol_hash, arm, seed)
                predictions[arm].append(rows)
                evidence[f"{split}/seed{seed}/{arm}"] = record
        batches[split] = {
            control: compare(
                predictions[control], predictions["candidate"], protocol["quality_gates"],
                protocol["alpha"], protocol["bootstrap_repetitions"],
                protocol["bootstrap_seed"] + split_index,
                chunk_size=protocol["bootstrap_chunk_size"])
            for control in protocol["required_controls"]}
    return {
        "scope": "fresh_confirmation", "protocol_sha256": protocol_hash,
        "attempt": protocol["attempt"], "spent_alpha": protocol["alpha"],
        "selected_dev_candidate": protocol["selected_dev_candidate"],
        "independent_regrading_completed": True, "output_evidence": evidence,
        "prediction_rows_regraded": sum(r["rows"] for r in evidence.values()),
        "batches": batches,
        "quality_advantage_supported": all(
            r["quality_advantage_supported"] for batch in batches.values() for r in batch.values()),
        "cost_advantage_supported": False,
        "coverage_boundary": protocol["coverage_boundary"],
        "population_boundary": protocol["population_boundary"],
        "boundary": "Every frozen seed, arm and batch is required. No early stop, resizing, "
                    "pooled rescue, seed removal or favorable comparison selection.",
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("packet", "family", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.packet, args.family, args.root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "attempt", "prediction_rows_regraded", "spent_alpha", "quality_advantage_supported")}))


if __name__ == "__main__":
    main()
