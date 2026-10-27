"""Evaluate exactly one sealed adapter on both batches with full output provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from scripts.worker import publish
from servedistill.advantage_contracts import file_digest, verify_files
from servedistill.data import canonical, load_rows
from servedistill.fresh_contracts import verify_packet
from servedistill.runtime import Runtime


def bind_query_identities(path: Path, source: dict) -> None:
    """The existing Runtime emits sample IDs; persist the frozen cluster identity too."""
    rows = load_rows(path)
    for row in rows:
        sample = source[row["sample_id"]]
        if row["input_sha256"] != sample["input_sha256"]:
            raise ValueError("Runtime output differs from its frozen input")
        if "query_group_sha256" in sample:
            row["query_group_sha256"] = sample["query_group_sha256"]
    temporary = path.with_suffix(".bound")
    with temporary.open("x") as stream:
        stream.write("".join(canonical(row) + "\n" for row in rows))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ("packet", "family", "student", "student-manifest", "adapter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("candidate", "offline_sft", "oracle_sft"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    protocol, sources = verify_packet(
        args.packet, args.family, Path(__file__).resolve().parents[1])
    if args.seed not in protocol["paired_training_seeds"]:
        raise ValueError("Unregistered paired training seed")
    hashes = protocol["adapters"][str(args.seed)][args.arm]
    if any(file_digest(args.adapter / name) != expected for name, expected in hashes.items()):
        raise ValueError("Adapter differs from the frozen DEV choice")
    if file_digest(args.student_manifest) != protocol["student_manifest_sha256"]:
        raise ValueError("Common student manifest changed")
    verified_student = verify_files(args.student_manifest, args.student, model_asset=True)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "protocol_sha256": file_digest(args.packet / "protocol.json"),
        "arm": args.arm, "seed": args.seed, "adapter_hashes": hashes,
        "student_manifest_sha256": file_digest(args.student_manifest),
        "student_verification": verified_student,
        "data_manifest_sha256": protocol["data_manifest_sha256"],
        "started_unix_seconds": time.time(), "test_used_for_selection": False,
        "evaluation": {"batch_size": 8, "max_new_tokens": 128, "sample": False},
    }
    (args.output / "evaluation-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    runtime = Runtime(str(args.student), adapter_path=str(args.adapter))
    summaries = {}
    for split in protocol["required_batches"]:
        summaries[split] = runtime.evaluate(list(sources[split].values()), args.output, split,
                                            batch_size=8, max_new_tokens=128)
        bind_query_identities(args.output / (split + ".jsonl"), sources[split])
        if args.destination:
            publish(args.output, args.destination)
    (args.output / "completed.json").write_text(json.dumps({
        "completed": True, "arm": args.arm, "seed": args.seed,
        "protocol_sha256": provenance["protocol_sha256"],
        "summaries": summaries, "completed_unix_seconds": time.time(),
        "test_used_for_selection": False, "holdout_observed": True,
    }, indent=2) + "\n")
    if args.destination:
        publish(args.output, args.destination)


if __name__ == "__main__":
    main()
