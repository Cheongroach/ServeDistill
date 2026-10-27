"""Freeze training-only hard trajectories from the same repaired teacher."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from servedistill.advantage_contracts import file_digest, verify_data, verify_files
from servedistill.data import canonical, load_rows, system_prompt
from servedistill.runtime import Runtime, seed_all, summarize, write_json
from scripts.run_advantage_arm import Publisher


def main():
    parser = argparse.ArgumentParser()
    for key in ("teacher", "data", "output", "destination"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verification = {
        "teacher": verify_files(args.teacher_manifest, args.teacher, model_asset=True),
        "data": verify_data(args.data, confirmation=False),
    }
    fingerprints = {
        "teacher_manifest_sha256": file_digest(args.teacher_manifest),
        "train_source_sha256": file_digest(args.data / "train.jsonl"),
        "runtime_source_sha256": file_digest(Path(__file__).resolve().parents[1] / "servedistill/runtime.py"),
        "data_source_sha256": file_digest(Path(__file__).resolve().parents[1] / "servedistill/data.py"),
        "contracts_source_sha256": file_digest(
            Path(__file__).resolve().parents[1] / "servedistill/advantage_contracts.py"),
        "driver_sha256": file_digest(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    publisher = Publisher(args.output, args.destination)
    progress_path = args.output / "progress.json"
    if args.resume:
        progress = json.loads(progress_path.read_text())
        if progress["fingerprints"] != fingerprints:
            raise ValueError("Offline label resume changes its teacher, data or execution source")
        if (args.output / "completed.json").exists():
            raise ValueError("Offline labels are already complete")
    else:
        progress = {"fingerprints": fingerprints, "verification": verification,
                    "completed_chunks": {}, "started_unix_seconds": time.time()}
        write_json(progress_path, progress)
    publisher.publish()
    seed_all(202610253)
    original = load_rows(args.data / "train.jsonl")
    ids = {row["sample_id"] for row in original}
    if len(ids) != len(original):
        raise ValueError("Training IDs are not unique")
    representatives, lookup = {}, {}
    for row in original:
        system = system_prompt(row, teacher=True)
        key = hashlib.sha256(canonical([system, row["prompt"]]).encode()).hexdigest()
        representatives.setdefault(key, {**row, "system": system})
        lookup[row["sample_id"]] = key
    started = time.time()
    session_index = len(list(args.output.glob("session-*.json")))
    execution_path = args.output / f"session-{session_index:04d}.json"
    write_json(execution_path, {"started_unix_seconds": started, "completed": False})
    publisher.publish()
    runtime = Runtime(str(args.teacher))
    write_json(args.output / "contract.json", runtime.contract)
    evidence = []
    unique = list(representatives.values())
    for start in range(0, len(unique), 256):
        chunk = unique[start:start + 256]
        key = str(start // 256)
        existing = progress["completed_chunks"].get(key)
        if existing:
            path = args.output / (existing["label"] + ".jsonl")
            if file_digest(path) != existing["sha256"]:
                raise ValueError("Persisted offline teacher responses changed")
            records = load_rows(path)
        else:
            prefix = f"unique-teacher-chunk-{int(key):04d}"
            attempt = len(list(args.output.glob(prefix + "-attempt-*.jsonl")))
            label = f"{prefix}-attempt-{attempt:04d}"
            runtime.evaluate(chunk, args.output, label)
            path = args.output / (label + ".jsonl")
            records = load_rows(path)
            progress["completed_chunks"][key] = {
                "label": label, "sha256": file_digest(path), "rows": len(records)}
        if [row["sample_id"] for row in records] != [row["sample_id"] for row in chunk]:
            raise ValueError("Teacher chunk changed its unique prompt schedule")
        if any(row["input_sha256"] != sample["input_sha256"]
               for row, sample in zip(records, chunk)):
            raise ValueError("Teacher chunk changed its original input")
        evidence.extend(records)
        write_json(progress_path, progress)
        publisher.publish()
    by_id = {row["sample_id"]: row for row in evidence}
    by_key = {key: by_id[row["sample_id"]] for key, row in representatives.items()}
    path = args.output / "teacher-labels.jsonl"
    final_labels = []
    for row in original:
        prediction = by_key[lookup[row["sample_id"]]]
        if row["input_sha256"] != prediction["input_sha256"]:
            raise ValueError("Teacher labels changed the training input")
        final_labels.append({
            **prediction, "sample_id": row["sample_id"],
            "teacher_prompt_sha256": lookup[row["sample_id"]],
            "split": "train",
        })
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(canonical(row) + "\n" for row in final_labels))
    temporary.replace(path)
    write_json(execution_path, {
        "completed": True, "started_unix_seconds": started, "end_unix_seconds": time.time(),
        "elapsed_seconds": time.time() - started, "resumed": args.resume})
    sessions = {path.name: json.loads(path.read_text())
                for path in sorted(args.output.glob("session-*.json"))}
    write_json(args.output / "completed.json", {
        "completed": True, "train_rows": len(original),
        "unique_teacher_prompts": len(representatives),
        **fingerprints,
        "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "teacher_contract": runtime.contract,
        "teacher_unique_training_quality": summarize(evidence),
        "elapsed_seconds": time.time() - started,
        "start_unix_seconds": started, "end_unix_seconds": time.time(),
        "label_sessions": sessions,
        "cost_boundary": "elapsed_seconds covers the final execution session. Total cost "
                         "must include all sessions, artifact verification, and platform "
                         "GPU allocation, including sessions interrupted before completion.",
        "decoding": "greedy, max_new_tokens=128; no grading-based filtering",
        "boundary": "Training-only offline hard labels. OPD uses student on-policy trajectories "
                    "scored by this same teacher; these are not the same trajectory distribution. "
                    "Repeated weighted inputs are labeled once and retain their training IDs.",
    })
    publisher.publish()
    print(json.dumps({"completed": True, "training_rows": len(original)}), flush=True)


if __name__ == "__main__":
    main()
