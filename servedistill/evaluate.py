"""Frozen final evaluation and teacher qualification using independent labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_rows
from .runtime import Runtime, seed_all, stratified, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--adapter")
    parser.add_argument("--evaluate-teacher", action="store_true")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("qualification", "final", "teacher_labels"), required=True)
    parser.add_argument("--per-domain", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample-ids", type=Path)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(20261024)
    runtime = Runtime(args.student, args.teacher, args.adapter)
    write_json(output / "contract.json", runtime.contract)
    if args.phase == "qualification":
        rows = stratified(load_rows(Path(args.data) / "dev.jsonl"), args.per_domain)
        runtime.evaluate(rows, output, "student", args.batch_size, args.max_new_tokens)
        if not args.teacher:
            raise ValueError("Qualification requires a teacher model")
        runtime.evaluate(rows, output, "teacher", args.batch_size, args.max_new_tokens, teacher=True)
    elif args.phase == "teacher_labels":
        if not args.teacher:
            raise ValueError("Frozen teacher trajectories require a teacher model")
        rows = load_rows(Path(args.data) / "train.jsonl")
        if args.sample_ids:
            ids = set(json.loads(args.sample_ids.read_text()))
            available = {row["sample_id"] for row in rows}
            if not ids <= available:
                raise ValueError("Teacher labeling IDs contain non-training tasks")
            rows = [row for row in rows if row["sample_id"] in ids]
        # Only train examples; independent labels retained in source, never in prompt.
        runtime.evaluate(rows, output, "teacher-train", args.batch_size,
                         args.max_new_tokens, teacher=True)
    else:
        for split in ("test", "ood"):
            runtime.evaluate(load_rows(Path(args.data) / f"{split}.jsonl"),
                             output, split, args.batch_size, args.max_new_tokens,
                             teacher=args.evaluate_teacher)
    print(json.dumps({"event": "evaluation_complete", "phase": args.phase}), flush=True)


if __name__ == "__main__":
    main()
