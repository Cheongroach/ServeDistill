"""Regrade development evidence and retain every screening checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_expanded import classification_metrics
from servedistill.data import grade, load_rows
from servedistill.runtime import summarize, write_json


def analyze(root: Path, data: Path):
    source = {row["sample_id"]: row for row in load_rows(data / "dev.jsonl")}
    reports = {}
    for path in sorted(root.rglob("*.jsonl")):
        if path.name == "teacher-labels.jsonl":
            continue
        if path.name.startswith(("metrics", "rollouts")):
            continue
        if not path.name.startswith(("dev-", "selected-dev", "student-", "teacher-")):
            continue
        rows = load_rows(path)
        if not rows:
            continue
        if any(row["sample_id"] not in source for row in rows):
            raise ValueError("Screen evidence contains a non-development query")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("Screen contains repeated evaluation queries")
        verified = []
        for record in rows:
            row = source[record["sample_id"]]
            if row["input_sha256"] != record["input_sha256"]:
                raise ValueError("Screen query hash changed")
            actual = grade(row, record["text"])
            for key in ("json_valid", "success", "reward", "ndcg", "target_class", "predicted_class"):
                if key in actual and actual[key] != record.get(key):
                    raise ValueError("Regrading differs: " + str(path) + "/" + key)
            verified.append(record)
        summary_path = path.with_name(path.stem + ".summary.json")
        completed = summary_path.exists()
        actual_summary = summarize(verified)
        if completed:
            expected = json.loads(summary_path.read_text())
            if sum(domain["n"] for domain in expected["domains"].values()) != len(verified):
                raise ValueError("Screen summary and response count differ")
            for domain, values in expected["domains"].items():
                if abs(values["selection_score"] - actual_summary["domains"][domain]["selection_score"]) > 1e-12:
                    raise ValueError("Regraded metric differs from the worker summary")
        reports[str(path.relative_to(root))] = {
            "summary": actual_summary,
            "evaluation_complete": completed,
            "classes": {
                domain: classification_metrics([row for row in verified if row["domain"] == domain])
                for domain in ("relevance", "substitute")
                if any(row["domain"] == domain for row in verified)},
            "boundary": "Observed development queries; selection evidence only",
        }
    return {"boundary": "No holdout outputs read; no confirmation claim",
            "evaluations": reports, "verified_development_evaluations": len(reports)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.data)
    write_json(args.output, result)
    print(json.dumps({"verified_development_evaluations": result["verified_development_evaluations"]}))


if __name__ == "__main__":
    main()
