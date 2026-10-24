"""Export score evidence without prompts, generated text, or platform identifiers."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

FIELDS = ("sample_id", "input_sha256", "domain", "locale", "task", "target_class",
          "predicted_class", "ndcg", "ndcg_eligible", "mrr", "success", "json_valid",
          "constraint_violation", "truncated", "output_tokens", "concurrency", "fallback",
          "latency_seconds", "ttft_seconds", "first_visible_answer_seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "task-metrics.jsonl.gz"
    counts = {}
    with gzip.open(destination, "wt", encoding="utf-8") as stream:
        for source in sorted(args.evidence.glob("host*/*/final/*.jsonl")):
            label, split = source.parent.parent.name, source.stem
            if split not in ("test", "ood"):
                continue
            for line in source.read_text().splitlines():
                row = json.loads(line)
                record = {key: row[key] for key in FIELDS if key in row}
                stream.write(json.dumps({"run": label, "split": split, **record}) + "\n")
                counts[label + "/" + split] = counts.get(label + "/" + split, 0) + 1
    serving_path = args.output / "serving-metrics.jsonl.gz"
    serving_counts = {}
    host = args.evidence / "host3"
    sources = list((host / "serving_native_v1").glob("*/http/requests.jsonl"))
    sources += list((host / "deployment_v1").glob("*/*/requests.jsonl"))
    with gzip.open(serving_path, "wt", encoding="utf-8") as stream:
        for source in sorted(sources):
            label = str(source.parent.relative_to(host))
            for line in source.read_text().splitlines():
                row = json.loads(line)
                record = {key: row[key] for key in FIELDS if key in row}
                stream.write(json.dumps({"run": label, **record}) + "\n")
                serving_counts[label] = serving_counts.get(label, 0) + 1
    (args.output / "serving-metrics-manifest.json").write_text(json.dumps({
        "file": serving_path.name, "sha256": hashlib.file_digest(serving_path.open("rb"), "sha256").hexdigest(),
        "records": serving_counts, "boundary": "separate sweeps at concurrency1/4/16; latency is per request"}, indent=2) + "\n")
    manifest = {"file": destination.name, "sha256": hashlib.file_digest(destination.open("rb"), "sha256").hexdigest(),
                "records": counts, "boundary": "per-run task scores; repeated seeds are not additional independent tasks"}
    (args.output / "task-metrics-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()
