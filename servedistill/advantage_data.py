"""Development-only class balancing; original labels and held-out queries stay intact."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

from .data import canonical, load_rows


def prepare(previous: Path, output: Path, seed: int = 202610251):
    if output.exists():
        raise FileExistsError("Refusing to overwrite development data")
    rng = random.Random(seed)
    original = {split: load_rows(previous / f"{split}.jsonl") for split in ("train", "dev")}
    rows, examples = [], {}
    for domain in ("search", "relevance", "substitute", "math", "instruction"):
        pool = [row for row in original["train"] if row["domain"] == domain]
        classes = (("E", "S", "C", "I") if domain == "relevance" else
                   ("yes", "no") if domain == "substitute" else ("all",))
        key = "label" if domain == "relevance" else "substitute"
        count = 400 if domain == "relevance" else 800
        if domain == "search":
            count = 1600
        examples[domain] = []
        for label in classes:
            eligible = [row for row in pool if label == "all" or row["target"][key] == label]
            if not eligible:
                raise ValueError("Missing training class: " + domain + "/" + label)
            rng.shuffle(eligible)
            examples[domain].extend(copy.deepcopy(eligible[:2]))
            for index in range(count):
                row = copy.deepcopy(eligible[index % len(eligible)])
                row["source_sample_id"] = row["sample_id"]
                row["sample_id"] += f":balance:{seed}:{index}"
                row["split"] = "train"
                rows.append(row)
    rng.shuffle(rows)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Development replica IDs must be unique")
    dev = original["dev"]
    train_queries = {row.get("query_group_sha256", row["input_sha256"]) for row in rows}
    dev_queries = {row.get("query_group_sha256", row["input_sha256"]) for row in dev}
    if train_queries & dev_queries:
        raise ValueError("Development train/dev leakage")
    output.mkdir(parents=True)
    hashes = {}
    for name, value in (("train.jsonl", rows), ("dev.jsonl", dev)):
        path = output / name
        path.write_text("".join(canonical(row) + "\n" for row in value))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = output / "training-examples.json"
    path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "seed": seed, "source_manifest_sha256": hashlib.sha256(
            (previous / "manifest.json").read_bytes()).hexdigest(),
        "train_rows": len(rows), "unique_training_inputs": len({r["input_sha256"] for r in rows}),
        "train_domains": dict(Counter(row["domain"] for row in rows)),
        "classification_support": dict(Counter(
            domain + "/" + canonical(row["target"])
            for row in rows if (domain := row["domain"]) in ("relevance", "substitute"))),
        "dev_rows": len(dev), "files": hashes,
        "boundary": "Development only; repeated real labeled inputs implement training reweighting. "
                    "Dev is already observed and must not be used as confirmatory evidence. "
                    "No test labels enter training examples or prompts.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.previous, args.output)))


if __name__ == "__main__":
    main()
