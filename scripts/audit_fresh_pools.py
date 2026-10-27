"""Audit source-pinned unused query and math inputs without evaluating any model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import unicodedata

from servedistill.advantage_contracts import file_digest
from servedistill.data import digest, load_rows
from servedistill.search_data import normalized_query


SUFFIX = "\n只给出最终数值，输出JSON的answer字段。"


def question_identity(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def observed_inputs(roots: list[Path]) -> dict:
    queries, prompts, identities, ids, files = set(), set(), set(), set(), []
    for root in roots:
        paths = sorted(root.glob("*.jsonl"))
        if not paths:
            raise ValueError("Historical input root has no JSONL files: " + str(root))
        for path in paths:
            rows = load_rows(path)
            for row in rows:
                if digest(row["prompt"]) != row["input_sha256"]:
                    raise ValueError("Historical prompt hash differs: " + row["sample_id"])
                prompts.add(row["input_sha256"])
                ids.add(row["sample_id"])
                if "query" in row:
                    queries.add(normalized_query(row["query"]))
                if row["domain"] == "math":
                    identities.add(question_identity(row["prompt"].removesuffix(SUFFIX)))
            files.append({"dataset": root.name, "file": path.name,
                          "sha256": file_digest(path), "rows": len(rows)})
    return {"queries": queries, "prompts": prompts, "math_questions": identities,
            "ids": ids, "files": files}


def available_math(rows: list[dict], observed: dict, source_split: str = "test") -> list[dict]:
    if source_split not in ("train", "test"):
        raise ValueError("Unknown official math split")
    unique, result = set(), []
    for row in rows:
        identity = question_identity(row["question"])
        prompt_hash = digest(row["question"] + SUFFIX)
        source_id = "gsm8k:" + source_split + ":" + digest(row["question"])[:16]
        if identity in unique:
            raise ValueError("Official math source contains duplicate normalized questions")
        unique.add(identity)
        if (identity not in observed["math_questions"] and prompt_hash not in observed["prompts"]
                and source_id not in observed["ids"]):
            result.append(row)
    return result


def audit(examples: Path, gsm8k: Path, source_path: Path, previous: list[Path],
          svamp: Path | None = None, esci_version: str = "small") -> dict:
    import pandas as pd

    source = json.loads(source_path.read_text())
    if (file_digest(examples) != source["esci"]["examples_sha256"]
            or examples.stat().st_size != source["esci"]["examples_bytes"]
            or file_digest(gsm8k) != source["gsm8k"]["sha256"]):
        raise ValueError("Pinned official source bytes differ")
    observed = observed_inputs(previous)
    frame = pd.read_parquet(examples, columns=["query", "split", "small_version",
                                              "product_id", "product_locale"])
    if esci_version == "small":
        frame = frame[frame["small_version"] == 1].copy()
    elif esci_version != "full":
        raise ValueError("Unknown ESCI version")
    frame["query_normalized"] = frame["query"].map(normalized_query)
    split_counts = frame.groupby("query_normalized")["split"].nunique()
    mixed = set(split_counts[split_counts > 1].index)
    test = frame[(frame["split"] == "test")
                 & ~frame["query_normalized"].isin(mixed | observed["queries"])]
    unique_products = test.drop_duplicates(["query_normalized", "product_id", "product_locale"])
    counts = unique_products.groupby("query_normalized").size()
    eligible = counts[counts >= 2]
    math_rows = load_rows(gsm8k)
    source_split = source["gsm8k"].get("split", "test")
    remaining = available_math(math_rows, observed, source_split)
    svamp_counts = None
    if svamp:
        rows = json.loads(svamp.read_text())
        unseen = [
            r for r in rows
            if question_identity(r["Body"].strip() + " " + r["Question"].strip())
            not in observed["math_questions"]
            and digest(r["Body"].strip() + " " + r["Question"].strip() + SUFFIX)
            not in observed["prompts"]
            and "svamp:" + str(r["ID"]) not in observed["ids"]
        ]
        svamp_counts = {"source_sha256": file_digest(svamp), "source_rows": len(rows),
                        "remaining_original_questions": len(unseen)}
    return {
        "scope": "input_pool_audit_only", "model_evaluated": False,
        "confirmation_frozen": False, "quality_advantage_supported": False,
        "source_lock_sha256": file_digest(source_path), "sources": source,
        "historical_files": observed["files"],
        "unique_historical_inputs": len(observed["prompts"]),
        "unique_historical_query_groups": len(observed["queries"]),
        "unique_historical_math_questions": len(observed["math_questions"]),
        "esci": {"version": esci_version,
                 "eligible_unseen_official_test_query_groups": len(eligible),
                 "mixed_official_split_queries_excluded": len(mixed),
                 "minimum_distinct_products": 2},
        "gsm8k": {"official_split": source_split, "official_source_questions": len(math_rows),
                  "remaining_original_questions": len(remaining),
                  "previously_observed_questions_excluded": len(math_rows) - len(remaining)},
        "svamp": svamp_counts,
        "boundary": "Unused in recorded ServeDistill inputs; public benchmark pretraining "
                    "exposure is unknown. Counts do not establish statistical power. Preserve "
                    "the complete historical-file list; labels never select model candidates.",
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("examples", "gsm8k", "source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--previous", type=Path, nargs="+", required=True)
    parser.add_argument("--svamp", type=Path)
    parser.add_argument("--esci-version", choices=("small", "full"), default="small")
    args = parser.parse_args()
    result = audit(args.examples, args.gsm8k, args.source, args.previous, args.svamp, args.esci_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Do not replace an input-pool audit")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("esci", "gsm8k", "svamp", "model_evaluated")}))


if __name__ == "__main__":
    main()
