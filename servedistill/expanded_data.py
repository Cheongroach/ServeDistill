"""Larger ESCI/general splits isolated from all pilot query groups and inputs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from .data import canonical, digest, load_rows, make_instruction, read_gsm8k
from .search_data import DOMAINS, GAINS, build_row, download, normalized_query, sha256_file


def prepare(cache: Path, previous: Path, output: Path, seed=20261025):
    import pandas as pd
    if output.exists():
        raise FileExistsError("Refusing to overwrite frozen expansion data")
    source = download(cache, "7916cdf6ab75a462e77f20ab40428a10923998d5")
    old = [row for split in ("train", "dev", "test", "ood")
           for row in load_rows(previous / f"{split}.jsonl")]
    old_queries = {normalized_query(row["query"]) for row in old if "query" in row}
    old_inputs = {row["input_sha256"] for row in old}
    old_ids = {row["sample_id"] for row in old}
    examples = pd.read_parquet(cache / "shopping_queries_dataset_examples.parquet")
    examples = examples[examples["small_version"] == 1].copy()
    examples["query_normalized"] = examples["query"].map(normalized_query)
    mixed = examples.groupby("query_normalized")["split"].nunique()
    excluded = set(mixed[mixed > 1].index) | old_queries
    examples = examples[~examples["query_normalized"].isin(excluded)]
    info = examples.groupby(["query_normalized", "split"]).size().reset_index(name="pairs")
    info = info[info["pairs"] >= 2]
    rng = random.Random(seed)
    pools = {}
    for split in ("train", "test"):
        pools[split] = info[info["split"] == split]["query_normalized"].tolist()
        rng.shuffle(pools[split])
    sizes = {"train": (3000, 2000, 3000), "dev": (300, 200, 300),
             "test": (1500, 600, 1500), "ood": (500, 200, 500)}
    chosen, offsets = {}, {"train": 0, "test": 0}
    for split, (business, _, _) in sizes.items():
        official = "train" if split in ("train", "dev") else "test"
        count, start = business * len(DOMAINS), offsets[official]
        chosen[split] = pools[official][start:start + count]
        if len(chosen[split]) != count:
            raise ValueError("Insufficient isolated official ESCI queries")
        offsets[official] += count
    selected = {query for values in chosen.values() for query in values}
    pairs = examples[examples["query_normalized"].isin(selected)]
    products = pd.read_parquet(cache / "shopping_queries_dataset_products.parquet")
    joined = pairs.merge(products, how="left", on=["product_locale", "product_id"],
                         validate="many_to_one")
    grouped = {query: frame.drop_duplicates(["product_id", "product_locale"]).to_dict("records")
               for query, frame in joined.groupby("query_normalized")}
    general, general_sources = {}, {}
    for official in ("train", "test"):
        expected = {"train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
                    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"}[official]
        cached = cache / f"gsm8k-{official}.jsonl"
        if cached.exists():
            if sha256_file(cached) != expected:
                raise ValueError("GSM8K cached source differs from pinned original source")
            values = [json.loads(line) for line in cached.read_text().split("\n") if line]
            metadata = {"sha256": expected, "license": "MIT (OpenAI grade-school-math)",
                        "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
                               f"master/grade_school_math/data/{official}.jsonl"}
        else:
            values, metadata = read_gsm8k(official)
            if metadata["sha256"] != expected:
                raise ValueError("GSM8K remote source differs from pinned original source")
        general_sources[official] = metadata
        general[official] = [value for value in values if
            "gsm8k:" + official + ":" + digest(value["question"])[:16] not in old_ids]
        rng.shuffle(general[official])
    math_offsets = {"train": 0, "test": 0}
    query_seen, input_seen = set(old_queries), set(old_inputs)
    product_sets, manifests = {}, {}
    output.mkdir(parents=True)
    for split, (business, math_count, instruction_count) in sizes.items():
        rows = []
        for index, query in enumerate(chosen[split]):
            if query in query_seen:
                raise ValueError("Query leakage")
            query_seen.add(query)
            rows.append(build_row(grouped[query], split, DOMAINS[index // business], seed))
        official = "train" if split in ("train", "dev") else "test"
        start = math_offsets[official]
        values = general[official][start:start + math_count]
        if len(values) != math_count:
            raise ValueError("Insufficient isolated GSM8K examples")
        math_offsets[official] += math_count
        for value in values:
            prompt = value["question"] + "\n只给出最终数值，输出JSON的answer字段。"
            rows.append({"sample_id": "gsm8k:" + official + ":" + digest(value["question"])[:16],
                "domain": "math", "split": split, "prompt": prompt,
                "input_sha256": digest(prompt),
                "target": {"answer": value["answer"].split("####")[-1].strip().replace(",", "")},
                "provenance": general_sources[official]["url"]})
        accepted, number = 0, 0
        pending = {row["input_sha256"] for row in rows}
        while accepted < instruction_count:
            row = make_instruction(split, number, seed)
            number += 1
            row["sample_id"] = f"expanded:{seed}:{row['sample_id']}"
            if row["input_sha256"] in input_seen or row["input_sha256"] in pending:
                continue
            rows.append(row)
            pending.add(row["input_sha256"])
            accepted += 1
        for row in rows:
            if row["input_sha256"] in input_seen or row["sample_id"] in old_ids:
                raise ValueError("Input or sample ID overlaps pilot/new split")
            input_seen.add(row["input_sha256"])
        rng.shuffle(rows)
        path = output / f"{split}.jsonl"
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        product_sets[split] = {p for row in rows for p in row.get("product_ids", [])}
        manifests[split] = {"rows": len(rows),
            "domains": dict(Counter(row["domain"] for row in rows)),
            "locales": dict(Counter(row.get("locale", "general") for row in rows)),
            "target_classes": dict(Counter(canonical(row["target"]) for row in rows
                if row["domain"] in ("relevance", "substitute"))), "sha256": sha256_file(path)}
    manifest = {"seed": seed, "source": source, "general_sources": general_sources,
        "previous_manifest_sha256": sha256_file(previous / "manifest.json"),
        "pilot_query_groups_excluded": len(old_queries), "pilot_inputs_excluded": len(old_inputs),
        "query_input_overlap": 0, "splits": manifests,
        "product_overlap_train_test": len(product_sets["train"] & product_sets["test"]),
        "candidate_limit": 8, "gains": GAINS,
        "replication_boundary": "disjoint official-test query replication, not distribution shift"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cache, args.previous, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
