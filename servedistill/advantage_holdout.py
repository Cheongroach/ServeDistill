"""Freeze unobserved real queries; selection never depends on their labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from .data import canonical, digest, load_rows
from .search_data import DOMAINS, build_row, download, normalized_query, sha256_file


def prepare(cache: Path, svamp_cache: Path, training: Path, previous: list[Path],
            output: Path, seed: int = 202610252):
    import pandas as pd

    if output.exists():
        raise FileExistsError("Refusing to replace an immutable holdout")
    source = download(cache, "7916cdf6ab75a462e77f20ab40428a10923998d5")
    old_queries, old_inputs, old_ids = set(), set(), set()
    previous_hashes = {}
    for root in previous:
        for path in sorted(root.glob("*.jsonl")):
            previous_hashes[str(path)] = sha256_file(path)
            for row in load_rows(path):
                if "query" in row:
                    old_queries.add(normalized_query(row["query"]))
                old_inputs.add(row["input_sha256"])
                old_ids.add(row["sample_id"])
    frame = pd.read_parquet(cache / "shopping_queries_dataset_examples.parquet")
    frame = frame[frame["small_version"] == 1].copy()
    frame["query_normalized"] = frame["query"].map(normalized_query)
    mixed = frame.groupby("query_normalized")["split"].nunique()
    excluded = old_queries | set(mixed[mixed > 1].index)
    frame = frame[~frame["query_normalized"].isin(excluded)]
    info = frame.groupby(["query_normalized", "split"]).size().reset_index(name="pairs")
    pool = sorted(info[(info["split"] == "test") & (info["pairs"] >= 2)]
                  ["query_normalized"].tolist())
    rng = random.Random(seed)
    rng.shuffle(pool)
    sizes = {"test": 1500, "ood": 500}
    needed = sum(sizes.values()) * len(DOMAINS)
    if len(pool) < needed:
        raise ValueError("Insufficient unseen official-test queries")
    chosen, offset = {}, 0
    for split, count in sizes.items():
        chosen[split] = pool[offset:offset + count * len(DOMAINS)]
        offset += count * len(DOMAINS)
    selected = {query for group in chosen.values() for query in group}
    pairs = frame[frame["query_normalized"].isin(selected)]
    products = pd.read_parquet(cache / "shopping_queries_dataset_products.parquet")
    joined = pairs.merge(products, how="left", on=["product_locale", "product_id"],
                         validate="many_to_one")
    grouped = {
        query: group.drop_duplicates(["product_id", "product_locale"]).to_dict("records")
        for query, group in joined.groupby("query_normalized")}
    svamp_source = json.loads((svamp_cache / "svamp-source.json").read_text())
    for name, metadata in svamp_source["files"].items():
        if sha256_file(svamp_cache / name) != metadata["sha256"]:
            raise ValueError("Pinned SVAMP source differs: " + name)
    arithmetic = []
    for row in json.loads((svamp_cache / "SVAMP.json").read_text()):
        question = row["Body"].strip() + " " + row["Question"].strip()
        prompt = question + "\n只给出最终数值，输出JSON的answer字段。"
        if digest(prompt) in old_inputs:
            continue
        answer = str(row["Answer"])
        if answer.endswith(".0"):
            answer = answer[:-2]
        arithmetic.append({
            "sample_id": "svamp:" + str(row["ID"]), "domain": "math",
            "prompt": prompt, "input_sha256": digest(prompt),
            "target": {"answer": answer},
            "provenance": svamp_source["files"]["SVAMP.json"]["url"],
            "math_source": "SVAMP",
        })
    rng.shuffle(arithmetic)
    if len(arithmetic) < 1000:
        raise ValueError("SVAMP regression requires 1000 unseen original examples")
    query_seen, input_seen, ids_seen = set(old_queries), set(old_inputs), set(old_ids)
    output.mkdir(parents=True)
    manifest = {
        "seed": seed, "source": source, "svamp_source": svamp_source,
        "previous_files_sha256": previous_hashes,
        "excluded_previous_query_groups": len(old_queries),
        "excluded_previous_inputs": len(old_inputs),
        "remaining_eligible_official_test_queries": len(pool),
        "query_input_overlap": 0, "candidate_limit": 8,
        "sampling": "query selection and domain assignment precede label inspection",
        "test_used_for_training_or_selection": False, "splits": {},
        "general_boundary": "SVAMP numeric regression only; no broad general-capability claim. "
                            "Instruction regression remains on already observed development data.",
        "replication_boundary": "ood is a second disjoint official-test query replication, "
                                "not evidence of distribution shift",
    }
    for split in ("train", "dev"):
        path = training / (split + ".jsonl")
        shutil.copyfile(path, output / path.name)
        manifest["splits"][split] = {
            "sha256": sha256_file(path), "rows": len(load_rows(path)),
            "inherited_from": str(training),
            "boundary": "Existing observed development data; not confirmation evidence",
        }
    for index, (split, count) in enumerate(sizes.items()):
        rows = []
        for number, query in enumerate(chosen[split]):
            if query in query_seen:
                raise ValueError("Holdout query group overlaps previous or another split")
            query_seen.add(query)
            rows.append(build_row(grouped[query], split, DOMAINS[number // count], seed))
        rows.extend({**row, "split": split} for row in arithmetic[index * 500:(index + 1) * 500])
        for row in rows:
            if row["input_sha256"] in input_seen or row["sample_id"] in ids_seen:
                raise ValueError("Holdout input or ID overlaps previous or another split")
            input_seen.add(row["input_sha256"])
            ids_seen.add(row["sample_id"])
        rng.shuffle(rows)
        path = output / (split + ".jsonl")
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        manifest["splits"][split] = {
            "sha256": sha256_file(path), "rows": len(rows),
            "domains": dict(Counter(row["domain"] for row in rows)),
            "locales": dict(Counter(row.get("locale", "general") for row in rows)),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for key in ("cache", "svamp-cache", "training", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--previous", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    result = prepare(args.cache, args.svamp_cache, args.training, args.previous, args.output)
    print(json.dumps({key: result[key] for key in (
        "seed", "excluded_previous_query_groups", "remaining_eligible_official_test_queries",
        "query_input_overlap", "splits")}))


if __name__ == "__main__":
    main()
