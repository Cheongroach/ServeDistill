"""Select two disjoint, source-pinned holdouts without label-based sampling."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random

from scripts.audit_fresh_pools import SUFFIX, available_math, observed_inputs
from .advantage_contracts import file_digest
from .data import canonical, digest, load_rows
from .search_data import DOMAINS, build_row, normalized_query


def choose_inputs(pool: list[str], math_rows: list[dict], business_count: int,
                  math_count: int, seed: int) -> tuple[dict, dict]:
    if (type(business_count) is not int or type(math_count) is not int
            or business_count < 1 or math_count < 1):
        raise ValueError("Positive integer sample counts are required")
    if len(pool) != len(set(pool)):
        raise ValueError("Duplicate normalized query identities")
    needed = 2 * business_count * len(DOMAINS)
    if len(pool) < needed or len(math_rows) < 2 * math_count:
        raise ValueError("Insufficient untouched inputs for both fixed batches")
    queries = sorted(pool)
    arithmetic = sorted(math_rows, key=lambda r: r["question"])
    rng = random.Random(seed)
    rng.shuffle(queries)
    rng.shuffle(arithmetic)
    chosen_queries, chosen_math = {}, {}
    for index, split in enumerate(("confirmation", "replication")):
        offset = index * business_count * len(DOMAINS)
        chosen_queries[split] = {
            domain: queries[offset + i * business_count:offset + (i + 1) * business_count]
            for i, domain in enumerate(DOMAINS)}
        chosen_math[split] = arithmetic[index * math_count:(index + 1) * math_count]
    return chosen_queries, chosen_math


def prepare(examples: Path, products: Path, gsm8k: Path, source_path: Path,
            previous: list[Path], output: Path, business_count: int, math_count: int,
            seed: int) -> dict:
    import pandas as pd

    if output.exists():
        raise FileExistsError("Refusing to replace a fresh confirmation packet")
    source = json.loads(source_path.read_text())
    for path, expected, size in (
        (examples, source["esci"]["examples_sha256"], source["esci"]["examples_bytes"]),
        (products, source["esci"]["products_sha256"], source["esci"]["products_bytes"]),
        (gsm8k, source["gsm8k"]["sha256"], None),
    ):
        if file_digest(path) != expected or (size is not None and path.stat().st_size != size):
            raise ValueError("Pinned fresh source bytes differ: " + path.name)
    if source["gsm8k"]["split"] != "train":
        raise ValueError("This population requires the declared GSM8K official train source")
    observed = observed_inputs(previous)
    # No relevance labels or product text are loaded until selection is written.
    frame = pd.read_parquet(examples, columns=["query", "split", "product_id", "product_locale"])
    frame["query_normalized"] = frame["query"].map(normalized_query)
    splits = frame.groupby("query_normalized")["split"].nunique()
    mixed = set(splits[splits > 1].index)
    test = frame[(frame["split"] == "test")
                 & ~frame["query_normalized"].isin(mixed | observed["queries"])]
    counts = test.drop_duplicates(["query_normalized", "product_id", "product_locale"]).groupby(
        "query_normalized").size()
    pool = counts[counts >= 2].index.tolist()
    arithmetic = available_math(load_rows(gsm8k), observed, "train")
    queries, questions = choose_inputs(pool, arithmetic, business_count, math_count, seed)
    output.mkdir(parents=True)
    selection = {
        "seed": seed, "queries": queries,
        "math_question_sha256": {
            split: [digest(r["question"]) for r in rows] for split, rows in questions.items()},
        "labels_used_for_selection": False,
    }
    (output / "selection.json").write_text(canonical(selection) + "\n")
    selected = {q for groups in queries.values() for group in groups.values() for q in group}
    labeled = pd.read_parquet(examples)
    labeled["query_normalized"] = labeled["query"].map(normalized_query)
    labeled = labeled[(labeled["split"] == "test")
                      & labeled["query_normalized"].isin(selected)]
    facts = pd.read_parquet(products)
    joined = labeled.merge(facts, on=["product_locale", "product_id"], how="left",
                           validate="many_to_one", indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError("Selected product facts are missing; do not redraw")
    groups = {
        query: group.sort_values(["product_locale", "product_id"]).drop_duplicates(
            ["product_locale", "product_id"]).to_dict("records")
        for query, group in joined.groupby("query_normalized")}
    inputs, identities = set(observed["prompts"]), set(observed["ids"])
    manifest = {
        "scope": "fresh_confirmation_inputs", "source_lock_sha256": file_digest(source_path),
        "selection_sha256": file_digest(output / "selection.json"), "seed": seed,
        "historical_files": observed["files"], "sources": source,
        "eligible_unseen_queries_before_sampling": len(pool),
        "eligible_unseen_math_before_sampling": len(arithmetic),
        "business_count_per_domain_per_batch": business_count,
        "math_count_per_batch": math_count, "labels_used_for_selection": False,
        "model_predictions_observed": False, "splits": {},
        "population": "ESCI full official test; GSM8K official train held out by this project",
        "boundary": "All selected queries and original questions are disjoint from supplied "
                    "historical inputs and from the other batch. Pretraining exposure unknown. "
                    "Class support is reported after fixed selection; missing support cannot "
                    "be repaired by label-based redraw.",
    }
    for split in ("confirmation", "replication"):
        rows = [
            build_row(groups[query], split, domain, seed)
            for domain, subset in queries[split].items() for query in subset]
        for question in questions[split]:
            prompt = question["question"] + SUFFIX
            rows.append({
                "sample_id": "gsm8k:train:" + digest(question["question"])[:16],
                "domain": "math", "split": split, "prompt": prompt,
                "input_sha256": digest(prompt),
                "target": {"answer": question["answer"].split("####")[-1].strip().replace(",", "")},
                "math_source": "GSM8K", "official_source_split": "train",
                "provenance": source["gsm8k"]["repository"] + "@" + source["gsm8k"]["revision"],
            })
        for row in rows:
            if row["input_sha256"] in inputs or row["sample_id"] in identities:
                raise ValueError("Fresh input or ID overlaps recorded inputs")
            inputs.add(row["input_sha256"])
            identities.add(row["sample_id"])
        random.Random(f"{seed}:{split}:order").shuffle(rows)
        path = output / (split + ".jsonl")
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        manifest["splits"][split] = {
            "sha256": file_digest(path), "rows": len(rows),
            "domains": dict(Counter(row["domain"] for row in rows)),
            "classes": {
                domain: dict(Counter(canonical(r["target"]) for r in rows if r["domain"] == domain))
                for domain in ("relevance", "substitute")},
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
