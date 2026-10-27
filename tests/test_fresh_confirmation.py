"""Independent checks of fresh sampling, attempt consumption and query inference."""
import copy
import json

import numpy as np
import pytest

from scripts.analyze_fresh_confirmation import verified_output
from scripts.analyze_fresh_confirmation import analyze as analyze_confirmation
from scripts.evaluate_fresh_confirmation import bind_query_identities
from servedistill.advantage_contracts import file_digest
from servedistill.data import canonical, digest, grade
from servedistill.fresh_confirmation import choose_inputs, prepare
from servedistill.runtime import summarize
from servedistill.sequential_confirmation import (
    attempt_alpha, compare, locked_ledger, lower_bound, read_attempts, reserve_attempt,
)


def test_sampling_is_disjoint_and_does_not_depend_on_labels():
    pool = [f"query-{i:03d}" for i in range(80)]
    math = [{"question": f"math-{i:03d}", "answer": f"#### {i}"} for i in range(40)]
    first = choose_inputs(pool, math, 5, 10, 101)
    relabeled = [{**r, "answer": "#### unrelated"} for r in reversed(math)]
    second = choose_inputs(list(reversed(pool)), relabeled, 5, 10, 101)
    assert first[0] == second[0]
    assert {s: [r["question"] for r in rows] for s, rows in first[1].items()} == {
        s: [r["question"] for r in rows] for s, rows in second[1].items()}
    batches = [{q for group in first[0][s].values() for q in group}
               for s in ("confirmation", "replication")]
    assert len(batches[0]) == len(batches[1]) == 15
    assert not batches[0] & batches[1]
    assert not ({r["question"] for r in first[1]["confirmation"]}
                & {r["question"] for r in first[1]["replication"]})
    with pytest.raises(ValueError, match="Insufficient"):
        choose_inputs(pool[:29], math, 5, 10, 101)
    with pytest.raises(ValueError, match="Duplicate"):
        choose_inputs(pool + [pool[0]], math, 5, 10, 101)


def test_spending_telescopes_and_failed_reservations_are_not_refunded(tmp_path):
    assert attempt_alpha(0.05, 1) == 0.025
    assert sum(attempt_alpha(0.05, r) for r in range(1, 1001)) == pytest.approx(
        0.05 * 1000 / 1001)
    family = tmp_path / "family.json"
    family.write_text(json.dumps({"total_one_sided_alpha": 0.05}))
    ledger = tmp_path / "ledger"
    with locked_ledger(ledger):
        first = reserve_attempt(ledger, family, {"candidate": "cpu-fixture"})
    # There is deliberately no completion file or dataset for the first attempt.
    with locked_ledger(ledger):
        second = reserve_attempt(ledger, family, {"candidate": "cpu-fixture2"})
    assert first["consumed_even_if_failed"]
    assert second["attempt"] == 2
    assert second["alpha"] == pytest.approx(0.05 / 6)
    (ledger / "attempt-0001.json").write_text("{}\n")
    with pytest.raises((ValueError, KeyError)):
        read_attempts(ledger, second["family_sha256"])


def fixture_rows():
    rows = []
    for domain, task, labels in (
        ("search", "esci_ranking", ("rank",)),
        ("relevance", "esci_classification", ("E", "S", "C", "I")),
        ("substitute", "esci_substitute", ("yes", "no")),
        ("math", "math", ("numeric",)),
    ):
        for index in range(40):
            identity = f"{domain}/{index}"
            label = labels[index % len(labels)]
            rows.append({
                "sample_id": identity, "input_sha256": digest(identity),
                "query_group_sha256": digest(identity), "domain": domain, "task": task,
                "target_class": label, "predicted_class": label, "ndcg": 1.0,
                "ndcg_eligible": True, "success": True, "json_valid": True,
            })
    return rows


def gates():
    return {
        "minimum_metric": {"search": 0.8, "relevance": 0.45, "substitute": 0.55},
        "minimum_json_valid": 0.99, "minimum_positive_seed_pairs": 3,
        "minimum_recall": {"relevance": dict.fromkeys(("E", "S", "C", "I"), 0.2),
                           "substitute": {"yes": 0.3, "no": 0.3}},
        "domain_noninferiority_margin": -0.03, "math_noninferiority_margin": -0.05,
    }


def run(before, after, **kwargs):
    return compare(before, after, gates(), 0.025, 8000, 101, **kwargs)


def test_identical_outputs_do_not_establish_superiority_and_pairing_is_enforced():
    rows = fixture_rows()
    pairs = [copy.deepcopy(rows) for _ in range(4)]
    result = run(pairs, pairs)
    assert result["business_macro"]["delta"] == 0
    assert not result["quality_advantage_supported"]
    assert result == run(pairs, pairs)
    changed = copy.deepcopy(pairs)
    changed[3][0]["input_sha256"] = "different"
    with pytest.raises(ValueError, match="Task/label changed"):
        run(pairs, changed)
    repeated = copy.deepcopy(pairs)
    for rows in repeated:
        rows[40]["query_group_sha256"] = rows[0]["query_group_sha256"]
    with pytest.raises(ValueError, match="across domains"):
        run(repeated, repeated)
    with pytest.raises(ValueError, match="tail draws"):
        compare(pairs, pairs, gates(), 0.001, 8000, 101)


def test_large_paired_advantage_still_requires_math_and_schema_gates():
    good = [fixture_rows() for _ in range(4)]
    weak = copy.deepcopy(good)
    for rows in weak:
        for row in rows:
            if row["domain"] == "search":
                row["ndcg"] = 0.8
            elif row["domain"] != "math" and int(row["sample_id"].split("/")[-1]) % 5 == 0:
                row["predicted_class"] = "INVALID"
    assert run(weak, good)["quality_advantage_supported"]
    regression = copy.deepcopy(good)
    for row in regression[0]:
        if row["domain"] == "relevance":
            row["json_valid"] = False
    assert not run(weak, regression)["quality_advantage_supported"]
    regression = copy.deepcopy(good)
    for rows in regression:
        for row in rows:
            if row["domain"] == "math":
                row["success"] = False
    assert not run(weak, regression)["quality_advantage_supported"]
    assert lower_bound(np.arange(100), 0.025) == 1


def test_real_parquet_join_excludes_history_and_mixed_official_splits(tmp_path):
    import pandas as pd

    examples, products = [], []
    for index in range(32):
        for product in range(2):
            identity = f"{index}/{product}"
            examples.append({"query": f"query {index}", "split": "test", "product_id": identity,
                             "product_locale": "us", "esci_label": "E" if product == 0 else "I"})
            products.append({"product_id": identity, "product_locale": "us",
                             "product_title": "ordinary product facts"})
    examples.append({**examples[2], "split": "train"})  # query 1 is a mixed official split.
    example_path, product_path = tmp_path / "examples.parquet", tmp_path / "products.parquet"
    pd.DataFrame(examples).to_parquet(example_path)
    pd.DataFrame(products).to_parquet(product_path)
    math = [{"question": f"math {i}", "answer": f"work #### {i}"} for i in range(20)]
    math_path = tmp_path / "math.jsonl"
    math_path.write_text("".join(canonical(r) + "\n" for r in math))
    source = {
        "esci": {"examples_sha256": file_digest(example_path), "examples_bytes": example_path.stat().st_size,
                 "products_sha256": file_digest(product_path), "products_bytes": product_path.stat().st_size},
        "gsm8k": {"sha256": file_digest(math_path), "split": "train", "repository": "cpu-fixture",
                  "revision": "cpu-fixture"},
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    old = tmp_path / "historical"
    old.mkdir()
    historical = [
        {"sample_id": "old/query", "domain": "search", "query": " QUERY   0 ",
         "prompt": "old product prompt", "input_sha256": digest("old product prompt")},
        {"sample_id": "old/math", "domain": "math", "prompt": "MATH   0",
         "input_sha256": digest("MATH   0")},
    ]
    (old / "train.jsonl").write_text("".join(canonical(r) + "\n" for r in historical))
    output = tmp_path / "packet"
    manifest = prepare(example_path, product_path, math_path, source_path, [old], output, 3, 5, 101)
    selection = json.loads((output / "selection.json").read_text())
    assert manifest["eligible_unseen_queries_before_sampling"] == 30
    assert manifest["eligible_unseen_math_before_sampling"] == 19
    chosen = {q for group in selection["queries"].values() for subset in group.values() for q in subset}
    assert not chosen & {"query 0", "query 1"}
    assert digest("math 0") not in {
        h for values in selection["math_question_sha256"].values() for h in values}
    assert manifest["splits"]["confirmation"]["rows"] == 14
    with pytest.raises(FileExistsError):
        prepare(example_path, product_path, math_path, source_path, [old], output, 3, 5, 101)


def test_runtime_outputs_are_bound_and_independently_regraded(tmp_path):
    sample = {"sample_id": "query/1", "domain": "relevance", "task": "esci_classification",
              "target": {"label": "E"}, "locale": "us", "prompt": "ordinary query",
              "input_sha256": digest("ordinary query"), "query_group_sha256": digest("query/1")}
    text = '{"label":"E"}'
    row = {k: sample[k] for k in ("sample_id", "domain", "input_sha256")}
    row.update(text=text, output_tokens=5, truncated=False, **grade(sample, text))
    path = tmp_path / "confirmation.jsonl"
    path.write_text(canonical(row) + "\n")
    bind_query_identities(path, {sample["sample_id"]: sample})
    records = [json.loads(path.read_text())]
    summary = summarize(records)
    (tmp_path / "confirmation.summary.json").write_text(json.dumps(summary))
    adapter = {"adapter_model.safetensors": "fixture-hash"}
    protocol = {"adapters": {"101": {"candidate": adapter}},
                "student_manifest_sha256": "student", "data_manifest_sha256": "data"}
    provenance = {
        "protocol_sha256": "protocol", "arm": "candidate", "seed": 101,
        "test_used_for_selection": False, "adapter_hashes": adapter,
        "student_manifest_sha256": "student", "data_manifest_sha256": "data",
        "evaluation": {"batch_size": 8, "max_new_tokens": 128, "sample": False},
        "started_unix_seconds": 1,
    }
    completed = {"completed": True, "holdout_observed": True, "test_used_for_selection": False,
                 "arm": "candidate", "seed": 101, "protocol_sha256": "protocol",
                 "completed_unix_seconds": 2, "summaries": {"confirmation": summary}}
    (tmp_path / "evaluation-provenance.json").write_text(json.dumps(provenance))
    (tmp_path / "completed.json").write_text(json.dumps(completed))
    actual, evidence = verified_output(tmp_path, "confirmation", {sample["sample_id"]: sample},
                                       protocol, "protocol", "candidate", 101)
    assert actual[0]["query_group_sha256"] == sample["query_group_sha256"]
    assert evidence["rows"] == 1
    altered = {**actual[0], "success": False}
    path.write_text(canonical(altered) + "\n")
    with pytest.raises(ValueError, match="regrading differs"):
        verified_output(tmp_path, "confirmation", {sample["sample_id"]: sample},
                        protocol, "protocol", "candidate", 101)


@pytest.mark.parametrize("failure", ["none", "strong_control_tie", "replication_regression", "missing_seed"])
def test_complete_fresh_analysis_requires_both_controls_and_both_batches(tmp_path, failure):
    """Synthetic predictions test the final decision; they are not model evidence."""
    packet, output = tmp_path / "packet", tmp_path / "outputs"
    (packet / "data").mkdir(parents=True)
    family = {
        "total_one_sided_alpha": 0.05, "paired_training_seeds": [101, 211, 307, 401],
        "required_controls": ["offline_sft", "oracle_sft"],
        "required_batches": ["confirmation", "replication"], "bootstrap_chunk_size": 256,
        "bootstrap_minimum_expected_tail_draws": 200,
    }
    family_path = tmp_path / "family.json"
    family_path.write_text(json.dumps(family))
    adapters = {str(s): {arm: {"adapter_model.safetensors": "fixture-adapter"}
                         for arm in ("candidate", *family["required_controls"])}
                for s in family["paired_training_seeds"]}
    receipt = {
        "family_sha256": file_digest(family_path), "attempt": 1, "alpha": 0.025,
        "consumed_even_if_failed": True,
        "bindings": {"adapters": adapters, "selected_dev_candidate": "anchor10",
                     "source_lock_sha256": "fixture-source",
                     "business_count_per_domain_per_batch": 40, "math_count_per_batch": 40},
    }
    (packet / "reservation.json").write_text(json.dumps(receipt))
    (packet / "dev-analysis.json").write_text(json.dumps({
        "selected_dev_candidate": "anchor10", "all_three_weights_and_controls_complete": True,
        "advance_to_fresh_confirmation": True}))
    (packet / "data/selection.json").write_text("{}\n")
    manifest = {
        "labels_used_for_selection": False, "model_predictions_observed": False,
        "selection_sha256": file_digest(packet / "data/selection.json"),
        "source_lock_sha256": "fixture-source",
        "business_count_per_domain_per_batch": 40, "math_count_per_batch": 40, "splits": {},
    }
    all_samples = {}
    for split in family["required_batches"]:
        samples = []
        for row in fixture_rows():
            identity = split + "/" + row["sample_id"]
            sample = {
                "sample_id": identity, "prompt": identity, "input_sha256": digest(identity),
                "domain": row["domain"], "split": split, "task": row["task"], "locale": "us",
                "query_group_sha256": digest(identity),
            }
            if row["domain"] == "search":
                sample.update(labels={"P0": "E", "P1": "I"}, target={"ranking": ["P0", "P1"]})
            elif row["domain"] == "relevance":
                sample["target"] = {"label": row["target_class"]}
            elif row["domain"] == "substitute":
                sample["target"] = {"substitute": row["target_class"]}
            else:
                sample["target"] = {"answer": "2"}
            samples.append(sample)
        path = packet / "data" / (split + ".jsonl")
        path.write_text("".join(canonical(r) + "\n" for r in samples))
        manifest["splits"][split] = {
            "sha256": file_digest(path), "rows": len(samples),
            "domains": dict.fromkeys(("search", "relevance", "substitute", "math"), 40)}
        all_samples[split] = samples
    (packet / "data/manifest.json").write_text(json.dumps(manifest))
    protocol = {
        "scope": "frozen_fresh_confirmation", "family_sha256": file_digest(family_path),
        "model_predictions_observed_before_freeze": False, "alpha": 0.025, "attempt": 1,
        "paired_training_seeds": family["paired_training_seeds"],
        "required_batches": family["required_batches"], "required_controls": family["required_controls"],
        "reservation_sha256": file_digest(packet / "reservation.json"),
        "dev_analysis_sha256": file_digest(packet / "dev-analysis.json"),
        "data_manifest_sha256": file_digest(packet / "data/manifest.json"),
        "data_splits": manifest["splits"], "adapters": adapters, "selected_dev_candidate": "anchor10",
        "source_sha256": {}, "bootstrap_repetitions": 8000, "bootstrap_seed": 101,
        "bootstrap_chunk_size": 256, "student_manifest_sha256": "fixture-student",
        "quality_gates": gates(), "coverage_boundary": "CPU fixture", "population_boundary": "CPU fixture",
    }
    protocol_path = packet / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    protocol_hash = file_digest(protocol_path)
    for seed in family["paired_training_seeds"]:
        for arm in ("candidate", *family["required_controls"]):
            root = output / f"seed{seed}" / arm
            root.mkdir(parents=True)
            summaries = {}
            for split, samples in all_samples.items():
                rows = []
                for index, sample in enumerate(samples):
                    wrong = arm != "candidate" and index % 5 == 0 and sample["domain"] != "math"
                    if failure == "strong_control_tie" and arm == "oracle_sft":
                        wrong = False
                    if (failure == "replication_regression" and arm == "candidate"
                            and split == "replication" and sample["domain"] == "math"):
                        wrong = True
                    text = "{}" if wrong else canonical(sample["target"])
                    row = {k: sample[k] for k in
                           ("sample_id", "domain", "input_sha256", "query_group_sha256")}
                    rows.append({**row, "text": text, "output_tokens": 5, "truncated": False,
                                 **grade(sample, text)})
                (root / (split + ".jsonl")).write_text("".join(canonical(r) + "\n" for r in rows))
                summaries[split] = summarize(rows)
                (root / (split + ".summary.json")).write_text(json.dumps(summaries[split]))
            (root / "evaluation-provenance.json").write_text(json.dumps({
                "protocol_sha256": protocol_hash, "arm": arm, "seed": seed,
                "test_used_for_selection": False, "adapter_hashes": adapters[str(seed)][arm],
                "student_manifest_sha256": "fixture-student",
                "data_manifest_sha256": protocol["data_manifest_sha256"],
                "evaluation": {"batch_size": 8, "max_new_tokens": 128, "sample": False},
                "started_unix_seconds": 1,
            }))
            (root / "completed.json").write_text(json.dumps({
                "completed": True, "holdout_observed": True, "test_used_for_selection": False,
                "arm": arm, "seed": seed, "protocol_sha256": protocol_hash,
                "completed_unix_seconds": 2, "summaries": summaries,
            }))
    if failure == "missing_seed":
        (output / "seed401/candidate/replication.jsonl").unlink()
        with pytest.raises(FileNotFoundError):
            analyze_confirmation(packet, family_path, output)
    else:
        result = analyze_confirmation(packet, family_path, output)
        assert result["prediction_rows_regraded"] == 3840
        assert result["quality_advantage_supported"] == (failure == "none")
