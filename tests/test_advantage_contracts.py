"""Regression checks for real-holdout grading and paired evidence contracts."""
import json

import pytest

from servedistill.advantage_contracts import file_digest, verify_data, verify_files, verify_labels
from servedistill.advantage_statistics import paired_quality
from servedistill.data import grade


GATE = {
    "minimum_metric": {"search": 0.8, "relevance": 0.4, "substitute": 0.55},
    "minimum_recall": {"substitute": {"yes": 0.25}},
    "minimum_json_valid": 0.95,
    "domain_noninferiority_margin": -0.03,
    "math_noninferiority_margin": -0.05,
    "minimum_positive_seed_pairs": 3,
}


def records(perfect):
    rows = []
    for domain in ("search", "relevance", "substitute", "math"):
        for index in range(24):
            identity = f"{domain}/{index}"
            row = {"sample_id": identity, "input_sha256": identity,
                   "query_group_sha256": identity, "domain": domain,
                   "json_valid": True, "success": perfect}
            if domain == "search":
                row.update(task="esci_ranking", ndcg_eligible=True,
                           ndcg=1.0 if perfect else 0.5)
            elif domain in ("relevance", "substitute"):
                labels = ("E", "S", "C", "I") if domain == "relevance" else ("yes", "no")
                truth = labels[index % len(labels)]
                row.update(task="esci_classification" if domain == "relevance" else "esci_substitute",
                           target_class=truth, predicted_class=truth if perfect else labels[-1])
            rows.append(row)
    return rows


def test_paired_seed_and_query_bindings():
    before, after = [records(False) for _ in range(4)], [records(True) for _ in range(4)]
    result = paired_quality(before, after, GATE, repetitions=40)
    assert result["quality_advantage_supported"]
    assert result["business_macro"]["positive_seed_pairs"] == 4
    assert result["business_macro"]["paired_query_ci95"][0] > 0
    after[0][0]["input_sha256"] = "changed"
    with pytest.raises(ValueError, match="Task/label changed"):
        paired_quality(before, after, GATE, repetitions=40)


def test_constant_substitute_cannot_pass_quality_gate():
    before, after = [records(False) for _ in range(4)], [records(True) for _ in range(4)]
    for rows in after:
        for row in rows:
            if row["domain"] == "substitute":
                row["predicted_class"] = "no"
    result = paired_quality(before, after, GATE, repetitions=40)
    assert not result["all_seed_quality_gates_passed"]
    assert not result["quality_advantage_supported"]


def test_incomplete_seed_matrix_rejected():
    with pytest.raises(ValueError, match="exactly four"):
        paired_quality([records(False)] * 3, [records(True)] * 3, GATE, repetitions=40)


@pytest.mark.parametrize("value", [1, 1.0, "1.00", "01", "1e0"])
def test_svamp_numeric_equivalence(value):
    row = {"domain": "math", "math_source": "SVAMP", "target": {"answer": "1"}}
    assert grade(row, json.dumps({"answer": value}))["success"]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1+0", True, "2"])
def test_svamp_rejects_nonfinite_expressions_and_wrong_answers(value):
    row = {"domain": "math", "math_source": "SVAMP", "target": {"answer": "1"}}
    assert not grade(row, json.dumps({"answer": value}))["success"]


def test_existing_non_svamp_grader_is_unchanged():
    row = {"domain": "math", "target": {"answer": "1"}}
    assert not grade(row, '{"answer":"1.0"}')["success"]


def test_valid_manifest_does_not_hide_changed_holdout_contents(tmp_path):
    splits = {}
    for name in ("train", "dev", "test", "ood"):
        path = tmp_path / (name + ".jsonl")
        path.write_text('{"sample_id":"' + name + '"}\n')
        splits[name] = {"sha256": file_digest(path), "rows": 1}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": splits}))
    before = file_digest(manifest)
    assert verify_data(tmp_path, confirmation=True)["verified_splits"] == list(splits)
    (tmp_path / "test.jsonl").write_text('{"sample_id":"different-query"}\n')
    assert file_digest(manifest) == before
    with pytest.raises(ValueError, match="split changed: test"):
        verify_data(tmp_path, confirmation=True)


def test_valid_model_manifest_does_not_hide_changed_weights(tmp_path):
    names = ("config.json", "tokenizer.json", "model.safetensors")
    records = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"pinned")
        records[name] = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": records}))
    assert verify_files(manifest, tmp_path, model_asset=True)["verified_files"] == 3
    (tmp_path / "model.safetensors").write_bytes(b"mutate")
    with pytest.raises(ValueError, match="contents changed: model.safetensors"):
        verify_files(manifest, tmp_path, model_asset=True)


def test_offline_labels_reject_cross_split_or_prompt_rebinding(tmp_path):
    import hashlib
    from servedistill.data import canonical

    source = {"sample_id": "train/1", "input_sha256": "input/1",
              "prompt": "A product query", "teacher_system": "Classify this query"}
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps(source) + "\n")
    record = {"sample_id": source["sample_id"], "input_sha256": source["input_sha256"],
              "teacher_prompt_sha256": hashlib.sha256(
                  canonical([source["teacher_system"], source["prompt"]]).encode()).hexdigest(),
              "text": '{"label":"E"}', "split": "train"}
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(record) + "\n")
    assert verify_labels(labels, train)["verified_training_rows"] == 1
    for key, value in (("split", "test"), ("input_sha256", "input/2"),
                       ("teacher_prompt_sha256", "other-system")):
        labels.write_text(json.dumps({**record, key: value}) + "\n")
        with pytest.raises(ValueError, match="provenance differs"):
            verify_labels(labels, train)
    labels.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="exactly once"):
        verify_labels(labels, train)


def test_general_offline_labels_bind_the_same_default_as_teacher_scoring(tmp_path):
    import hashlib
    from servedistill.data import TEACHER_SYSTEM, canonical, system_prompt

    row = {"sample_id": "train/math", "input_sha256": "math/1", "prompt": "Compute 1+1"}
    assert system_prompt(row, teacher=True) == TEACHER_SYSTEM
    train = tmp_path / "train.jsonl"
    train.write_text(canonical(row) + "\n")
    record = {**row, "split": "train", "text": '{"answer":"2"}',
              "teacher_prompt_sha256": hashlib.sha256(
                  canonical([TEACHER_SYSTEM, row["prompt"]]).encode()).hexdigest()}
    labels = tmp_path / "labels.jsonl"
    labels.write_text(canonical(record) + "\n")
    assert verify_labels(labels, train)["verified_training_rows"] == 1
    record["teacher_prompt_sha256"] = hashlib.sha256(
        canonical(["", row["prompt"]]).encode()).hexdigest()
    labels.write_text(canonical(record) + "\n")
    with pytest.raises(ValueError, match="provenance differs"):
        verify_labels(labels, train)


def test_teacher_labels_resume_completed_chunks_and_preserve_interrupted_attempt(tmp_path, monkeypatch):
    import sys
    from scripts import run_advantage_labels as driver
    from servedistill.data import canonical

    model, data, output, destination = [tmp_path / name for name in ("model", "data", "out", "persist")]
    model.mkdir()
    data.mkdir()
    files = {}
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        path = model / name
        path.write_bytes(b"pinned-contract-fixture")
        files[name] = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
    manifest = model / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    rows = [{"sample_id": str(index), "input_sha256": str(index),
             "prompt": "Arithmetic " + str(index), "domain": "math",
             "target": {"answer": "1"}} for index in range(260)]
    rows.extend([{**rows[0], "sample_id": "repeat/0"}, {**rows[1], "sample_id": "repeat/1"}])
    train = data / "train.jsonl"
    train.write_text("".join(canonical(row) + "\n" for row in rows))
    (data / "dev.jsonl").write_text("")
    (data / "manifest.json").write_text(json.dumps({
        "files": {"train.jsonl": file_digest(train), "dev.jsonl": file_digest(data / "dev.jsonl")}}))
    calls, failed = [], False

    class FakeRuntime:
        contract = {"fixture": "CPU persistence contract; no GPU quality claim"}

        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, chunk, target, label):
            nonlocal failed
            calls.append([row["sample_id"] for row in chunk])
            records = [{**row, "text": '{"answer":"1"}', "success": True, "json_valid": True,
                        "constraint_violation": False, "truncated": False, "output_tokens": 1}
                       for row in chunk]
            path = target / (label + ".jsonl")
            if len(chunk) < 256 and not failed:
                failed = True
                path.write_text(canonical(records[0]) + "\n")
                raise RuntimeError("Simulated interrupted label generation")
            path.write_text("".join(canonical(row) + "\n" for row in records))

    monkeypatch.setattr(driver, "Runtime", FakeRuntime)
    argv = ["labels", "--teacher", str(model), "--teacher-manifest", str(manifest),
            "--data", str(data), "--output", str(output), "--destination", str(destination)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="Simulated interrupted"):
        driver.main()
    assert (destination / "progress.json").exists()
    assert json.loads((destination / "progress.json").read_text())["completed_chunks"]["0"]["rows"] == 256
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    driver.main()
    assert len(calls) == 3
    assert calls[1] == calls[2]
    assert len(list(output.glob("unique-teacher-chunk-0001-attempt-*.jsonl"))) == 2
    assert verify_labels(output / "teacher-labels.jsonl", train)["verified_training_rows"] == 262
    assert json.loads((destination / "completed.json").read_text())["unique_teacher_prompts"] == 260
