"""CPU checks for rejecting incomplete or altered independent evidence."""
import dataclasses
import json

import pytest

from scripts.analyze_advantage_confirmation import verified_arm
from servedistill.data import grade
from servedistill.runtime import summarize, write_json
from servedistill.scheduling import Config


@pytest.fixture
def confirmation(tmp_path):
    config = {"objective": "oracle_sft", "steps": 80, "seed": 29}
    samples = [
        {"sample_id": "math/1", "domain": "math", "target": {"answer": "2"},
         "input_sha256": "fixture/math", "math_source": "SVAMP"},
        {"sample_id": "relevance/1", "domain": "relevance", "target": {"label": "E"},
         "input_sha256": "fixture/relevance", "task": "esci_classification", "locale": "en"},
    ]
    records = [
        {**sample, "text": json.dumps(sample["target"]), "output_tokens": 5,
         "truncated": False, **grade(sample, json.dumps(sample["target"]))}
        for sample in samples
    ]
    path = tmp_path / "arm"
    write_json(path / "start-provenance.json", {
        "phase": "confirmation", "label": "arm", "seed": 29,
        "protocol_sha256": "cpu-fixture-protocol", "test_used_for_selection": False,
        "config_sha256": "cpu-fixture-config",
    })
    write_json(path / "train/config.json", dataclasses.asdict(Config(**config)))
    write_json(path / "train/completed.json", {
        "completed": True, "steps": 80, "stop_reason": "update_budget",
    })
    write_json(path / "completed.json", {
        "completed": True, "phase": "confirmation", "checkpoint": "checkpoint-0080",
        "test_used_for_selection": False,
    })
    (path / "final").mkdir()
    (path / "final/test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    write_json(path / "final/test.summary.json", summarize(records))
    return tmp_path, config, {row["sample_id"]: row for row in samples}


def verify(fixture):
    root, config, source = fixture
    return verified_arm(
        root, "arm", {"config_sha256": "cpu-fixture-config"}, "cpu-fixture-protocol",
        config, source, "test")


def test_independent_evidence_regrades_successfully(confirmation):
    rows, evidence = verify(confirmation)
    assert len(rows) == evidence["rows"] == 2
    assert evidence["final_checkpoint"] == "checkpoint-0080"


@pytest.mark.parametrize("change", ["grading", "missing_row", "config", "checkpoint", "protocol"])
def test_altered_or_incomplete_confirmation_is_rejected(confirmation, change):
    root = confirmation[0] / "arm"
    if change in ("grading", "missing_row"):
        path = root / "final/test.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "grading":
            records[0]["success"] = False
        else:
            records.pop()
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    elif change == "config":
        path = root / "train/config.json"
        config = json.loads(path.read_text())
        write_json(path, {**config, "learning_rate": 1e-3})
    elif change == "checkpoint":
        path = root / "completed.json"
        completed = json.loads(path.read_text())
        write_json(path, {**completed, "checkpoint": "checkpoint-0040"})
    else:
        path = root / "start-provenance.json"
        provenance = json.loads(path.read_text())
        write_json(path, {**provenance, "protocol_sha256": "different-protocol"})
    with pytest.raises(ValueError):
        verify(confirmation)
