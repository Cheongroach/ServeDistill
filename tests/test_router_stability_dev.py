"""Reject changed DEV evidence and format failures before candidate selection."""
import json

import pytest

from scripts.analyze_router_stability_dev import quality, verify_predictions
from servedistill.data import grade
from servedistill.runtime import summarize, write_json


def sample(identity, domain, task, target, **extra):
    return {"sample_id": identity, "domain": domain, "task": task,
            "input_sha256": identity, "locale": "us", "target": target, **extra}


def prediction(source, text=None):
    text = json.dumps(source["target"]) if text is None else text
    return {**source, "text": text, "output_tokens": 8, "truncated": False,
            **grade(source, text)}


@pytest.mark.parametrize("change", ["grade", "input", "duplicate", "summary"])
def test_changed_prediction_evidence_is_rejected(tmp_path, change):
    source = sample("yes", "substitute", "esci_substitute", {"substitute": "yes"})
    row = prediction(source)
    write_json(tmp_path / "fixed320.summary.json", summarize([row]))
    rows = [row]
    if change == "grade":
        row["predicted_class"] = "no"
    elif change == "input":
        row["input_sha256"] = "changed"
    elif change == "duplicate":
        rows.append(row)
    else:
        write_json(tmp_path / "fixed320.summary.json", {"changed": True})
    (tmp_path / "fixed320.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in rows))
    with pytest.raises(ValueError):
        verify_predictions(tmp_path, "fixed320", {source["sample_id"]: source})


def test_format_failure_blocks_an_otherwise_high_quality_candidate():
    sources = [
        sample("search", "search", "esci_ranking", {"ranking": ["a", "b"]},
               labels={"a": "E", "b": "I"}),
        *[sample(label, "relevance", "esci_classification", {"label": label})
          for label in ("E", "S", "C", "I")],
        *[sample(f"substitute/{label}/{index}", "substitute", "esci_substitute",
                 {"substitute": label}) for label in ("yes", "no") for index in range(100)],
    ]
    gate = {
        "minimum_metric": {"search": .8, "relevance": .45, "substitute": .55},
        "minimum_json_valid": .99,
        "minimum_recall": {"relevance": {label: .2 for label in ("E", "S", "C", "I")},
                           "substitute": {"yes": .3, "no": .3}},
    }
    rows = [prediction(source) for source in sources]
    assert quality(rows, gate)["absolute_gates_passed"]
    for index in range(3):
        rows[-1 - index] = prediction(sources[-1 - index], "invalid")
    result = quality(rows, gate)
    assert not result["absolute_gates_passed"]
    failed = [c["criterion"] for c in result["checks"] if not c["passed"]]
    assert failed == ["substitute/json_valid"]
    assert result["summary"]["domains"]["substitute"]["macro_f1"] > .98
