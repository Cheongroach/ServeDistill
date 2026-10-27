"""Keep offline training trajectories separate from development predictions."""
import json

import pytest

from scripts.analyze_advantage_screen import analyze


def test_offline_training_labels_are_not_development_predictions(tmp_path):
    data = tmp_path / "data"
    root = tmp_path / "evidence"
    data.mkdir()
    (root / "teacher_labels").mkdir(parents=True)
    (data / "dev.jsonl").write_text("")
    (root / "teacher_labels/teacher-labels.jsonl").write_text(
        json.dumps({"sample_id": "training-only", "split": "train", "text": "{}"}) + "\n")
    assert analyze(root, data)["verified_development_evaluations"] == 0
    (root / "selected-dev.jsonl").write_text(
        json.dumps({"sample_id": "training-only", "text": "{}"}) + "\n")
    with pytest.raises(ValueError, match="non-development"):
        analyze(root, data)
