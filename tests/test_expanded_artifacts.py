"""Evidence guards reject incomplete or altered evaluations and executions."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_expanded import (
    attach_source, export_rows, require_completed, require_execution, require_runtime,
    validate_training_work,
)
from servedistill.data import grade


class ExpandedArtifactTests(unittest.TestCase):
    def test_complete_frozen_evaluation_regrades_and_binds_query(self):
        source = [{
            "sample_id": "q1", "input_sha256": "digest", "domain": "math",
            "target": {"answer": "12"}, "query_group_sha256": "group1",
        }]
        text = '{"answer":"12"}'
        rows = [{
            "sample_id": "q1", "input_sha256": "digest", "text": text,
            **grade(source[0], text),
        }]
        result = attach_source(rows, source)
        self.assertEqual(result[0]["query_group_sha256"], "group1")
        for altered in ([], [*rows, rows[0]]):
            with self.assertRaisesRegex(ValueError, "full split"):
                attach_source(altered, source)
        altered = copy.deepcopy(rows)
        altered[0]["input_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "hash changed"):
            attach_source(altered, source)
        altered = copy.deepcopy(rows)
        altered[0]["success"] = False
        with self.assertRaisesRegex(ValueError, "grading"):
            attach_source(altered, source)

    def test_terminal_and_execution_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps({"completed": False}))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                require_completed(path)
            for status, begin, end in ((1, 1, 2), (0, 2, 1), (0, 1, 1)):
                path.write_text(json.dumps({
                    "returncode": status, "start_unix_seconds": begin, "end_unix_seconds": end}))
                with self.assertRaisesRegex(ValueError, "Failed execution"):
                    require_execution(path)
            path.write_text(json.dumps({
                "returncode": 0, "start_unix_seconds": 1, "end_unix_seconds": 2}))
            self.assertEqual(require_execution(path)["returncode"], 0)

    def test_public_row_digest_matches_decompressed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl.gz"
            rows = [{"sample_id": "q1", "success": True}]
            manifest = export_rows(path, rows)
            self.assertEqual(manifest["records"], 1)
            self.assertEqual(manifest["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with gzip.open(path, "rt") as stream:
                self.assertEqual([json.loads(line) for line in stream], rows)

    def test_generated_work_reconciles_schedule_tokens_and_completed_record(self):
        metrics = [
            {"step": 1, "response_tokens": 3, "cumulative_response_tokens": 3,
             "parameter_delta_l2": .01},
            {"step": 2, "response_tokens": 5, "cumulative_response_tokens": 8,
             "parameter_delta_l2": .02},
        ]
        rollouts = [
            {"step": step, "version": step - 1, "sample_ids": [identity] * 2,
             "texts": ["a", "b"], "rewards": [0., 1.]}
            for step, identity in ((1, "q0"), (2, "q1"))
        ]
        schedule, completed = [["q0"], ["q1"]], {"response_tokens": 8}
        def validate(a=metrics, b=rollouts, c=schedule, d=completed):
            return validate_training_work(a, b, c, d, {"q0", "q1"},
                                          updates=2, prompts=1, group_size=2)
        self.assertEqual(validate(), {"generated_trajectories": 4, "response_tokens": 8})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate(b=rollouts[:1])
        with self.assertRaisesRegex(ValueError, "frozen prompt pool"):
            validate(c=[["outside"], ["q1"]])
        altered = copy.deepcopy(rollouts)
        altered[1]["sample_ids"][0] = "q0"
        with self.assertRaisesRegex(ValueError, "trajectories"):
            validate(b=altered)
        altered = copy.deepcopy(metrics)
        altered[1]["cumulative_response_tokens"] = 9
        with self.assertRaisesRegex(ValueError, "Cumulative"):
            validate(a=altered)
        altered = copy.deepcopy(metrics)
        altered[1]["parameter_delta_l2"] = 0.
        with self.assertRaisesRegex(ValueError, "no effective"):
            validate(a=altered)
        with self.assertRaisesRegex(ValueError, "Completed response"):
            validate(d={"response_tokens": 9})

    def test_runtime_guard_rejects_precision_thinking_and_missing_model_weights(self):
        value = {
            "precision": "float32_torch", "enable_thinking": False,
            "architecture": "qwen3_5_text", "student_vocab": 248320,
            "loading": {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []},
            "load_seconds": 15.,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(value))
            self.assertNotIn("load_seconds", require_runtime(path, "float32_torch"))
            for change in ({"precision": "bf16"}, {"enable_thinking": True},
                           {"loading": {"missing_keys": ["weight"]}}):
                path.write_text(json.dumps({**value, **change}))
                with self.assertRaisesRegex(ValueError, "Runtime/model"):
                    require_runtime(path, "float32_torch")


if __name__ == "__main__":
    unittest.main()
