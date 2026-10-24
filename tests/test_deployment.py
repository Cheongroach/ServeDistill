import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from servedistill.cascade_benchmark import protocol_valid, request
from servedistill.selection import DOMAINS, freeze


class SelectionTests(unittest.TestCase):
    def snapshot(self):
        snapshot = {}
        for index in range(12):
            for seed in (17, 29):
                report = {
                    "business_macro": 0.5 + index * 0.01,
                    "general_macro": 0.5,
                    "domains": {domain: {"selection_score": 0.5} for domain in DOMAINS},
                }
                snapshot[f"method{index:02d}_seed{seed}"] = {
                    "completed": True, "steps": 80, "uses_test": False,
                    "selection": {"step": 40, "path": "checkpoint-0040",
                                  "dev_score": 0.8 * report["business_macro"] + 0.2 * report["general_macro"]},
                    "adapter_file_sha256": "adapter",
                    "dev_task_sha256": "same", "development": report,
                    "baseline": copy.deepcopy(report),
                }
        return snapshot

    def test_final_metrics_cannot_change_selection_and_output_cannot_be_replaced(self):
        snapshot = self.snapshot()
        with tempfile.TemporaryDirectory() as folder:
            first = freeze(snapshot, Path(folder) / "first.json")
            for run in snapshot.values():
                run["final_test_score"] = 1 if run["development"]["business_macro"] == 0.5 else 0
            second = freeze(snapshot, Path(folder) / "second.json")
            self.assertEqual(first["label"], second["label"])
            self.assertEqual(first["label"], "method11_seed17")
            with self.assertRaises(FileExistsError):
                freeze(snapshot, Path(folder) / "first.json")

    def test_missing_seed_or_incomplete_run_cannot_win(self):
        snapshot = self.snapshot()
        del snapshot["method11_seed29"]
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            freeze(snapshot, Path(folder) / "selection.json")
        snapshot = self.snapshot()
        snapshot["method11_seed29"]["completed"] = False
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            freeze(snapshot, Path(folder) / "selection.json")

    def test_domain_regression_excludes_highest_scoring_method(self):
        snapshot = self.snapshot()
        for seed in (17, 29):
            snapshot[f"method11_seed{seed}"]["development"]["domains"]["math"]["selection_score"] = 0.1
        with tempfile.TemporaryDirectory() as folder:
            result = freeze(snapshot, Path(folder) / "selection.json")
        self.assertEqual(result["method"], "method10")
        self.assertTrue(result["deployment_gate_passed"])


class ProtocolFallbackTests(unittest.IsolatedAsyncioTestCase):
    def test_ranking_router_uses_ids_without_label_values(self):
        class KeysOnly(dict):
            def values(self):
                raise AssertionError("Labels must not be accessed")
            def __getitem__(self, key):
                raise AssertionError("Labels must not be accessed")
        row = {"task": "esci_ranking", "labels": KeysOnly(P0="E", P1="I")}
        self.assertTrue(protocol_valid(row, '{"ranking":["P1","P0"]}'))
        self.assertFalse(protocol_valid(row, '{"ranking":["P0","P0"]}'))

    async def test_valid_but_wrong_answer_does_not_get_oracle_fallback(self):
        args = SimpleNamespace(student_endpoint="student", teacher_endpoint="teacher",
                               student_model="s", teacher_model="t", max_new_tokens=128)
        record = {"text": '{"label":"I"}', "truncated": False,
                  "latency_seconds": 0.1, "server_response_id": "student-id", "success": False}
        with patch("servedistill.cascade_benchmark.stream_request", new=AsyncMock(return_value=record)) as call:
            result = await request(None, args, {"task": "esci_classification"})
        self.assertEqual(call.await_count, 1)
        self.assertFalse(result["fallback"])
        self.assertFalse(result["success"])

    async def test_malformed_answer_is_replaced_after_teacher_call(self):
        args = SimpleNamespace(student_endpoint="student", teacher_endpoint="teacher",
                               student_model="s", teacher_model="t", max_new_tokens=128)
        before = {"text": "bad JSON", "truncated": False,
                  "latency_seconds": 0.1, "server_response_id": "student-id", "success": False}
        after = {"text": '{"label":"E"}', "truncated": False,
                 "latency_seconds": 0.2, "server_response_id": "teacher-id", "success": True}
        with patch("servedistill.cascade_benchmark.stream_request",
                   new=AsyncMock(side_effect=[before, after])) as call:
            result = await request(None, args, {"task": "esci_classification", "teacher_system": "fixed"})
        self.assertEqual(call.await_count, 2)
        self.assertEqual(call.await_args.args[3]["system"], "fixed")
        self.assertTrue(result["fallback"])
        self.assertTrue(result["success"])
        self.assertEqual(result["teacher_response_id"], "teacher-id")


if __name__ == "__main__":
    unittest.main()
