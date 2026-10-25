import copy
import unittest

import numpy as np
import torch

from servedistill.expanded_statistics import performance, quality
from scripts.check_expanded_numerics import stable_comparison


class ExpandedStatisticsTests(unittest.TestCase):
    def timings(self):
        return [
            {"worker_index": w, "bank": b, "microbatch": m, "repetition": r,
             "warmup": False, "profiled": False, "parent_reset": True,
             "learner_update_seconds": (b + 1) * 8 / m,
             "effective_response_tokens": 100 * (b + 1), "response_width": 20,
             "peak_allocated_bytes": m * 2**30, "peak_reserved_bytes": m * 2**30}
            for w in range(2) for b in range(2)
            for m in (1, 2, 4, 8) for r in range(3)]

    def test_bank_statistic_excludes_warmup_and_profiler(self):
        rows = self.timings()
        rows += [{**rows[0], "warmup": True, "learner_update_seconds": 900},
                 {**rows[0], "profiled": True, "learner_update_seconds": 900}]
        result = performance(rows, banks_per_worker=2)
        self.assertEqual(result["n_independent_banks"], 4)
        self.assertEqual(result["n_formal_measurements"], 48)
        self.assertEqual(result["comparisons"]["4"]["paired_bank_ci95"], [4., 4.])
        self.assertFalse(result["comparisons"]["1"]["stable_improvement"])

    def test_missing_duplicate_or_changed_trajectories_fail(self):
        rows = self.timings()
        for altered in (rows[:-1], [*rows, rows[0]]):
            with self.assertRaises(ValueError):
                performance(altered, banks_per_worker=2)
        altered = copy.deepcopy(rows)
        altered[0]["effective_response_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "trajectory"):
            performance(altered, banks_per_worker=2)

    def evaluations(self, invert=False):
        return [
            [{"sample_id": str(i), "input_sha256": str(i), "domain": "math",
              "success": bool((i + s) % 2) ^ invert, "json_valid": True,
              "truncated": False}
             for i in range(40)] for s in range(4)]

    def test_seed_pairing_does_not_inflate_query_count(self):
        rows = self.evaluations()
        result = quality(rows, copy.deepcopy(rows), repetitions=100)
        part = result["slices"]["math"]
        self.assertEqual(part["n_independent_queries"], 40)
        self.assertEqual(part["n_paired_training_seeds"], 4)
        self.assertEqual(part["paired_query_ci95"], [0., 0.])
        self.assertEqual(part["noninferiority"], "supported")

    def test_query_draws_are_shared_across_seeds(self):
        # Opposite deltas cancel for every query, including every bootstrap draw.
        result = quality(self.evaluations(), self.evaluations(True), repetitions=100)
        self.assertEqual(result["slices"]["math"]["paired_query_ci95"], [0., 0.])

    def test_macro_f1_and_label_pairing_remain_intact(self):
        rows = [[{"sample_id": str(i), "input_sha256": str(i), "domain": "relevance",
                  "task": "esci_classification", "target_class": label,
                  "predicted_class": label}
                 for i, label in enumerate(("E", "S", "C", "I"))] for _ in range(4)]
        candidates = copy.deepcopy(rows)
        for part in candidates:
            for row in part:
                row["predicted_class"] = "INVALID"
        result = quality(rows, candidates, repetitions=100)
        self.assertEqual(result["slices"]["relevance"]["delta"], -1.)
        self.assertEqual(result["slices"]["relevance"]["noninferiority"], "inferior")
        candidates[0][0]["target_class"] = "I"
        with self.assertRaisesRegex(ValueError, "label changed"):
            quality(rows, candidates, repetitions=100)

    def test_repeated_query_requires_cluster_resampling(self):
        rows = self.evaluations()
        for part in rows:
            for row in part:
                row["query_group_sha256"] = "same"
        with self.assertRaisesRegex(ValueError, "cluster resampling"):
            quality(rows, rows, repetitions=100)

    def test_fp64_reduction_preserves_parallel_and_orthogonal_vectors(self):
        a = torch.linspace(-1., 1., 200000)
        same = stable_comparison(a, a.clone())
        self.assertAlmostEqual(same["gradient_cosine_fp64"], 1., places=12)
        self.assertEqual(same["gradient_relative_l2_fp64"], 0.)
        orthogonal = stable_comparison(torch.tensor([1., 1.]), torch.tensor([1., -1.]))
        self.assertEqual(orthogonal["gradient_cosine_fp64"], 0.)

    def test_length_quartiles_resample_banks(self):
        rows = self.timings()
        for row in rows:
            row["mean_prompt_token_length"] = 100 * row["worker_index"] + row["bank"]
        result = performance(rows, banks_per_worker=2)
        self.assertEqual(len(result["prompt_length_quartiles"]), 4)
        for bucket in result["prompt_length_quartiles"].values():
            self.assertEqual(bucket["n_independent_banks"], 1)
            self.assertEqual(bucket["paired_bank_ci95"], [4., 4.])


if __name__ == "__main__":
    unittest.main()
