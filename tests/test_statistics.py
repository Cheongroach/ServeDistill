import unittest
import numpy as np
from servedistill.statistics import bootstrap_metric, compare, metric


class PairedStatisticsTests(unittest.TestCase):
    def test_vectorized_resampling_matches_direct_nonlinear_metrics(self):
        collections = [
            [{"task": "esci_classification", "target_class": truth, "predicted_class": predicted}
             for truth, predicted in (("E", "E"), ("S", "E"), ("C", "INVALID"), ("I", "I"))],
            [{"task": "esci_ranking", "ndcg": value, "ndcg_eligible": eligible}
             for value, eligible in ((0, False), (0.5, True), (1, True), (0.7, True))],
            [{"success": flag} for flag in (True, False, False, True)],
        ]
        indices = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [3, 2, 2, 1]])
        for rows in collections:
            direct = [metric([rows[i] for i in sample]) for sample in indices]
            np.testing.assert_allclose(bootstrap_metric(rows, indices), direct)

    def test_pairing_rejects_changed_task_and_duplicate_ids(self):
        row = {"sample_id": "a", "input_sha256": "hash", "domain": "math", "success": True}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare([row, row], [[row, row]], 20)
        with self.assertRaisesRegex(ValueError, "Task/label changed"):
            compare([row], [[{**row, "input_sha256": "other"}]], 20)

    def test_identical_model_has_zero_paired_interval(self):
        rows = [{"sample_id": str(i), "input_sha256": str(i),
                 "domain": "math", "success": i % 2 == 0} for i in range(30)]
        result = compare(rows, [rows, rows], 100)
        self.assertEqual(result["slices"]["math"]["paired_task_ci95"], [0.0, 0.0])
        self.assertEqual(result["slices"]["math"]["n_independent_tasks"], 30)


if __name__ == "__main__":
    unittest.main()
