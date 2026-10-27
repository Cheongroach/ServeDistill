import unittest

import torch

from scripts.compare_profile_states import compare_tensors


class ProfileNumerics(unittest.TestCase):
    def test_drift_is_not_exact_equality_and_missing_tensor_fails(self):
        tolerance = {"rtol": 1e-5, "atol": 1e-6}
        reference = {"w": torch.tensor([1.0, 0.0])}
        small = {"w": torch.tensor([1.0, 5e-7])}
        result = compare_tensors(reference, small, tolerance)
        self.assertTrue(result["passed"])
        self.assertFalse(result["tensors"][0]["exact"])
        self.assertFalse(compare_tensors(reference, {}, tolerance)["passed"])
        self.assertFalse(compare_tensors({}, {}, tolerance)["passed"])
        self.assertFalse(compare_tensors(
            reference, {"w": torch.tensor([1.0, 0.01])}, tolerance)["passed"])
        self.assertFalse(compare_tensors(
            reference, {"w": torch.tensor([1.0, float("nan")])}, tolerance)["passed"])


if __name__ == "__main__":
    unittest.main()
