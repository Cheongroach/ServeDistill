import copy
import json
from pathlib import Path
import tempfile
import unittest

from servedistill.profiling import build_command, runtime_validity, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ProfileContracts(unittest.TestCase):
    def test_ncu_capture_is_bounded_and_warmup_excluded(self):
        config = json.loads((ROOT / "configs/monitoring/ncu.json").read_text())
        argv = build_command(config, ["python", "train.py"], Path("/tmp/trace"), "ncu")
        self.assertEqual(argv[argv.index("--profile-from-start") + 1], "off")
        self.assertEqual(argv[argv.index("--launch-count") + 1], "8")
        self.assertEqual(argv[argv.index("--nvtx-include") + 1], "learner_forward/")
        unsafe = copy.deepcopy(config)
        unsafe["options"]["launch_count"] = 0
        with self.assertRaises(ValueError):
            validate_config(unsafe)

    def test_empty_or_wrong_window_cannot_pass(self):
        config = json.loads((ROOT / "configs/monitoring/baseline.json").read_text())
        gates = json.loads((ROOT / "configs/validity/runtime.json").read_text())
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            train = root / "application/train"
            train.mkdir(parents=True)
            (train / "restoration-0160.json").write_text('{"passed":true}')
            self.assertFalse(runtime_validity(root, config, gates)["passed"])
            row = dict(step=162, rollout_version=161, loss=1.0,
                       gradient_norm=0.1, parameter_delta_l2=0.1, behavior_max_error=0.01)
            (train / "metrics.jsonl").write_text(json.dumps(row) + "\n")
            self.assertTrue(runtime_validity(root, config, gates)["passed"])
            row["behavior_max_error"] = 0.151
            (train / "metrics.jsonl").write_text(json.dumps(row) + "\n")
            self.assertFalse(runtime_validity(root, config, gates)["passed"])


if __name__ == "__main__":
    unittest.main()
