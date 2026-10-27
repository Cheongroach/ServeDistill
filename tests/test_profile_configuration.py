import copy
import json
from pathlib import Path
import tempfile
import unittest

from servedistill.profiling import build_command, report_validity, runtime_validity, validate_config


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

    def test_ncu_requires_actual_kernel_counters_and_units(self):
        config = json.loads((ROOT / "configs/monitoring/ncu.json").read_text())
        audit = {"raw_metrics": {"returncode": 0}}
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            path = root / "ncu-metrics.csv"
            header = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            path.write_text(header)
            self.assertFalse(report_validity(root, config, audit)["passed"])
            rows = [
                '"0","gemm","gpu__time_duration.sum","us","12.5"\n',
                '"0","gemm","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","50"\n',
                '"0","gemm","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","20"\n',
            ]
            path.write_text("CLI preamble\n" + header + "".join(rows))
            self.assertTrue(report_validity(root, config, audit)["passed"])
            path.write_text(header + "".join(rows).replace('"50"', '"N/A"'))
            self.assertFalse(report_validity(root, config, audit)["passed"])
            path.write_text(header + "".join(rows[:-1]))
            self.assertFalse(report_validity(root, config, audit)["passed"])
            path.write_text(header + "".join(rows))
            self.assertFalse(report_validity(
                root, config, {"raw_metrics": {"returncode": 1}})["passed"])

    def test_torch_cpu_only_trace_cannot_pass_gpu_capture(self):
        config = json.loads((ROOT / "configs/monitoring/torch.json").read_text())
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            path = root / "application/observation/torch-trace.json"
            path.parent.mkdir(parents=True)
            events = [{"name": "learner_forward", "ph": "X", "dur": 10}]
            path.write_text(json.dumps({"traceEvents": events}))
            self.assertFalse(report_validity(root, config, {})["passed"])
            events.append({"name": "gemm", "cat": "kernel", "ph": "X", "dur": 5})
            path.write_text(json.dumps({"traceEvents": events}))
            self.assertTrue(report_validity(root, config, {})["passed"])

    def test_nsys_requires_real_kernel_api_and_nvtx_rows(self):
        config = json.loads((ROOT / "configs/monitoring/nsys.json").read_text())
        audit = {report: {"returncode": 0}
                 for report in ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum")}
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "cuda_gpu_kern_sum.csv").write_text(
                "CLI preamble\nTotal Time (ns),Instances,Name\n100,1,gemm\n")
            (root / "cuda_api_sum.csv").write_text(
                "Total Time (ns),Num Calls,Name\n200,2,cudaLaunchKernel\n")
            path = root / "nvtx_sum.csv"
            path.write_text("Total Time (ns),Instances,Range\n300,1,:learner_forward\n")
            self.assertTrue(report_validity(root, config, audit)["passed"])
            path.write_text("Total Time (ns),Instances,Range\n")
            self.assertFalse(report_validity(root, config, audit)["passed"])


if __name__ == "__main__":
    unittest.main()
