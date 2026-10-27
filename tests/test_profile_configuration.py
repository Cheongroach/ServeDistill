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

    def test_diagnostic_allocator_override_reaches_child_without_changing_parent(self):
        from servedistill.profiling import application_environment
        config = json.loads((ROOT / "configs/monitoring/sanitizer_no_expandable.json").read_text())
        validate_config(config)
        parent = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                  "PYTORCH_ALLOC_CONF": "expandable_segments:True", "OTHER": "kept"}
        child = application_environment(config, parent)
        self.assertEqual(child["PYTORCH_ALLOC_CONF"], "expandable_segments:False")
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", child)
        self.assertEqual(parent["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(child["OTHER"], "kept")
        self.assertEqual(application_environment({}, parent), parent)
        config["cuda_allocator"] = "typo"
        with self.assertRaisesRegex(ValueError, "CUDA allocator"):
            validate_config(config)

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

    def test_actual_ncu_raw_wide_layout_preserves_units_and_rejects_missing_counter(self):
        config = json.loads((ROOT / "configs/monitoring/ncu.json").read_text())
        audit = {"raw_metrics": {"returncode": 0}}
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            path = root / "ncu-metrics.csv"
            header = ('"ID","Kernel Name","gpu__time_duration.sum",'
                      '"sm__throughput.avg.pct_of_peak_sustained_elapsed",'
                      '"dram__throughput.avg.pct_of_peak_sustained_elapsed"\n')
            units = '"","","us","%","%"\n'
            data = '"0","copy<4>","3.456000","0.260615","0.142045"\n'
            path.write_text(header + units + data)
            result = report_validity(root, config, audit)
            self.assertTrue(result["passed"])
            self.assertEqual(result["kernel_launches"], 1)
            self.assertEqual(result["metric_rows"], 3)
            path.write_text(header + units + data.replace('"0.142045"', '"N/A"'))
            self.assertFalse(report_validity(root, config, audit)["passed"])
            path.write_text(header + units.replace('"us"', '"%"') + data)
            self.assertFalse(report_validity(root, config, audit)["passed"])
            path.write_text(header + units + data + data)
            self.assertFalse(report_validity(root, config, audit)["passed"])
            path.write_text(header + data)
            with self.assertRaisesRegex(ValueError, "units row"):
                report_validity(root, config, audit)

    def test_ncu_sections_require_each_sections_quantitative_evidence(self):
        from servedistill.profiling import NCU_SECTION_METRICS
        import csv
        config = json.loads((ROOT / "configs/monitoring/ncu_gemm_no_cudnn.json").read_text())
        names = list(dict.fromkeys(name for section in config["options"]["sections"]
                                   for name in NCU_SECTION_METRICS[section]))
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            path = root / "ncu-metrics.csv"
            with path.open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ID", "Kernel Name", *names])
                units = {
                    "gpu__time_duration.sum": "ms",
                    "dram__bytes.sum.per_second": "Gbyte/s",
                    "launch__registers_per_thread": "register/thread",
                    "launch__shared_mem_per_block": "Kbyte/block",
                    "launch__occupancy_limit_registers": "block",
                }
                writer.writerow(["", "", *(units.get(name, "%") for name in names)])
                writer.writerow(["0", "gemm", *([1.] * len(names))])
            audit = {"raw_metrics": {"returncode": 0}}
            self.assertTrue(report_validity(root, config, audit)["passed"])
            config["options"]["sections"].append("UnknownSection")
            self.assertFalse(report_validity(root, config, audit)["passed"])
            config["options"]["sections"].pop()
            path.write_text(path.read_text().replace(
                "lts__t_sector_hit_rate.pct", "unrequested__metric"))
            self.assertFalse(report_validity(root, config, audit)["passed"])

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
