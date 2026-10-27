"""Auditable profiler commands and Linux host sampling; no training dependencies."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time


BACKENDS = ("none", "nsys", "ncu", "torch", "perf", "sanitizer")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def locate(name: str, tools_root: Path | None = None) -> str | None:
    override = os.environ.get("SERVEDISTILL_" + name.upper().replace("-", "_"))
    if override:
        if not Path(override).is_file() or not os.access(override, os.X_OK):
            raise ValueError("Profiler override is not executable: " + name)
        return override
    existing = shutil.which(name)
    if existing:
        return existing
    roots = [Path("/opt/nvidia"), Path("/usr/local/cuda")]
    if tools_root:
        roots.insert(0, tools_root)
    for root in roots:
        if root.exists():
            for match in sorted(root.glob(f"**/{name}")):
                if match.is_file() and os.access(match, os.X_OK):
                    return str(match.resolve())
    return None


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("backend") not in BACKENDS:
        raise ValueError("Unsupported profiler configuration")
    for key in ("timeout_seconds", "first_update", "last_update", "microbatch",
                "gpu_poll_ms", "host_poll_seconds"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Positive finite profiler option required: " + key)
    if config["last_update"] < config["first_update"]:
        raise ValueError("Observation window is reversed")
    if config.get("attention_backend", "default") not in ("default", "no_cudnn"):
        raise ValueError("Unknown diagnostic attention backend")
    if config.get("cuda_allocator", "default") not in ("default", "no_expandable"):
        raise ValueError("Unknown diagnostic CUDA allocator")
    if "rollout_batch" in config:
        value = config["rollout_batch"]
        if type(value) is not int or value < 1:
            raise ValueError("Diagnostic rollout_batch must be a positive integer")
    if config["backend"] == "ncu":
        options = config["options"]
        if options["replay_mode"] != "kernel" or not options["nvtx_include"]:
            raise ValueError("NCU requires a bounded kernel replay range")
        if not isinstance(options["launch_count"], int) or not 1 <= options["launch_count"] <= 100:
            raise ValueError("NCU launch count must be between 1 and 100")
        if not options.get("metrics") and not options.get("sections"):
            raise ValueError("NCU needs metrics or sections")


def application_environment(config: dict, environment: dict) -> dict:
    """Apply an explicit diagnostic allocator choice only to the child process."""
    result = dict(environment)
    if config.get("cuda_allocator", "default") == "no_expandable":
        result.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        result["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
    return result


def build_command(config: dict, application: list[str], output: Path,
                  binary: str | None = None) -> list[str]:
    validate_config(config)
    backend, options = config["backend"], config["options"]
    if backend in ("none", "torch"):
        return application
    if binary is None:
        raise FileNotFoundError("Profiler executable is required: " + backend)
    if backend == "nsys":
        return [binary, "profile", "--trace=" + options["trace"],
                "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                "--sample=" + options["sample"], "--cpuctxsw=" + options["cpuctxsw"],
                "--force-overwrite=true", "--output=" + str(output / "capture"), *application]
    if backend == "ncu":
        argv = [binary, "--profile-from-start", "off", "--nvtx", "--nvtx-include",
                options["nvtx_include"], "--replay-mode", options["replay_mode"],
                "--launch-count", str(options["launch_count"])]
        if options.get("metrics"):
            argv += ["--metrics", ",".join(options["metrics"])]
        for section in options.get("sections", []):
            argv += ["--section", section]
        if options.get("kernel_name"):
            argv += ["--kernel-name", options["kernel_name"]]
        return [*argv, "--force-overwrite", "--export", str(output / "capture"), *application]
    if backend == "perf":
        return [binary, "stat", "-x", ",", "-o", str(output / "perf-stat.csv"),
                "-e", ",".join(options["events"]), "--", *application]
    if options["tool"] not in ("memcheck", "racecheck", "initcheck", "synccheck"):
        raise ValueError("Unsupported CUDA sanitizer tool")
    return [binary, "--tool", options["tool"], "--error-exitcode", "86",
            "--log-file", str(output / "sanitizer.log"), *application]


class HostSampler:
    """Sample host-wide and process-tree CPU/IO counters; units remain explicit."""

    def __init__(self, path: Path, interval: float, root_pid: int):
        self.path, self.interval, self.root_pid = path, interval, root_pid
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self) -> dict:
        processes, parents = {}, {}
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                raw = (proc / "stat").read_text()
                fields = raw[raw.rfind(")") + 2:].split()
                pid = int(proc.name)
                parents[pid] = int(fields[1])
                processes[pid] = {
                    "pid": pid, "ppid": parents[pid],
                    "user_cpu_ticks": int(fields[11]), "system_cpu_ticks": int(fields[12]),
                    "minor_faults": int(fields[7]), "major_faults": int(fields[9]),
                    "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                    "io": dict(line.split(": ", 1) for line in (proc / "io").read_text().splitlines()),
                }
            except (OSError, ValueError, IndexError):
                continue
        owned = {self.root_pid}
        while True:
            expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
            if expanded == owned:
                break
            owned = expanded
        return {
            "unix_seconds": time.time(), "monotonic_ns": time.monotonic_ns(),
            "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
            "load_average": os.getloadavg(),
            "host_meminfo": Path("/proc/meminfo").read_text(),
            "host_cpu_disk_counters": Path("/proc/stat").read_text(),
            "host_diskstats": Path("/proc/diskstats").read_text(),
            "processes": [processes[p] for p in sorted(owned) if p in processes],
            "boundary": "host totals may include other tenants; process counters cover the visible tree",
        }

    def run(self) -> None:
        with self.path.open("w") as stream:
            while not self.stop_event.is_set():
                try:
                    record = self.snapshot()
                except OSError as error:
                    record = {"sampling_error": str(error), "unix_seconds": time.time()}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2)


def runtime_validity(root: Path, config: dict, gates: dict) -> dict:
    train = root / "application/train"
    paths = list(train.glob("restoration-*.json"))
    restored = len(paths) == 1 and json.loads(paths[0].read_text()).get("passed") is True
    metrics_path = train / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    selected = [r for r in metrics if config["first_update"] <= r["step"] <= config["last_update"]]
    expected_steps = list(range(config["first_update"], config["last_update"] + 1))
    checks = {
        "cold_restoration": restored,
        "window_complete": [r["step"] for r in selected] == expected_steps,
        "finite_loss_gradient_delta": bool(selected) and all(
            math.isfinite(r[key]) for r in selected
            for key in ("loss", "gradient_norm", "parameter_delta_l2")),
        "nonzero_parameter_update": bool(selected) and all(r["parameter_delta_l2"] > 0 for r in selected),
        "behavior_probability": bool(selected) and all(
            r["behavior_max_error"] <= gates["behavior_max_abs_error"] for r in selected),
        "rollout_version": bool(selected) and all(r["rollout_version"] == r["step"] - 1 for r in selected),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "observed_steps": [r["step"] for r in selected],
            "teacher_frozen_and_finite_parameters": "enforced in train.apply_rollout; failure is fatal",
            "boundary": gates["quality_boundary"]}


def csv_records(path: Path, required: set[str]) -> list[dict]:
    """Ignore CLI preambles but require the real exported CSV header."""
    if not path.is_file():
        return []
    lines = path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if required.issubset(fields):
            return list(csv.DictReader(io.StringIO("\n".join(lines[index:]))))
    return []


def metric_number(value: str | None) -> float | None:
    try:
        result = float((value or "").replace(",", "").strip())
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def ncu_metric_records(path: Path) -> list[dict]:
    """Normalize long CSV and NCU raw wide CSV without changing source bytes."""
    long = csv_records(path, {"ID", "Kernel Name", "Metric Name", "Metric Value", "Metric Unit"})
    if long:
        return long
    wide = csv_records(path, {"ID", "Kernel Name"})
    if not wide:
        return []
    units = [r for r in wide if not r.get("ID") and not r.get("Kernel Name")]
    if len(units) != 1:
        raise ValueError("NCU wide CSV requires exactly one units row")
    unit_row = units[0]
    return [
        {"ID": row["ID"], "Kernel Name": row["Kernel Name"],
         "Metric Name": name, "Metric Value": value, "Metric Unit": unit_row.get(name)}
        for row in wide if row.get("Kernel Name")
        for name, value in row.items() if name and "__" in name
    ]


NCU_SECTION_METRICS = {
    "SpeedOfLight": ("gpu__time_duration.sum",
                     "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    "MemoryWorkloadAnalysis": ("dram__bytes.sum.per_second", "lts__t_sector_hit_rate.pct"),
    "LaunchStats": ("launch__registers_per_thread", "launch__shared_mem_per_block"),
    "Occupancy": ("sm__warps_active.avg.pct_of_peak_sustained_active",
                  "launch__occupancy_limit_registers"),
}


def report_validity(root: Path, config: dict, audit: dict) -> dict:
    """A nonempty report alone does not establish usable profiler evidence."""
    backend = config["backend"]
    checks, details = {}, {}
    if backend == "ncu":
        rows = ncu_metric_records(root / "ncu-metrics.csv")
        options = config["options"]
        requested = list(options.get("metrics", []))
        sections = options.get("sections", [])
        unknown_sections = [name for name in sections if name not in NCU_SECTION_METRICS]
        for section in sections:
            requested.extend(NCU_SECTION_METRICS.get(section, ()))
        requested = list(dict.fromkeys(requested))
        kernels = {}
        duplicate_metric = False
        for row in rows:
            if not row.get("Kernel Name") or not row.get("ID"):
                continue
            key = (row.get("ID"), row["Kernel Name"])
            duplicate_metric |= row.get("Metric Name") in kernels.get(key, {})
            kernels.setdefault(key, {})[row.get("Metric Name")] = row
        checks["raw_export_succeeded"] = audit.get("raw_metrics", {}).get("returncode") == 0
        checks["kernels_captured"] = bool(kernels)
        checks["unique_kernel_metric_rows"] = not duplicate_metric
        checks["bounded_launch_count"] = 0 < len(kernels) <= options["launch_count"]
        checks["declared_section_validation_supported"] = not unknown_sections
        checks["all_requested_metrics_present"] = bool(kernels) and all(
            all(name in metrics for name in requested) for metrics in kernels.values())
        values = [row for metrics in kernels.values() for name, row in metrics.items()
                  if name in requested]
        checks["finite_values_with_units"] = bool(values) and all(
            row.get("Metric Unit") and metric_number(row.get("Metric Value")) is not None
            and metric_number(row.get("Metric Value")) >= 0 for row in values)
        checks["positive_kernel_duration"] = bool(kernels) and all(
            metric_number(metrics.get("gpu__time_duration.sum", {}).get("Metric Value"))
            is not None
            and metric_number(metrics["gpu__time_duration.sum"]["Metric Value"]) > 0
            and metrics["gpu__time_duration.sum"]["Metric Unit"] in ("ns", "us", "ms", "s")
            for metrics in kernels.values())
        details = {"kernel_launches": len(kernels), "metric_rows": len(values),
                   "requested_metrics": requested, "unknown_sections": unknown_sections,
                   "section_validation_scope": "Named quantitative evidence per requested section; "
                                               "not exhaustive validation of every derived counter."}
    elif backend == "nsys":
        for report in ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum"):
            name_key = "Range" if report == "nvtx_sum" else "Name"
            rows = csv_records(root / (report + ".csv"), {name_key})
            checks[report + "_export"] = audit.get(report, {}).get("returncode") == 0
            checks[report + "_has_data"] = any(
                row.get(name_key) and any(
                    key and ("Instances" in key or "Calls" in key or "Range Count" in key)
                    and (metric_number(value) or 0) > 0
                    for key, value in row.items())
                and (metric_number(row.get("Total Time (ns)")) or 0) > 0 for row in rows)
        nvtx = csv_records(root / "nvtx_sum.csv", {"Range"})
        checks["learner_range_present"] = any(
            "learner_forward" in row.get("Range", "") for row in nvtx)
    elif backend == "torch":
        path = root / "application/observation/torch-trace.json"
        events = json.loads(path.read_text()).get("traceEvents", []) if path.is_file() else []
        checks["gpu_kernel_events"] = any(
            event.get("cat") == "kernel" and event.get("ph") == "X"
            and (metric_number(str(event.get("dur", ""))) or 0) > 0 for event in events)
        checks["learner_range_present"] = any(
            event.get("name") == "learner_forward" for event in events)
    elif backend == "perf":
        path = root / "perf-stat.csv"
        rows = list(csv.reader(path.read_text().splitlines())) if path.is_file() else []
        present = {row[2]: row[0] for row in rows if len(row) >= 3}
        checks["all_events_counted"] = all(
            metric_number(present.get(event)) is not None
            for event in config["options"]["events"])
    elif backend == "sanitizer":
        path = root / "sanitizer.log"
        checks["zero_error_summary"] = (
            path.is_file() and "ERROR SUMMARY: 0 errors" in path.read_text())
    return {"passed": all(checks.values()), "checks": checks, **details,
            "boundary": "Captured diagnostics only; representative kernel selection is a separate check."}
