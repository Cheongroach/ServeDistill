"""Auditable profiler commands and Linux host sampling; no training dependencies."""
from __future__ import annotations

import hashlib
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
    if config["backend"] == "ncu":
        options = config["options"]
        if options["replay_mode"] != "kernel" or not options["nvtx_include"]:
            raise ValueError("NCU requires a bounded kernel replay range")
        if not isinstance(options["launch_count"], int) or not 1 <= options["launch_count"] <= 100:
            raise ValueError("NCU launch count must be between 1 and 100")
        if not options.get("metrics") and not options.get("sections"):
            raise ValueError("NCU needs metrics or sections")


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
