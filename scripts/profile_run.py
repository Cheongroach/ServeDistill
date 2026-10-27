"""Run one bounded diagnostic clone and retain commands, telemetry and validity."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from servedistill.profiling import (
    HostSampler, application_environment, build_command, digest, locate,
    report_validity, runtime_validity, validate_config,
)


REPO = Path(__file__).resolve().parents[1]


def run_log(argv, target, timeout=120):
    with target.open("w") as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    return {"argv": argv, "returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--validity", type=Path, default=REPO / "configs/validity/runtime.json")
    parser.add_argument("--tools-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    for name in ("student", "teacher", "data", "config", "resume", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-update", type=int)
    parser.add_argument("--last-update", type=int)
    args = parser.parse_args()
    config = json.loads(args.profile.read_text())
    for key in ("first_update", "last_update"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    validate_config(config)
    backend = config["backend"]
    name = "compute-sanitizer" if backend == "sanitizer" else backend
    binary = None if backend in ("none", "torch") else locate(name, args.tools_root)
    application = [sys.executable, str(REPO / "scripts/profile_training.py")]
    for key in ("student", "teacher", "data", "config", "resume"):
        application.extend(["--" + key, str(getattr(args, key).resolve())])
    application.extend([
        "--output", str(args.output.resolve() / "application"),
        "--backend", backend if backend in ("nsys", "ncu", "torch") else "none",
        "--first-update", str(config["first_update"]),
        "--last-update", str(config["last_update"]), "--microbatch", str(config["microbatch"]),
        "--attention-backend", config.get("attention_backend", "default"),
    ])
    if "rollout_batch" in config:
        application.extend(["--rollout-batch", str(config["rollout_batch"])])
    if backend == "torch":
        for key, value in config["options"].items():
            if value:
                application.append("--" + key.replace("_", "-"))
    argv = build_command(config, application, args.output.resolve(), binary)
    if args.dry_run:
        print(json.dumps({"command": argv, "configuration": config}, indent=2))
        return
    checkpoint_step = json.loads((args.resume / "state.json").read_text())["step"]
    if config["first_update"] <= checkpoint_step + 1:
        raise ValueError("Require one complete warmup update after cold restoration")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = application_environment(config, os.environ)
    manifest = {
        "configuration": config, "command": argv, "checkpoint_step": checkpoint_step,
        "profile_configuration_sha256": digest(args.profile),
        "validity_configuration_sha256": digest(args.validity),
        "training_configuration_sha256": digest(args.config),
        "input_files": {str(p): digest(p) for p in (
            args.resume / "state.json", args.resume / "training.pt",
            args.resume / "adapter_model.safetensors", args.data / "train.jsonl", args.data / "dev.jsonl")},
        "source_files": {str(p.relative_to(REPO)): digest(p) for directory in ("servedistill", "scripts")
                         for p in sorted((REPO / directory).glob("*.py"))},
        "allocator_environment": {key: environment.get(key) for key in (
            "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
        "started_unix_seconds": time.time(), "profiler_times_eligible_for_speed_claim": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audit = {}
    if binary:
        audit["version"] = run_log([binary, "--version"], args.output / "tool-version.txt")
        if backend == "ncu":
            audit["metrics"] = run_log([binary, "--query-metrics"], args.output / "available-metrics.txt")
        if backend == "nsys":
            audit["environment"] = run_log([binary, "status", "--environment"], args.output / "tool-environment.txt")
    telemetry, gpu_stream = None, None
    smi = locate("nvidia-smi")
    if smi:
        audit["gpu"] = run_log([smi, "-q"], args.output / "gpu-details.txt")
        gpu_stream = (args.output / "gpu-telemetry.csv").open("w")
        telemetry = subprocess.Popen([
            smi, "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,"
                 "memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv", "--loop-ms=" + str(config["gpu_poll_ms"]),
        ], stdout=gpu_stream, stderr=subprocess.STDOUT)
    returncode = 125
    try:
        with (args.output / "application.log").open("w") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=REPO, env=environment, start_new_session=True)
            with HostSampler(args.output / "host-telemetry.jsonl", config["host_poll_seconds"], process.pid):
                try:
                    returncode = process.wait(timeout=config["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    returncode = 124
        if returncode == 0 and backend == "nsys":
            for report in config["options"]["reports"]:
                audit[report] = run_log(
                    [binary, "stats", "--report=" + report, "--format=csv",
                     str(args.output / "capture.nsys-rep")], args.output / (report + ".csv"), 180)
        if returncode == 0 and backend == "ncu":
            audit["raw_metrics"] = run_log([
                binary, "--import", str(args.output / "capture.ncu-rep"), "--csv", "--page", "raw",
            ], args.output / "ncu-metrics.csv", 180)
    finally:
        if telemetry:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
            gpu_stream.close()
        audit["application_returncode"] = returncode
        audit["ended_unix_seconds"] = time.time()
        (args.output / "tool-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        validity = runtime_validity(args.output, config, json.loads(args.validity.read_text()))
        validity["application_succeeded"] = returncode == 0
        reports = {"nsys": "capture.nsys-rep", "ncu": "capture.ncu-rep",
                   "torch": "application/observation/torch-trace.json",
                   "perf": "perf-stat.csv", "sanitizer": "sanitizer.log"}
        expected = args.output / reports[backend] if backend in reports else None
        validity["trace_present"] = expected is None or (expected.is_file() and expected.stat().st_size > 0)
        validity["report"] = report_validity(args.output, config, audit)
        validity["passed"] &= (validity["application_succeeded"] and validity["trace_present"]
                               and validity["report"]["passed"])
        (args.output / "validity.json").write_text(json.dumps(validity, indent=2) + "\n")
        outputs = {str(p.relative_to(args.output)): {"sha256": digest(p), "bytes": p.stat().st_size}
                   for p in args.output.rglob("*") if p.is_file()}
        (args.output / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    if not validity["passed"]:
        raise SystemExit("Profiler/application validity failed; inspect application.log and validity.json")
    print(json.dumps(validity, indent=2))


if __name__ == "__main__":
    main()
