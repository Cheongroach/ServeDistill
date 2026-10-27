"""Finite standalone host/GPU/DCGM collection with explicit failure records."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from servedistill.profiling import HostSampler, digest, locate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring/system.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-pid", type=int, default=os.getpid())
    parser.add_argument("--enable-dcgm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    duration = config["duration_seconds"]
    if not math.isfinite(duration) or not 0 < duration <= 3600:
        raise ValueError("System collection must be finite, between zero and 3600 seconds")
    if args.enable_dcgm:
        config["collectors"]["dcgm"]["enabled"] = True
    argv = {}
    for name, options in config["collectors"].items():
        if not options["enabled"]:
            continue
        binary = locate("nvidia-smi" if name == "nvidia_smi" else "dcgmi")
        if not binary:
            raise FileNotFoundError("Requested telemetry executable is unavailable: " + name)
        if name == "nvidia_smi":
            argv[name] = [binary, "--query-gpu=" + ",".join(options["fields"]),
                          "--format=csv", "--loop-ms=" + str(options["interval_ms"])]
        else:
            count = math.ceil(duration * 1000 / options["interval_ms"])
            argv[name] = [binary, "dmon", "-i", options["gpu"], "-e",
                          ",".join(map(str, options["fields"])), "-d", str(options["interval_ms"]),
                          "-c", str(count)]
    if args.dry_run:
        print(json.dumps({"commands": argv, "configuration": config}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    records, processes, streams = {}, {}, []
    with HostSampler(args.output / "host.jsonl", config["host_poll_seconds"], args.root_pid) as host:
        try:
            for name, command in argv.items():
                stream = (args.output / (name + ".txt")).open("w")
                streams.append(stream)
                processes[name] = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            host.stop_event.wait(duration)
        finally:
            for name, process in processes.items():
                terminated = process.poll() is None
                if terminated:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                records[name] = {"argv": argv[name], "returncode": process.returncode,
                                 "terminated_at_declared_duration": terminated}
            for stream in streams:
                stream.close()
    (args.output / "audit.json").write_text(json.dumps({
        "configuration": config, "config_sha256": digest(args.config), "collectors": records,
        "boundary": "Hardware-wide telemetry. Host data includes co-tenants; DCGM N/A is not zero.",
    }, indent=2) + "\n")
    if any(not r["terminated_at_declared_duration"] and r["returncode"] != 0 for r in records.values()):
        raise SystemExit("A telemetry collector failed; inspect its retained output")


if __name__ == "__main__":
    main()
