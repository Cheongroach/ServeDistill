"""Collect bounded CUDA traces on a worker with previously verified local assets."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


def command(argv, output, timeout=60):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    output.write_text(result.stdout + result.stderr)
    return {"argv": argv, "returncode": result.returncode}


def locate(name):
    path = shutil.which(name)
    if path:
        return path
    patterns = (f"/opt/nvidia/nsight*/*/bin/{name}",
                f"/opt/nvidia/nsight*/*/target-linux-x64/{name}",
                f"/usr/local/cuda/bin/{name}")
    import glob
    matches = [p for pattern in patterns for p in glob.glob(pattern)]
    return sorted(matches)[-1] if matches else None


def install_nsys(root):
    base = "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
    filename = "NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb"
    expected = "37a1cb9b12cdbc9c40a8cb42e58b05a141e44c63c800cac435cb8e43256e0780"
    package = root / filename
    with urllib.request.urlopen(base + filename, timeout=30) as response:
        with package.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    actual = hashlib.file_digest(package.open("rb"), "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError("Nsight package differs from the official repository SHA256")
    target = root / "nsight-cli"
    subprocess.run(["dpkg-deb", "-x", str(package), str(target)], check=True, timeout=60)
    binary = next(target.rglob("target-linux-x64/nsys"))
    return str(binary), {"url": base + filename, "sha256": actual,
                         "bytes": package.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root, args.root / "output"
    code = root / "code"
    overlay = json.loads(gzip.decompress(args.overlay.read_bytes()))
    for relative, source in overlay.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe source overlay")
        target = code / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    sources = {name: hashlib.sha256(text.encode()).hexdigest()
               for name, text in overlay.items()}
    (output / "source-hashes.json").write_text(json.dumps(sources, indent=2))
    checkpoint = root / "checkpoint-0040"
    shutil.copytree(args.checkpoint, checkpoint)
    (output / "checkpoint-state.json").write_text((checkpoint / "state.json").read_text())
    command(["nvidia-smi", "-q"], output / "nvidia-smi-details.txt")
    nsys, ncu = locate("nsys"), locate("ncu")
    audit = {"nsys_before_install": nsys, "ncu": ncu, "collected_utc_unix": time.time()}
    if not nsys:
        try:
            nsys, audit["nsys_install"] = install_nsys(root)
        except (OSError, RuntimeError, StopIteration, subprocess.SubprocessError) as error:
            audit["nsys_install_error"] = str(error)
    if nsys:
        audit["nsys_version"] = command([nsys, "--version"], output / "nsys-version.txt")
        audit["nsys_environment"] = command(
            [nsys, "status", "--environment"], output / "nsys-environment.txt")
        command([nsys, "profile", "--help"], output / "nsys-profile-help.txt")
    if ncu:
        audit["ncu_version"] = command([ncu, "--version"], output / "ncu-version.txt")
    (output / "tool-audit.json").write_text(json.dumps(audit, indent=2))
    import torch
    (output / "cuda-profiler-capabilities.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "activities": [str(a) for a in torch.profiler.supported_activities()],
        "device": torch.cuda.get_device_name(0),
    }, indent=2))
    telemetry_stream = (output / "gpu-telemetry.csv").open("w")
    telemetry = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
        "memory.used,power.draw,clocks.sm", "--format=csv", "--loop-ms=200"],
        stdout=telemetry_stream, stderr=subprocess.STDOUT)
    cases = [("baseline_m1", "none", 1), ("nsys_m1", "nsys" if nsys else "torch", 1),
             ("baseline_m1_warm", "none", 1), ("baseline_m4", "none", 4),
             ("nsys_m4", "nsys" if nsys else "torch", 4), ("torch_m4", "torch", 4)]
    records = []
    try:
        for label, backend, microbatch in cases:
            case = root / "runs" / label
            destination = output / label
            destination.mkdir()
            argv = [
                sys.executable, str(code / "scripts/profile_training.py"),
                "--student", str(root / "student/model"),
                "--teacher", str(root / "teacher/model"), "--data", str(code / "data_search"),
                "--config", str(code / "configs/matrix/full_kl_seed17.json"),
                "--resume", str(checkpoint), "--output", str(case),
                "--backend", backend, "--first-update", "42", "--last-update", "42",
                "--microbatch", str(microbatch)]
            if label == "torch_m4":
                argv.extend(["--record-shapes", "--torch-first-microbatch"])
            if backend == "nsys":
                argv = [nsys, "profile", "--trace=cuda,nvtx,osrt",
                        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                        "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
                        "--output=" + str(destination / "capture"), *argv]
            started = time.time()
            try:
                with (destination / "application.log").open("w") as stream:
                    result = subprocess.run(argv, cwd=code, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=300)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
            record = {"case": label, "backend": backend, "microbatch": microbatch,
                      "returncode": returncode, "elapsed_seconds": time.time() - started}
            records.append(record)
            (output / "case-status.json").write_text(json.dumps(records, indent=2))
            for name in ("observation",):
                if (case / name).exists():
                    shutil.copytree(case / name, destination / name)
            for name in ("metrics.jsonl", "restoration-0040.json", "last-rollout-contract.json"):
                source = case / "train" / name
                if source.exists():
                    shutil.copyfile(source, destination / name)
            print(json.dumps(record), flush=True)
            if returncode:
                from scripts.worker import publish
                publish(output, Path(os.environ["PROFILE_DESTINATION"]))
                raise RuntimeError("Profile continuation failed: " + label)
            if backend == "nsys":
                report = destination / "capture.nsys-rep"
                for report_name in ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum",
                                    "cuda_gpu_mem_time_sum"):
                    record[report_name] = command(
                        [nsys, "stats", "--report=" + report_name, "--format=csv", str(report)],
                        destination / (report_name + ".csv"), timeout=90)
            from scripts.worker import publish
            publish(output, Path(os.environ["PROFILE_DESTINATION"]))
        (output / "profiling-completed.json").write_text(json.dumps({
            "completed": True, "cases": records, "warmup_update": 41, "capture_update": 42,
            "scope": "one H20, restored update40, diagnostic continuations; not new quality evaluation",
        }, indent=2))
    finally:
        telemetry.send_signal(signal.SIGTERM)
        telemetry.wait(timeout=10)
        telemetry_stream.close()


if __name__ == "__main__":
    main()
