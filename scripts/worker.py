"""Platform-independent, bounded worker using local copies of verified assets."""
from __future__ import annotations

import argparse
import hashlib
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def asset(spec: dict, local: Path) -> Path:
    source = Path(spec["source"])
    manifest = source / "manifest.json"
    if digest(manifest) != spec["manifest_sha256"]:
        raise RuntimeError("Source manifest hash differs from pinned worker input")
    value = json.loads(manifest.read_text())
    is_model = "repo_id" in value
    local.mkdir(parents=True, exist_ok=True)
    for relative, record in value["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe manifest path")
        suffix = Path("model") / path if is_model else path
        target = local / suffix
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / suffix, target)
        if digest(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
            raise RuntimeError("Worker asset verification failed: " + relative)
    write(local / "manifest.json", value)
    print(json.dumps({"event": "worker_asset_verified", "name": local.name,
                      "files": len(value["files"])}), flush=True)
    return local / "model" if is_model else local


def publish(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".upload")
        shutil.copyfile(path, temporary)
        temporary.replace(target)


def probe(output: Path) -> None:
    import torch
    versions = {}
    for name in ("torch", "transformers", "peft", "vllm", "flash-linear-attention",
                 "causal-conv1d", "triton", "numpy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    value = {
        "python": sys.version, "versions": versions, "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "optional_kernels": {name: importlib.util.find_spec(name) is not None
                             for name in ("fla", "causal_conv1d")},
        "unix_seconds": time.time(),
    }
    if not value["device_count"] or not all("H20" in name for name in value["devices"]):
        raise RuntimeError("Worker does not have the requested H20 devices")
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x.square().mean().backward()
    value["bf16_backward_finite"] = bool(torch.isfinite(x.grad).all())
    write(output / "environment.json", value)
    print(json.dumps({"event": "environment", **value}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["local_root"])
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    destination = Path(config["output_destination"])
    deadline = config["deadline_unix_seconds"]
    started = time.time()
    values = {"output": str(output), "python": sys.executable}
    events = []
    try:
        probe(output)
        if config.get("probe_only"):
            return
        for name, spec in config["assets"].items():
            values[name] = str(asset(spec, root / name))
            if time.time() >= deadline:
                raise TimeoutError("Global worker deadline reached during asset preparation")
        wheelhouse = Path(values["wheelhouse"])
        dependencies = root / "deps"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
             "--target", str(dependencies),
             *map(str, sorted(wheelhouse.glob("*.whl")))],
            check=True, timeout=180)
        code = root / "code"
        code.mkdir(exist_ok=True)
        with tarfile.open(Path(values["bundle"]) / "code.tar.gz") as archive:
            members = archive.getmembers()
            for member in members:
                path = Path(member.name)
                if (path.is_absolute() or ".." in path.parts
                        or not (member.isfile() or member.isdir())):
                    raise ValueError("Unsafe code archive member")
            archive.extractall(code, members=members)
        extra_wheels = sorted((code / "extra_wheels").glob("*.whl"))
        if extra_wheels:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                 "--target", str(dependencies), *map(str, extra_wheels)],
                check=True, timeout=180)
        values["code"] = str(code)
        values["data"] = str(code / "data_search")
        environment = {
            **os.environ, "PYTHONPATH": str(dependencies) + ":" + str(code),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        (output / "environment.json").rename(output / "base-environment.json")
        subprocess.run(
            [sys.executable, "-c", "from scripts.worker import probe; from pathlib import Path; "
             + "probe(Path(" + repr(str(output)) + "))"],
            env=environment, cwd=code, check=True, timeout=120)
        for index, task in enumerate(config["tasks"]):
            seconds = min(task["timeout_seconds"], int(deadline - time.time()))
            if seconds <= 0:
                raise TimeoutError("Global worker deadline reached")
            timestamp = time.time()
            print(json.dumps({"event": "task_start", "index": index, "name": task["name"]}), flush=True)
            if "parallel" in task:
                devices = [child["gpu"] for child in task["parallel"]]
                if len(devices) != len(set(devices)):
                    raise ValueError("Independent experiments must own distinct GPUs")
                def run_child(child):
                    argv = [argument.format_map(values) for argument in child["argv"]]
                    child_environment = {**environment, "CUDA_VISIBLE_DEVICES": str(child["gpu"])}
                    child_start = time.time()
                    with (output / (child["name"] + ".stdout.log")).open("wb") as stdout:
                        with (output / (child["name"] + ".stderr.log")).open("wb") as stderr:
                            try:
                                result = subprocess.run(
                                    argv, env=child_environment, cwd=code, stdout=stdout,
                                    stderr=stderr, timeout=min(seconds, child["timeout_seconds"]))
                                child_status = result.returncode
                            except subprocess.TimeoutExpired:
                                child_status = 124
                    result = {"name": child["name"], "gpu_index": child["gpu"],
                              "returncode": child_status, "start_unix_seconds": child_start,
                              "end_unix_seconds": time.time()}
                    write(output / (child["name"] + ".task.json"), result)
                    print(json.dumps({"event": "child_complete", **result}), flush=True)
                    return result
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                    children = list(pool.map(run_child, task["parallel"]))
                status = max(child["returncode"] for child in children)
                events.extend(children)
            else:
                argv = [argument.format_map(values) for argument in task["argv"]]
                try:
                    result = subprocess.run(argv, env=environment, cwd=code, timeout=seconds)
                    status = result.returncode
                except subprocess.TimeoutExpired:
                    status = 124
            events.append({"name": task["name"], "returncode": status,
                           "start_unix_seconds": timestamp, "end_unix_seconds": time.time()})
            write(output / "task-events.json", events)
            publish(output, destination)
            if status:
                raise RuntimeError("Worker task failed: " + task["name"])
        write(output / "worker-completed.json", {
            "completed": True, "elapsed_seconds": time.time() - started,
            "tasks": events, "assets_verified": True,
        })
    except Exception as error:
        write(output / "worker-failed.json", {
            "error": type(error).__name__, "message": str(error),
            "unix_seconds": time.time(), "tasks": events,
        })
        raise
    finally:
        publish(output, destination)


if __name__ == "__main__":
    main()
