"""Isolate cuDNN dispatch in bounded cold-restored diagnostics and retain failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.install_profile_tools import install
from scripts.worker import publish
from servedistill.profiling import digest, locate


def main():
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "data", "checkpoint", "config", "output", "tools_root"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    candidates_path = repo / "configs/monitoring/ncu-compatibility-candidates.json"
    candidates = json.loads(candidates_path.read_text())
    selected = candidates["candidates"][0]
    binary = install({"repository": candidates["repository"], "packages": {"ncu": selected}},
                     "ncu", args.tools_root)
    install(json.loads((repo / "configs/monitoring/tools.lock.json").read_text()),
            "nsys", args.tools_root)
    system = {}
    for name, argv in {
        "perf": ["perf", "--version"], "dcgmi": ["dcgmi", "discovery", "-l"],
        "compute-sanitizer": ["compute-sanitizer", "--version"],
    }.items():
        executable = locate(name)
        if executable is None:
            system[name] = {"available": False}
            continue
        with (args.output / (name + "-capability.log")).open("w") as stream:
            try:
                result = subprocess.run([executable, *argv[1:]], stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=30)
                system[name] = {"available": True, "binary": executable,
                                "returncode": result.returncode}
            except subprocess.TimeoutExpired:
                system[name] = {"available": True, "binary": executable, "returncode": 124}
    (args.output / "system-tools.json").write_text(json.dumps(system, indent=2) + "\n")
    watch_argv = [sys.executable, str(repo / "scripts/system_watch.py"),
                  "--config", str(repo / "configs/monitoring/system.json"),
                  "--output", str(args.output / "system-watch")]
    if system["dcgmi"].get("returncode") == 0:
        watch_argv.append("--enable-dcgm")
    with (args.output / "system-watch.log").open("w") as stream:
        watch = subprocess.run(watch_argv, stdout=stream, stderr=subprocess.STDOUT, timeout=90)
    (args.output / "system-watch-status.json").write_text(json.dumps({
        "returncode": watch.returncode, "argv": watch_argv,
        "boundary": "Standalone idle host telemetry; serial and outside kernel profiling.",
    }, indent=2) + "\n")
    cases = []
    for label, preset, attention in (
        ("none_default", "baseline", "default"),
        ("none_no_cudnn", "baseline", "no_cudnn"),
        ("ncu_no_cudnn", "ncu_no_cudnn", "no_cudnn"),
        ("ncu_gemm_no_cudnn", "ncu_gemm_no_cudnn", "no_cudnn"),
        ("nsys_no_cudnn", "nsys", "no_cudnn"),
        ("perf_no_cudnn", "perf", "no_cudnn"),
        ("sanitizer_no_cudnn", "sanitizer", "no_cudnn"),
    ):
        profile = json.loads((repo / f"configs/monitoring/{preset}.json").read_text())
        profile["attention_backend"] = attention
        config_path = args.output / (label + "-config.json")
        config_path.write_text(json.dumps(profile, indent=2) + "\n")
        tool = {"nsys": "nsys", "perf": "perf", "sanitizer": "compute-sanitizer"}.get(profile["backend"])
        if tool and not locate(tool, args.tools_root):
            cases.append({"case": label, "returncode": 127, "error": "Executable unavailable: " + tool})
        else:
            argv = [sys.executable, str(repo / "scripts/profile_run.py"),
                    "--profile", str(config_path), "--tools-root", str(args.tools_root),
                    "--student", str(args.student), "--teacher", str(args.teacher),
                    "--data", str(args.data), "--config", str(args.config),
                    "--resume", str(args.checkpoint), "--output", str(args.output / label)]
            with (args.output / (label + ".log")).open("w") as stream:
                try:
                    result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=profile["timeout_seconds"] + 600,
                                            env={**os.environ, "SERVEDISTILL_NCU": binary})
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            validity_path = args.output / label / "validity.json"
            cases.append({"case": label, "returncode": code,
                          "validity": json.loads(validity_path.read_text()) if validity_path.is_file() else None})
        (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        if args.destination:
            publish(args.output, args.destination)
        print(json.dumps(cases[-1]), flush=True)
    (args.output / "completed.json").write_text(json.dumps({
        "completed": True, "ncu_package": selected,
        "candidates_sha256": digest(candidates_path), "cases": cases,
        "all_cases_passed": all(c["returncode"] == 0 and c.get("validity", {}).get("passed")
                                for c in cases),
        "boundary": "Every unavailable/failed tool retained; no speed or quality conclusion. "
                    "no_cudnn is a diagnostic-only attention dispatch change.",
    }, indent=2) + "\n")
    if args.destination:
        publish(args.output, args.destination)


if __name__ == "__main__":
    main()
