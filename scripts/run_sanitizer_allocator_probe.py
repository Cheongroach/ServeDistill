"""Test the observed cuMemCreate failure with a matched full-workload allocator pair."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.worker import publish
from servedistill.profiling import digest


def main():
    parser = argparse.ArgumentParser()
    for name in ("student", "teacher", "data", "checkpoint", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name in ("baseline_no_expandable", "sanitizer_no_expandable"):
        profile = repo / "configs/monitoring" / (name + ".json")
        configuration = json.loads(profile.read_text())
        argv = [sys.executable, str(repo / "scripts/profile_run.py"),
                "--profile", str(profile), "--student", str(args.student),
                "--teacher", str(args.teacher), "--data", str(args.data),
                "--config", str(args.config), "--resume", str(args.checkpoint),
                "--output", str(args.output / name)]
        with (args.output / (name + ".log")).open("w") as stream:
            try:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=configuration["timeout_seconds"] + 300)
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = 124
        validity = args.output / name / "validity.json"
        cases.append({
            "case": name, "configuration_sha256": digest(profile), "returncode": status,
            "validity": json.loads(validity.read_text()) if validity.is_file() else None,
        })
        (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        if args.destination:
            publish(args.output, args.destination)
        print(json.dumps(cases[-1]), flush=True)
    (args.output / "completed.json").write_text(json.dumps({
        "completed": True, "cases": cases,
        "all_cases_passed": all(c["returncode"] == 0 and (c["validity"] or {}).get("passed")
                                for c in cases),
        "hypothesis": "Disabling expandable segments may avoid observed cuMemCreate "
                      "CUDA_ERROR_INVALID_VALUE under Compute Sanitizer; not established.",
        "boundary": "Same cold checkpoint, train inputs, two updates, model precision, "
                    "rollout configuration, no_cudnn attention and learner microbatch1. "
                    "Full application is instrumented. Allocator policy is a diagnostic change, "
                    "not a speed or quality result. Earlier failures remain separate.",
    }, indent=2) + "\n")
    if args.destination:
        publish(args.output, args.destination)


if __name__ == "__main__":
    main()
