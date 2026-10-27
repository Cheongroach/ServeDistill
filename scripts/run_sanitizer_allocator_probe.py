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
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--profiles", type=Path, nargs=2, default=[
        repo / "configs/monitoring/baseline_no_expandable.json",
        repo / "configs/monitoring/sanitizer_no_expandable.json"])
    args = parser.parse_args()
    profiles = [(path, json.loads(path.read_text())) for path in args.profiles]
    baseline, sanitizer = [config for _, config in profiles]
    if baseline["backend"] != "none" or sanitizer["backend"] != "sanitizer":
        raise ValueError("Require an unprofiled baseline followed by a sanitizer")
    workload_keys = ("first_update", "last_update", "microbatch",
                     "attention_backend", "cuda_allocator", "rollout_batch")
    if (args.profiles[0].stem == args.profiles[1].stem
            or any(baseline.get(key) != sanitizer.get(key) for key in workload_keys)):
        raise ValueError("Pair must have distinct names and identical diagnostic workload settings")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for profile, configuration in profiles:
        name = profile.stem
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
        "paired_configuration": {key: baseline.get(key) for key in workload_keys},
        "hypothesis": "Frozen matched diagnostic of allocator/batch settings under Compute "
                      "Sanitizer. Completion alone does not establish compatibility.",
        "boundary": "Same cold checkpoint, train inputs, two updates, model precision, "
                    "prompt/group/token budgets and matched diagnostic configuration. "
                    "Full sanitizer application is instrumented. Rollout batch overrides "
                    "may change trajectories and are not speed or quality results. "
                    "Earlier failures remain separate.",
    }, indent=2) + "\n")
    if args.destination:
        publish(args.output, args.destination)


if __name__ == "__main__":
    main()
