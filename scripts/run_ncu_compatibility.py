"""Test pinned NCU releases serially against the actual cold-restored workload."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.install_profile_tools import install
from scripts.worker import publish
from servedistill.profiling import digest


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "data", "checkpoint", "config", "output", "tools_root"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--candidates", type=Path,
                        default=repo / "configs/monitoring/ncu-compatibility-candidates.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = json.loads(args.candidates.read_text())
    cases, selected = [], None
    for candidate in candidates["candidates"]:
        version = candidate["version"]
        case = args.output / version
        case.mkdir()
        started = time.time()
        record = {"version": version, "package": candidate, "started": started}
        try:
            lock = {"repository": candidates["repository"], "packages": {"ncu": candidate}}
            binary = install(lock, "ncu", args.tools_root / version)
            record["binary"] = binary
            argv = [sys.executable, str(repo / "scripts/profile_run.py"),
                    "--profile", str(repo / "configs/monitoring/ncu.json"),
                    "--student", str(args.student), "--teacher", str(args.teacher),
                    "--data", str(args.data), "--config", str(args.config),
                    "--resume", str(args.checkpoint), "--output", str(case / "capture")]
            with (case / "runner.log").open("w") as stream:
                process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=1200, env={**os.environ, "SERVEDISTILL_NCU": binary})
            record["returncode"] = process.returncode
            path = case / "capture/validity.json"
            record["validity"] = json.loads(path.read_text()) if path.is_file() else {"passed": False}
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            record["returncode"] = 125
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        record["ended"] = time.time()
        cases.append(record)
        (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        if args.destination:
            publish(args.output, args.destination)
        print(json.dumps({"version": version, "returncode": record["returncode"],
                          "validity": record.get("validity")}), flush=True)
        if record["returncode"] == 0 and record["validity"]["passed"]:
            selected = {**candidate, "binary": record["binary"]}
            break
    result = {"completed": selected is not None, "selected": selected, "cases": cases,
              "candidates_sha256": digest(args.candidates),
              "boundary": "Actual-workload compatibility and counters; not a performance comparison."}
    (args.output / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.destination:
        publish(args.output, args.destination)
    if selected is None:
        raise SystemExit("No NCU version captured valid counters; all failures retained")


if __name__ == "__main__":
    main()
