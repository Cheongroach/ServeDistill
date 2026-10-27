"""Run separate cold-restored diagnostic processes; keep failed tools visible."""
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
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "data", "checkpoint", "config", "output", "tools_root"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--backends", nargs="+", default=["none", "ncu", "nsys", "torch"])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((repo / "configs/monitoring/tools.lock.json").read_text())
    binaries = {}
    for name in ("ncu", "nsys"):
        if name in args.backends:
            binaries[name] = install(lock, name, args.tools_root)
    installation = {"binaries": binaries, "lock_sha256": digest(repo / "configs/monitoring/tools.lock.json")}
    (args.output / "installation.json").write_text(json.dumps(installation, indent=2) + "\n")
    cases = []
    for backend in args.backends:
        profile = repo / "configs/monitoring" / (("baseline" if backend == "none" else backend) + ".json")
        argv = [sys.executable, str(repo / "scripts/profile_run.py"),
                "--profile", str(profile), "--student", str(args.student),
                "--teacher", str(args.teacher), "--data", str(args.data),
                "--config", str(args.config), "--resume", str(args.checkpoint),
                "--output", str(args.output / backend), "--tools-root", str(args.tools_root)]
        started = time.time()
        with (args.output / (backend + ".log")).open("w") as stream:
            try:
                environment = {**os.environ, **{
                    "SERVEDISTILL_" + name.upper(): binary for name, binary in binaries.items()}}
                process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=1200, env=environment)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
        cases.append({"backend": backend, "returncode": returncode,
                      "started": started, "ended": time.time()})
        (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        if args.destination:
            publish(args.output, args.destination)
        print(json.dumps(cases[-1]), flush=True)
    validity = {}
    for backend in args.backends:
        path = args.output / backend / "validity.json"
        validity[backend] = json.loads(path.read_text()) if path.is_file() else {"passed": False}
    states = {}
    for backend in args.backends:
        paths = list((args.output / backend / "application/train").glob("checkpoint-*/state.json"))
        if len(paths) == 1:
            state = json.loads(paths[0].read_text())
            states[backend] = {key: state[key] for key in ("adapter_sha256", "optimizer_sha256", "rng_sha256")}
    numerical_equivalence = (len(states) == len(args.backends)
                             and len({json.dumps(value, sort_keys=True) for value in states.values()}) == 1)
    completed = {"completed": all(r["returncode"] == 0 for r in cases),
                 "all_runtime_gates_passed": all(r["passed"] for r in validity.values()),
                 "exact_final_state_equivalence": numerical_equivalence,
                 "states": states, "cases": cases,
                 "boundary": "Profiling diagnosis only; speed claims require unprofiled repeated fixed-workload timing."}
    (args.output / "suite-completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    if args.destination:
        publish(args.output, args.destination)
    if not completed["completed"] or not completed["all_runtime_gates_passed"]:
        raise SystemExit("At least one profiler failed; retained evidence records the actual boundary")


if __name__ == "__main__":
    main()
