"""Collect a complete DEV stability study after every job has closed."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ops = repo / ".ops/router-stability-dev-v1"
    submissions = json.loads((ops / "submissions.json").read_text())
    cli = [os.environ.get("MERLIN_CLI", "merlin-cli"), "--control-plane",
           os.environ.get("MERLIN_CONTROL_PLANE", "cn"), "job-v2"]
    hdfs = os.environ.get("SERVEDISTILL_HDFS_ADVANTAGE")
    if not hdfs:
        raise ValueError("Set SERVEDISTILL_HDFS_ADVANTAGE to your evidence root")
    env = {**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true"}
    deadline = time.monotonic() + 4 * 3600
    successful = {"done", "success"}
    terminal = successful | {"failed", "stopped", "failed_to_launch"}
    while time.monotonic() < deadline:
        states = []
        for record in submissions:
            response = json.loads(subprocess.check_output([
                *cli, "runs", "get", "--json", json.dumps({"sid": record["sid"]}),
            ], timeout=90))
            (ops / f"current-seed{record['seed']}.json").write_text(
                json.dumps(response, indent=2) + "\n"
            )
            states.append({
                "seed": record["seed"], "sid": record["sid"],
                "status": response["data"]["status"],
                "trial_id": response["data"]["runtime_context"]["trial_id"],
            })
        event = {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "states": states,
        }
        with (ops / "monitor-events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        if any(r["status"] in terminal - successful for r in states):
            raise RuntimeError("DEV stability job did not succeed")
        if all(r["status"] in successful for r in states):
            root = repo / "reports/advantage/raw/router-stability-dev-v1"
            root.mkdir(exist_ok=True)
            for record in submissions:
                seed = record["seed"]
                target = root / f"seed{seed}"
                target.mkdir(exist_ok=True)
                remote = f"{hdfs.rstrip('/')}/router-stability-dev-v1/seed{seed}/"
                names = ["completed.json", "worker-completed.json"]
                names += [
                    label + suffix
                    for label in ("fixed320", "final640", "last2", "all4",
                                  "offline_sft", "oracle_sft")
                    for suffix in (".jsonl", ".summary.json")
                ]
                for name in names:
                    value = subprocess.check_output(
                        ["hdfs", "dfs", "-cat", remote + name],
                        env=env, timeout=60,
                    )
                    (target / name).write_bytes(value)
            (ops / "terminal-completed.json").write_text(json.dumps({
                "completed": True,
                "all_jobs_terminal_successful": True,
                "scope": "development_only",
                "holdout_observed": False,
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, indent=2) + "\n")
            return
        time.sleep(30)
    raise TimeoutError("DEV monitoring deadline reached")


if __name__ == "__main__":
    main()
