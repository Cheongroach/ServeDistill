"""Finish the DEV audit after the existing job monitor closes all four runs."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from servedistill.advantage_contracts import file_digest
from servedistill.runtime import write_json


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ops = repo / ".ops/router-stability-dev-v1"
    deadline = time.monotonic() + 4 * 3600
    while not (ops / "terminal-completed.json").exists():
        events = ops / "monitor-events.jsonl"
        if events.exists():
            last = json.loads(events.read_text().splitlines()[-1])
            if any(r["status"] in ("failed", "stopped", "failed_to_launch") for r in last["states"]):
                raise RuntimeError("Existing DEV monitor observed a failed job; diagnose before retry")
        if time.monotonic() >= deadline:
            raise TimeoutError("DEV finalization deadline reached without terminal evidence")
        time.sleep(30)
    commands = [
        [sys.executable, "scripts/collect_router_stability_dev.py"],
        [sys.executable, "scripts/analyze_router_stability_dev.py",
         "--root", "reports/advantage/raw/router-stability-dev-v1",
         "--contract", ".ops/router-stability-dev-v1/contract.json",
         "--dev", ".ops/router-stability-dev-v1/dev.jsonl",
         "--output", "reports/advantage/ROUTER_STABILITY_DEV.json"],
        [sys.executable, "scripts/write_router_stability_dev_report.py"],
    ]
    for index, command in enumerate(commands):
        with (ops / f"finalize-step{index}.stdout").open("wb") as stdout:
            with (ops / f"finalize-step{index}.stderr").open("wb") as stderr:
                result = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr,
                                        timeout=3600 if index == 0 else 180)
        if result.returncode:
            raise RuntimeError(f"DEV finalization step {index} failed; inspect saved stderr")
    result_path = repo / "reports/advantage/ROUTER_STABILITY_DEV.json"
    result = json.loads(result_path.read_text())
    status_path = repo / "reports/advantage/STATUS.json"
    status = json.loads(status_path.read_text())
    status["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status["development_stability"].update({
        "status": "terminal_collected_and_independently_regraded",
        "gpu_result_verified": True,
        "analysis": "reports/advantage/ROUTER_STABILITY_DEV.json",
        "report": "reports/advantage/ROUTER_STABILITY_DEV.md",
        "cost": "reports/advantage/ROUTER_STABILITY_DEV_COST.json",
        "selected_dev_rule": result["selected_dev_rule"],
        "visual_and_editorial_review_completed": False,
        "quality_advantage_supported": False,
        "cost_advantage_supported": False,
    })
    status["next"] = (
        "Inspect the rendered DEV figure and finish the three scientific editing passes. "
        "Assess the selected DEV rule against both SFT controls before considering any new "
        "independent frozen confirmation. Preserve the failed original confirmation."
    )
    write_json(status_path, status)
    write_json(ops / "finalization-completed.json", {
        "completed": True, "scope": "development_only",
        "independent_regrading_completed": True, "source_sha256": file_digest(result_path),
        "selected_dev_rule": result["selected_dev_rule"],
        "visual_inspection_completed": False, "holdout_confirmation_launched": False,
    })
    print(json.dumps({
        "DEV_audit_completed": True, "selected_dev_rule": result["selected_dev_rule"],
        "report": str(repo / "reports/advantage/ROUTER_STABILITY_DEV.md"),
        "quality_advantage_supported": False,
        "remaining": "Manual figure inspection and scientific editing",
    }), flush=True)


if __name__ == "__main__":
    main()
