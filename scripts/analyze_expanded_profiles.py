"""Validate and analyze actual fixed-bank Nsight captures independently of timing."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.analyze_profile import analyze_sqlite, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {}
    for worker in range(2):
        root = args.performance / f"performance{worker}"
        audit = json.loads((root / "trace-audit.json").read_text())
        for microbatch in (1, 4):
            label = f"worker{worker}-m{microbatch}"
            report = root / f"trace-m{microbatch}.nsys-rep"
            sqlite = root / f"trace-m{microbatch}.sqlite"
            result = {
                "worker_index": worker, "microbatch": microbatch,
                "profile_returncode": audit.get(f"m{microbatch}_returncode"),
                "report_exists": report.exists(), "sqlite_exists": sqlite.exists(),
                "completed": False,
            }
            if result["profile_returncode"] == 0 and report.exists() and sqlite.exists():
                summary = analyze_sqlite(sqlite, args.output,
                                         update_name="fixed_bank_update", label=label)
                if summary["cuda_kernel_count"] <= 0 or summary["host_update_seconds"] <= 0:
                    raise ValueError("Empty Nsight capture: " + label)
                result.update({
                    "completed": True, "cuda": summary,
                    "nsys_report_sha256": hashlib.file_digest(report.open("rb"), "sha256").hexdigest(),
                    "nsys_report_bytes": report.stat().st_size,
                    "sqlite_bytes": sqlite.stat().st_size,
                })
            else:
                result["reason"] = "Nsight exit or nonempty exported artifacts not established"
            cases[label] = result
    write_json(args.output / "profiles.json", {
        "cases": cases,
        "all_four_captures_verified": all(row["completed"] for row in cases.values()),
        "boundary": "Profiled fixed bank0 only; excluded from formal throughput statistics",
    })
    print(json.dumps({key: row["completed"] for key, row in cases.items()}))


if __name__ == "__main__":
    main()
