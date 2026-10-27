"""Regrade immutable captures with the current validator and retain old verdicts."""
import argparse
import json
from pathlib import Path

from servedistill.profiling import digest, report_validity, runtime_validity


def revalidate(root: Path, cases: list[str], gates_path: Path) -> dict:
    gates = json.loads(gates_path.read_text())
    result = {}
    for label in cases:
        relative = Path(label)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Capture case must stay inside evidence root")
        case = root / relative
        index = json.loads((case / "outputs.json").read_text())
        for name, record in index.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Indexed capture must stay inside case")
            path = case / path
            if (not path.is_file() or path.stat().st_size != record["bytes"]
                    or digest(path) != record["sha256"]):
                raise ValueError("Original indexed capture changed: " + name)
        manifest = json.loads((case / "manifest.json").read_text())
        if manifest["validity_configuration_sha256"] != digest(gates_path):
            raise ValueError("Original runtime gates changed")
        config = manifest["configuration"]
        audit = json.loads((case / "tool-audit.json").read_text())
        runtime = runtime_validity(case, config, gates)
        report = report_validity(case, config, audit)
        original = json.loads((case / "validity.json").read_text())
        result[label] = {
            "original_index_sha256": digest(case / "outputs.json"),
            "original_validity_sha256": digest(case / "validity.json"),
            "original_files_verified": len(index),
            "original_passed": original["passed"],
            "runtime": runtime, "report": report,
            "application_succeeded": audit["application_returncode"] == 0,
            "revalidated_passed": bool(runtime["passed"] and report["passed"]
                                       and audit["application_returncode"] == 0
                                       and original["trace_present"]),
        }
    return {
        "cases": result, "runtime_gates_sha256": digest(gates_path),
        "revalidation_source_sha256": digest(Path(__file__)),
        "validator_source_sha256": digest(
            Path(__file__).resolve().parents[1] / "servedistill/profiling.py"),
        "quality_advantage_supported": False, "speed_advantage_supported": False,
        "boundary": "Original capture, completion and failure bytes remain untouched. "
                    "Revalidation fixes report parsing only and does not turn unavailable tools, "
                    "failed runtime windows, or sampled kernels into a full workload validation.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--gates", type=Path, default=Path(__file__).resolve().parents[1]
                        / "configs/validity/runtime.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = revalidate(args.root, args.cases, args.gates)
    if args.output.exists():
        raise FileExistsError("Do not replace an evidence revalidation")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {"passed": record["revalidated_passed"],
                            "files_verified": record["original_files_verified"]}
                      for name, record in result["cases"].items()}))


if __name__ == "__main__":
    main()
