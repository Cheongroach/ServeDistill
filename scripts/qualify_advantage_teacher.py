"""Qualify a development teacher adapter under the actual distillation prompt."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.analyze_expanded import classification_metrics
from scripts.run_advantage_arm import Publisher
from servedistill.advantage_contracts import file_digest, verify_data, verify_files
from servedistill.data import canonical, load_rows, system_prompt
from servedistill.runtime import Runtime, seed_all, write_json


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "model-manifest", "adapter", "data", "output", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    verification = {
        "model": verify_files(args.model_manifest, args.model, model_asset=True),
        "data": verify_data(args.data, confirmation=False),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.data / "dev.jsonl")
    rows = [row for row in rows if row["domain"] in ("search", "relevance", "substitute")] + [
        row for domain in ("math", "instruction")
        for row in [record for record in rows if record["domain"] == domain][:100]]
    rows = [{**row, "system": system_prompt(row, teacher=True)} for row in rows]
    source = Path(__file__).resolve().parents[1]
    write_json(args.output / "provenance.json", {
        "adapter_sha256": file_digest(args.adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_digest(args.adapter / "adapter_config.json"),
        "adapter_checkpoint": args.adapter.name, "artifact_verification": verification,
        "prompt_systems_sha256": hashlib.sha256(
            canonical([row["system"] for row in rows]).encode()).hexdigest(),
        "execution_source_files": {
            name: file_digest(source / name) for name in (
                "servedistill/runtime.py", "servedistill/data.py",
                "servedistill/advantage_contracts.py", "scripts/qualify_advantage_teacher.py")},
        "boundary": "Full development qualification under the system used by OPD teacher scoring. "
                    "No independent holdout predictions or method advantage claim.",
    })
    publisher = Publisher(args.output, args.destination)
    try:
        seed_all(202610255)
        runtime = Runtime(str(args.model), adapter_path=str(args.adapter))
        write_json(args.output / "contract.json", runtime.contract)
        report = runtime.evaluate(rows, args.output, "teacher-actual")
        evidence = load_rows(args.output / "teacher-actual.jsonl")
        write_json(args.output / "teacher-actual.classes.json", {
            domain: classification_metrics([row for row in evidence if row["domain"] == domain])
            for domain in ("relevance", "substitute")})
        write_json(args.output / "completed.json", {
            "completed": True, "report": report, "started_unix_seconds": started,
            "end_unix_seconds": time.time(), "elapsed_seconds": time.time() - started,
            "boundary": "Development only; actual teacher scoring prompt"})
    except Exception as error:
        write_json(args.output / "failed.json", {"error": type(error).__name__, "message": str(error)})
        raise
    finally:
        publisher.publish()


if __name__ == "__main__":
    main()
