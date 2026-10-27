"""Run common-start development arms or a hash-frozen confirmation arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from servedistill.advantage_contracts import verify_cost_credit, verify_data, verify_files, verify_labels
from servedistill.advantage_failures import control_failure_checkpoint
from servedistill.adapter_average import materialize_average, validate_postprocessing
from servedistill.data import load_rows
from servedistill.runtime import Runtime, write_json
from servedistill.time_budget import completed_checkpoint
from scripts.analyze_expanded import classification_metrics


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Publisher:
    def __init__(self, source, destination):
        self.source, self.destination = source, destination
        self.saved = {}

    def publish(self):
        for path in self.source.rglob("*"):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            signature = (path.stat().st_size, path.stat().st_mtime_ns)
            if self.saved.get(path) == signature:
                continue
            target = self.destination / path.relative_to(self.source)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".upload")
            shutil.copyfile(path, temporary)
            temporary.replace(target)
            self.saved[path] = signature


def main():
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "initial-adapter", "data", "config",
                "output", "destination"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--teacher-labels", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--student-manifest", type=Path)
    parser.add_argument("--cost-manifest", type=Path)
    parser.add_argument("--defer-confirmation-evaluation", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    confirmation = args.phase == "confirmation"
    if args.defer_confirmation_evaluation and not confirmation:
        raise ValueError("Only confirmation evaluation can be deferred")
    verification = {
        "data": verify_data(
            args.data,
            confirmation and not args.defer_confirmation_evaluation,
        )
    }
    if confirmation and not args.student_manifest:
        raise ValueError("Confirmation requires verified base student weights")
    if args.student_manifest:
        verification["student"] = verify_files(args.student_manifest, args.student, model_asset=True)
    if args.teacher_manifest:
        verification["teacher"] = verify_files(args.teacher_manifest, args.teacher, model_asset=True)
    if args.teacher_labels:
        verification["labels"] = verify_labels(args.teacher_labels, args.data / "train.jsonl")
    if config.get("method_time_budget_seconds") is not None:
        if confirmation and not args.cost_manifest:
            raise ValueError("Equal-time confirmation requires a measured cost credit ledger")
        if args.cost_manifest:
            verification["prior_method_costs"] = verify_cost_credit(args.cost_manifest, config)
    source_root = Path(__file__).resolve().parents[1]
    source_files = (
        "servedistill/train.py", "servedistill/runtime.py", "servedistill/data.py",
        "servedistill/losses.py", "servedistill/scheduling.py",
        "servedistill/advantage_contracts.py", "servedistill/evaluate.py",
        "scripts/run_advantage_arm.py", "servedistill/time_budget.py",
        "servedistill/statistics.py", "scripts/analyze_expanded.py",
        "servedistill/advantage_failures.py", "servedistill/adapter_average.py",
    )
    source_hashes = {name: digest(source_root / name) for name in source_files}
    fingerprints = {
        "config_sha256": digest(args.config),
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "initial_adapter_file_sha256": digest(args.initial_adapter / "adapter_model.safetensors"),
        "initial_adapter_config_sha256": digest(args.initial_adapter / "adapter_config.json"),
        "student_manifest_sha256": digest(args.student_manifest) if args.student_manifest else None,
        "teacher_manifest_sha256": digest(args.teacher_manifest) if args.teacher_manifest else None,
        "teacher_labels_sha256": digest(args.teacher_labels) if args.teacher_labels else None,
        "cost_manifest_sha256": digest(args.cost_manifest) if args.cost_manifest else None,
        "execution_source_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
    }
    if confirmation:
        if config["objective"] not in ("oracle_sft", "offline_sft") and not args.teacher_manifest:
            raise ValueError("Confirmation OPD requires the verified teacher manifest")
        if config["objective"] == "offline_sft" and not args.teacher_labels:
            raise ValueError("Confirmation offline SFT requires verified hard trajectories")
        if not args.protocol:
            raise ValueError("Confirmation requires an immutable pre-test protocol")
        protocol = json.loads(args.protocol.read_text())
        expected = protocol["allowed_arms"].get(args.label)
        if expected != fingerprints or protocol.get("test_observed_before_freeze"):
            raise ValueError("Execution differs from the pre-test arm contract")
        postprocessing = protocol["arm_postprocessing"][args.label]
        validate_postprocessing(postprocessing, config["steps"])
    args.output.mkdir(parents=True, exist_ok=False)
    publisher = Publisher(args.output, args.destination)
    write_json(args.output / "start-provenance.json", {
        **fingerprints, "label": args.label, "phase": args.phase,
        "optimizer": "fresh AdamW from the shared adapter",
        "seed": config["seed"], "test_used_for_selection": False,
        "artifact_verification": verification, "execution_source_files": source_hashes,
        "protocol_sha256": digest(args.protocol) if args.protocol else None,
    })

    def call(label, argv):
        started = time.time()
        with (args.output / (label + ".stdout.log")).open("w") as stream:
            process = subprocess.Popen([sys.executable, *argv], stdout=stream,
                                       stderr=subprocess.STDOUT)
            while True:
                try:
                    returncode = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    publisher.publish()
        write_json(args.output / (label + "-execution.json"), {
            "returncode": returncode, "start_unix_seconds": started,
            "end_unix_seconds": time.time()})
        publisher.publish()
        if returncode:
            raise RuntimeError("Advantage arm failed at " + label)

    try:
        argv = [
            "-m", "servedistill.train", "--student", str(args.student),
            "--teacher", str(args.teacher), "--adapter", str(args.initial_adapter),
            "--data", str(args.data), "--config", str(args.config),
            "--output", str(args.output / "train"),
        ]
        if args.teacher_labels:
            argv.extend(["--teacher-labels", str(args.teacher_labels)])
        failure_policy = protocol.get("control_failure_policy", {}).get(args.label) if confirmation else None
        control_failure = None
        try:
            call("train", argv)
        except RuntimeError:
            if failure_policy is None:
                raise
            checkpoint, control_failure = control_failure_checkpoint(
                args.output, config, failure_policy)
            if checkpoint is not None:
                call("final", [
                    "-m", "servedistill.evaluate", "--student", str(args.student),
                    "--adapter", str(args.output / "train" / checkpoint),
                    "--data", str(args.data), "--phase", "final",
                    "--output", str(args.output / "final"),
                ])
            write_json(args.output / "control-outcome.json", {
                **control_failure, "phase": args.phase, "label": args.label,
                "status": "training_failed", "test_used_for_selection": False,
                "checkpoint_evaluated": checkpoint is not None,
                "protocol_sha256": digest(args.protocol),
            })
            return
        completed = json.loads((args.output / "train/completed.json").read_text())
        final_checkpoint = completed_checkpoint(config, completed)
        selected = completed["selected"]
        checkpoint = (selected["path"] if args.phase == "development"
                      else final_checkpoint)
        if args.phase == "development":
            runtime = Runtime(str(args.student), adapter_path=str(args.output / "train" / checkpoint))
            rows = load_rows(args.data / "dev.jsonl")
            rows = [row for row in rows if row["domain"] in ("search", "relevance", "substitute")] + [
                row for domain in ("math", "instruction")
                for row in [r for r in rows if r["domain"] == domain][:100]]
            report = runtime.evaluate(rows, args.output, "selected-dev")
            evidence = load_rows(args.output / "selected-dev.jsonl")
            write_json(args.output / "selected-dev.classes.json", {
                domain: classification_metrics([row for row in evidence if row["domain"] == domain])
                for domain in ("relevance", "substitute")})
            write_json(args.output / "development.json", {
                "selected": selected, "report": report,
                "boundary": "Already observed development queries; not confirmatory evidence"})
        else:
            if postprocessing["kind"] == "fixed_checkpoint":
                checkpoint = f"checkpoint-{postprocessing['step']:04d}"
            evaluated_adapter = args.output / "train" / checkpoint
            if postprocessing["kind"] == "lora_factor_mean_fp32":
                evaluated_adapter = args.output / "averaged-adapter"
                average_evidence = materialize_average(
                    args.output / "train", evaluated_adapter, postprocessing, config["steps"])
                write_json(args.output / "averaged-adapter-evidence.json", average_evidence)
                publisher.publish()
            if not evaluated_adapter.is_dir():
                raise ValueError("Frozen evaluation adapter is missing")
            if args.defer_confirmation_evaluation:
                write_json(args.output / "training-completed.json", {
                    "completed": True,
                    "phase": args.phase,
                    "training_final_checkpoint": final_checkpoint,
                    "evaluation_checkpoint": checkpoint,
                    "selection": selected,
                    "test_used_for_selection": False,
                    "holdout_observed": False,
                    "time_budget": completed.get("time_budget"),
                    "postprocessing": postprocessing,
                    "evaluated_adapter_path": str(
                        evaluated_adapter.relative_to(args.output)),
                    "evaluated_adapter_file_sha256": digest(
                        evaluated_adapter / "adapter_model.safetensors"),
                    "evaluated_adapter_config_sha256": digest(
                        evaluated_adapter / "adapter_config.json"),
                    "primary_rule": (
                        "frozen adapter rule; independent evaluation deferred "
                        "until every paired training arm is terminal"),
                })
                return
            call("final", [
                "-m", "servedistill.evaluate", "--student", str(args.student),
                "--adapter", str(evaluated_adapter),
                "--data", str(args.data), "--phase", "final",
                "--output", str(args.output / "final"),
            ])
        write_json(args.output / "completed.json", {
            "completed": True, "phase": args.phase, "checkpoint": checkpoint,
            "training_final_checkpoint": final_checkpoint,
            "selection": selected, "test_used_for_selection": False,
            "time_budget": completed.get("time_budget"),
            "postprocessing": postprocessing if confirmation else None,
            "evaluated_adapter_file_sha256": digest(evaluated_adapter / "adapter_model.safetensors")
                if confirmation else None,
            "primary_rule": "final completed update under frozen budget" if args.phase == "confirmation"
                            else "existing development selection"})
    except Exception as error:
        write_json(args.output / "failed.json", {
            "error": type(error).__name__, "message": str(error),
            "unix_seconds": time.time()})
        raise
    finally:
        publisher.publish()


if __name__ == "__main__":
    main()
