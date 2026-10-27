"""Regrade complete DEV evidence, consume an attempt, then freeze fresh inputs."""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

from scripts.analyze_anchor_weights import analyze
from servedistill.advantage_contracts import file_digest
from servedistill.fresh_confirmation import prepare
from servedistill.sequential_confirmation import (
    attempt_alpha, locked_ledger, read_attempts, reserve_attempt,
)


def freeze(args) -> dict:
    family = json.loads(args.family.read_text())
    if args.source.name != Path(family["source_lock"]).name:
        raise ValueError("Fresh source lock differs from family declaration")
    if (len(args.previous) != 5
            or {p.name for p in args.previous} != set(family["historical_input_roots_required"])):
        raise ValueError("Provide all five original historical input roots")
    # The caller cannot substitute a cached true flag for complete independent regrading.
    result = analyze(args.base_root, args.weight_root, args.protocol, args.extension,
                     args.training, args.dev)
    chosen = result["selected_dev_candidate"]
    if (chosen is None or not result["advance_to_fresh_confirmation"]
            or not result["all_three_weights_and_controls_complete"]
            or result["seeds"] != family["paired_training_seeds"]):
        raise ValueError("No eligible complete DEV candidate; fresh inputs remain untouched")
    base = json.loads(args.protocol.read_text())
    adapters = {}
    for seed in result["seeds"]:
        paths = {
            control: args.base_root / f"seed{seed}/study/{control}/train/checkpoint-0320"
            for control in ("offline_sft", "oracle_sft")}
        paths["candidate"] = (
            args.base_root / f"seed{seed}/study/aopd_anchor/train/checkpoint-0320"
            if chosen == "aopd_anchor" else
            args.weight_root / chosen / f"seed{seed}/study/train/checkpoint-0320")
        adapters[str(seed)] = {
            name: {file: file_digest(path / file)
                   for file in ("adapter_model.safetensors", "adapter_config.json", "state.json")}
            for name, path in paths.items()}
    with locked_ledger(args.ledger):
        previous_attempts = read_attempts(args.ledger, file_digest(args.family))
        attempt = len(previous_attempts) + 1
        alpha = attempt_alpha(family["total_one_sided_alpha"], attempt)
        repetitions = max(20000, math.ceil(
            family["bootstrap_minimum_expected_tail_draws"] / alpha))
        directory = args.ledger / f"packet-{attempt:04d}"
        if directory.exists():
            raise FileExistsError("Existing packet without a reservation requires investigation")
        historical = list(args.previous)
        for receipt in previous_attempts:
            root = args.ledger / f"packet-{receipt['attempt']:04d}/data"
            # Failed preparation still consumes alpha; any materialized inputs are excluded.
            if list(root.glob("*.jsonl")):
                historical.append(root)
        bindings = {
            "selected_dev_candidate": chosen, "adapters": adapters,
            "business_count_per_domain_per_batch": args.business_count,
            "math_count_per_batch": args.math_count,
            "sample_size_rationale": args.sample_size_rationale,
            "source_lock_sha256": file_digest(args.source),
        }
        receipt = reserve_attempt(args.ledger, args.family, bindings)
        directory.mkdir()
        (directory / "reservation.json").write_bytes(
            (args.ledger / f"attempt-{attempt:04d}.json").read_bytes())
        (directory / "dev-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        # Failure leaves the immutable reservation in place, including alpha consumption.
        manifest = prepare(
            args.examples, args.products, args.gsm8k, args.source, historical,
            directory / "data", args.business_count, args.math_count,
            family["sampling_seed_base"] + attempt)
        repo = Path(__file__).resolve().parents[1]
        source_names = [
            "scripts/freeze_fresh_confirmation.py", "scripts/analyze_anchor_weights.py",
            "scripts/audit_fresh_pools.py", "servedistill/fresh_confirmation.py",
            "servedistill/sequential_confirmation.py", "servedistill/statistics.py",
            "servedistill/expanded_statistics.py", "servedistill/search_data.py",
            "servedistill/data.py", "servedistill/fresh_contracts.py",
            "servedistill/advantage_statistics.py", "servedistill/advantage_contracts.py",
            "servedistill/runtime.py", "scripts/evaluate_fresh_confirmation.py",
            "scripts/analyze_fresh_confirmation.py",
        ]
        protocol = {
            "scope": "frozen_fresh_confirmation", "frozen_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "attempt": attempt, "alpha": alpha, "family_sha256": file_digest(args.family),
            "reservation_sha256": file_digest(args.ledger / f"attempt-{attempt:04d}.json"),
            "dev_analysis_sha256": file_digest(directory / "dev-analysis.json"),
            "data_manifest_sha256": file_digest(directory / "data/manifest.json"),
            "data_splits": manifest["splits"], "adapters": adapters,
            "student_manifest_sha256": base["common_student_manifest_sha256"],
            "selected_dev_candidate": chosen, "paired_training_seeds": result["seeds"],
            "quality_gates": base["quality_gates"],
            "bootstrap_repetitions": repetitions,
            "bootstrap_chunk_size": family["bootstrap_chunk_size"],
            "bootstrap_seed": family["sampling_seed_base"] + 10000 + attempt,
            "source_sha256": {name: file_digest(repo / name) for name in source_names},
            "required_controls": family["required_controls"],
            "required_batches": family["required_batches"],
            "model_predictions_observed_before_freeze": False,
            "sample_size_rationale": args.sample_size_rationale,
            "acceptance_rule": family["acceptance_rule"],
            "coverage_boundary": family["coverage_boundary"],
            "population_boundary": family["population_boundary"],
        }
        (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        return {"packet": str(directory), "attempt": receipt["attempt"], "alpha": alpha,
                "candidate": chosen, "bootstrap_repetitions": repetitions}


def main():
    parser = argparse.ArgumentParser()
    for name in ("family", "source", "base-root", "weight-root", "protocol", "extension",
                 "training", "dev", "examples", "products", "gsm8k", "ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous", type=Path, nargs="+", required=True)
    parser.add_argument("--business-count", type=int, required=True)
    parser.add_argument("--math-count", type=int, required=True)
    parser.add_argument("--sample-size-rationale", required=True)
    args = parser.parse_args()
    if not args.sample_size_rationale.strip():
        raise ValueError("Record the DEV-based sample-size rationale before predictions")
    print(json.dumps(freeze(args)))


if __name__ == "__main__":
    main()
