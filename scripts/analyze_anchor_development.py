"""Regrade all five DEV outputs per seed before applying the frozen advance rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_router_stability_dev import quality, verify_predictions
from servedistill.advantage_contracts import file_digest
from servedistill.advantage_statistics import paired_quality
from servedistill.data import load_rows
from servedistill.runtime import write_json


def advance_rule(comparisons: dict, absolute_passed: bool, gate: dict) -> bool:
    for name in ("offline_sft", "oracle_sft"):
        macro = comparisons[name]["business_macro"]
        if (macro["delta"] <= 0
                or macro["positive_seed_pairs"] < gate["minimum_positive_seed_pairs"]):
            return False
    return bool(absolute_passed and all(
        c["all_business_domains_noninferior"] and c["svamp_regression_noninferior"]
        for c in comparisons.values()))


def analyze(root: Path, protocol_path: Path, dev: Path, repetitions: int = 5000) -> dict:
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = file_digest(protocol_path)
    audit = json.loads((root / "collection-audit.json").read_text())
    if (protocol["scope"] != "development_only" or protocol["holdout_files_packaged"]
            or protocol["holdout_observed_in_this_round"] or len(protocol["seeds"]) != 4
            or file_digest(dev) != protocol["dev_sha256"]
            or not audit["verified"] or not audit["all_runs_terminal_successful"]
            or audit["protocol_sha256"] != protocol_hash
            or audit["scope"] != "development_only" or audit["holdout_observed"]):
        raise ValueError("Verified closed four-seed DEV evidence is required")
    for record in audit["files"]:
        path = root / record["path"]
        if file_digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Collected evidence changed after adapter verification")
    if file_digest(root / "cost.json") != audit["cost_sha256"]:
        raise ValueError("Allocation cost evidence changed")
    source_rows = load_rows(dev)
    selected = [r for r in source_rows if r["domain"] in ("search", "relevance", "substitute")]
    selected += [r for domain in ("math", "instruction")
                 for r in [r for r in source_rows if r["domain"] == domain][:100]]
    source = {r["sample_id"]: r for r in selected}
    if len(selected) != 1100 or len(source) != 1100:
        raise ValueError("Full DEV coverage changed")
    rows_by_method = {name: [] for name in ("initial", *protocol["methods"])}
    per_seed = {}
    for seed in protocol["seeds"]:
        study = root / f"seed{seed}/study"
        completed = json.loads((study / "completed.json").read_text())
        per_seed[str(seed)] = {}
        for name in rows_by_method:
            arm, label = (study, "initial-dev") if name == "initial" else (study / name, "final-dev")
            rows, summary = verify_predictions(arm, label, source)
            if summary != completed["summaries"][name]:
                raise ValueError("Final DEV summary changed")
            per_seed[str(seed)][name] = quality(rows, protocol["quality_gates"])
            rows_by_method[name].append(rows)
    comparisons = {
        name: paired_quality(rows_by_method[name], rows_by_method["aopd_anchor"],
                             protocol["quality_gates"], repetitions, 202610270 + index)
        for index, name in enumerate(("offline_sft", "oracle_sft", "initial"))
    }
    # The legacy helper key names SVAMP; this DEV math set is GSM8K + SVAMP.
    for comparison in comparisons.values():
        comparison["math_regression_noninferior"] = comparison["svamp_regression_noninferior"]
        comparison["math_source_boundary"] = "Observed DEV mix; not new SVAMP confirmation."
    absolute = all(per_seed[str(seed)]["aopd_anchor"]["absolute_gates_passed"]
                   for seed in protocol["seeds"])
    advance = advance_rule(comparisons, absolute, protocol["quality_gates"])
    return {
        "scope": "development_only", "holdout_observed": False,
        "independent_regrading_completed": True, "protocol_sha256": protocol_hash,
        "collection_audit_sha256": file_digest(root / "collection-audit.json"),
        "prediction_rows_regraded": 1100 * len(rows_by_method) * len(protocol["seeds"]),
        "per_seed": per_seed, "comparisons": comparisons,
        "candidate_all_seed_absolute_gates_passed": absolute,
        "advance_to_fresh_confirmation": advance, "quality_advantage_supported": False,
        "boundary": "Frozen final update320 only. All arms and seeds retained. A DEV advance "
                    "is permission to freeze new independent confirmation, not quality evidence. "
                    "Noninferiority uses conditional paired-query intervals at the declared margins.",
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("root", "protocol", "dev", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    result = analyze(args.root, args.protocol, args.dev, args.repetitions)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "prediction_rows_regraded", "candidate_all_seed_absolute_gates_passed",
        "advance_to_fresh_confirmation", "quality_advantage_supported")}))


if __name__ == "__main__":
    main()
