"""Independently regrade both domain CE candidates and all original paired controls."""
import argparse
import json
import math
from pathlib import Path

from scripts.analyze_anchor_development import advance_rule
from scripts.analyze_router_stability_dev import quality, verify_predictions
from scripts.collect_domain_anchor_development import (
    verify_collection, verify_historical_controls, verify_seed_candidate,
)
from scripts.run_domain_anchor_development import load_protocol
from servedistill.advantage_contracts import file_digest
from servedistill.advantage_statistics import paired_quality
from servedistill.data import load_rows
from servedistill.runtime import write_json


def select_candidate(results, protocol):
    if set(results) != set(protocol["variants"]):
        raise ValueError("Both domain candidates required before DEV selection")
    if any(not math.isfinite(r["mean_dev_business_macro"]) for r in results.values()):
        raise ValueError("Non-finite DEV score")
    eligible = sorted((n for n, r in results.items() if r["advance_to_fresh_confirmation"]),
                      key=lambda n: (-results[n]["mean_dev_business_macro"],
                                     protocol["variants"][n]["search"]))
    return eligible, eligible[0] if eligible else None


def analyze(base_root, root, base_protocol, protocol_path, training, dev, repetitions=5000):
    base, protocol = load_protocol(base_protocol, protocol_path)
    if file_digest(training) != base["train_sha256"] or file_digest(dev) != base["dev_sha256"]:
        raise ValueError("Original train/DEV input changed")
    protocol_hash = file_digest(protocol_path)
    audit = verify_collection(root, base, protocol, protocol_hash)
    training_rows = load_rows(training)
    historic = verify_historical_controls(base_root, base, protocol, training_rows)
    dev_rows = load_rows(dev)
    selected = [r for r in dev_rows if r["domain"] in ("search", "relevance", "substitute")] + [
        r for domain in ("math", "instruction")
        for r in [r for r in dev_rows if r["domain"] == domain][:100]]
    source = {r["sample_id"]: r for r in selected}
    if len(source) != 1100 or len(selected) != 1100:
        raise ValueError("Full DEV coverage changed")
    controls = {name: [] for name in ("offline_sft", "oracle_sft", "initial")}
    per_seed, results = {}, {}
    for seed in protocol["seeds"]:
        per_seed[str(seed)] = {}
        study = base_root / f"seed{seed}/study"
        completed = json.loads((study / "completed.json").read_text())
        for name in controls:
            arm, label = (study, "initial-dev") if name == "initial" else (study / name, "final-dev")
            rows, summary = verify_predictions(arm, label, source)
            if summary != completed["summaries"][name]:
                raise ValueError("Historical control scoring changed")
            controls[name].append(rows)
            per_seed[str(seed)][name] = quality(rows, base["quality_gates"])
    for variant, weights in protocol["variants"].items():
        predictions = []
        for seed in protocol["seeds"]:
            candidate_root = root / variant / f"seed{seed}"
            checked = verify_seed_candidate(candidate_root, base, protocol, protocol_hash,
                                            variant, seed, training_rows)
            if checked != audit["adapter_checks"][f"{variant}/seed{seed}"]:
                raise ValueError("Domain adapter audit changed")
            if checked["prompt_schedule_sha256"] != historic[str(seed)]["offline_sft"]["prompt_schedule_sha256"]:
                raise ValueError("Candidate and historical control schedule differ")
            rows, summary = verify_predictions(candidate_root / "study", "final-dev", source)
            if summary != json.loads((candidate_root / "study/completed.json").read_text())["summary"]:
                raise ValueError("Domain completion scoring changed")
            predictions.append(rows)
            per_seed[str(seed)][variant] = quality(rows, base["quality_gates"])
        comparisons = {name: paired_quality(rows, predictions, base["quality_gates"],
                                            repetitions, 202610270 + i)
                       for i, (name, rows) in enumerate(controls.items())}
        for comparison in comparisons.values():
            comparison["math_regression_noninferior"] = comparison["svamp_regression_noninferior"]
            comparison["math_source_boundary"] = "Observed mixed DEV; not fresh confirmation."
        absolute = all(per_seed[str(s)][variant]["absolute_gates_passed"] for s in protocol["seeds"])
        results[variant] = {
            "anchor_ce_domain_weights": weights, "comparisons": comparisons,
            "mean_dev_business_macro": sum(
                per_seed[str(s)][variant]["summary"]["business_macro"] for s in protocol["seeds"]) / 4,
            "candidate_all_seed_absolute_gates_passed": absolute,
            "advance_to_fresh_confirmation": advance_rule(comparisons, absolute, base["quality_gates"]),
        }
    eligible, chosen = select_candidate(results, protocol)
    costs = {name: json.loads((p / "cost.json").read_text())
             for name, p in (("historical_controls", base_root), ("new_domain_runs", root))}
    pods = [{r["trial_pod_sid"] for r in c["records"]} for c in costs.values()]
    if pods[0] & pods[1]:
        raise ValueError("Historical and new allocation cost overlap")
    return {
        "scope": "development_only", "holdout_observed": False,
        "protocol_sha256": protocol_hash, "independent_regrading_completed": True,
        "prediction_rows_regraded": 22000, "candidates": results, "per_seed": per_seed,
        "eligible_candidates_ranked": eligible, "selected_dev_candidate": chosen,
        "advance_to_fresh_confirmation": chosen is not None, "quality_advantage_supported": False,
        "cost_advantage_supported": False,
        "allocated_gpu_hours": {n: c["allocated_gpu_hours"] for n, c in costs.items()},
        "boundary": "Both candidates and original controls regraded at four paired seeds. "
                    "Historical costs are reference allocations, not charged again. "
                    "An eligible DEV candidate still requires prospective fresh confirmation.",
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("base-root", "root", "base-protocol", "protocol", "training", "dev", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    result = analyze(args.base_root, args.root, args.base_protocol, args.protocol,
                     args.training, args.dev, args.repetitions)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("selected_dev_candidate", "quality_advantage_supported")}))


if __name__ == "__main__":
    main()
