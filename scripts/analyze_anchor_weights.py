"""Regrade all three anchor weights and controls before the frozen DEV choice."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.analyze_anchor_development import advance_rule, analyze as analyze_base
from scripts.analyze_router_stability_dev import quality, verify_predictions
from scripts.collect_anchor_weights import load_protocols, verify_collection, verify_weight_seed
from servedistill.advantage_contracts import file_digest
from servedistill.advantage_statistics import paired_quality
from servedistill.data import load_rows
from servedistill.runtime import write_json


def select_candidate(results: dict, weights: dict[str, float]) -> tuple[list[str], str | None]:
    if set(results) != set(weights) or len(weights) != 3:
        raise ValueError("All three weight results are required before selection")
    if any(not math.isfinite(r["mean_dev_business_macro"]) for r in results.values()):
        raise ValueError("Non-finite DEV score")
    ranked = sorted((name for name in weights if results[name]["advance_to_fresh_confirmation"]),
                    key=lambda name: (-results[name]["mean_dev_business_macro"], weights[name]))
    return ranked, ranked[0] if ranked else None


def analyze(base_root: Path, weight_root: Path, protocol_path: Path, extension_path: Path,
            training: Path, dev: Path, repetitions: int = 5000) -> dict:
    base, extension = load_protocols(protocol_path, extension_path)
    if file_digest(training) != base["train_sha256"]:
        raise ValueError("Original paired training input changed")
    extension_hash = file_digest(extension_path)
    weight_audit = verify_collection(weight_root, base, extension, extension_hash)
    base_result = analyze_base(base_root, protocol_path, dev, repetitions)
    source_rows = load_rows(dev)
    selected = [r for r in source_rows if r["domain"] in ("search", "relevance", "substitute")]
    selected += [r for domain in ("math", "instruction")
                 for r in [r for r in source_rows if r["domain"] == domain][:100]]
    source = {r["sample_id"]: r for r in selected}
    training_rows = load_rows(training)
    controls = {name: [] for name in ("offline_sft", "oracle_sft", "initial")}
    for seed in base["seeds"]:
        study = base_root / f"seed{seed}/study"
        # analyze_base has already independently regraded and bound these exact files.
        for name in controls:
            path = study / ("initial-dev.jsonl" if name == "initial" else name + "/final-dev.jsonl")
            controls[name].append(load_rows(path))
    per_seed = base_result["per_seed"]
    results = {"aopd_anchor": {
        "anchor_ce_weight": 0.5,
        "mean_dev_business_macro": sum(
            per_seed[str(s)]["aopd_anchor"]["summary"]["business_macro"] for s in base["seeds"]) / 4,
        "candidate_all_seed_absolute_gates_passed":
            base_result["candidate_all_seed_absolute_gates_passed"],
        "comparisons": base_result["comparisons"],
        "advance_to_fresh_confirmation": base_result["advance_to_fresh_confirmation"],
    }}
    for variant, spec in extension["variants"].items():
        predictions = []
        for seed in base["seeds"]:
            root = weight_root / variant / f"seed{seed}"
            checked = verify_weight_seed(root, base, extension, extension_hash,
                                         variant, seed, training_rows)
            if checked != weight_audit["adapter_checks"][f"{variant}/seed{seed}"]:
                raise ValueError("Weight adapter audit changed")
            base_binding = json.loads((base_root / f"seed{seed}/study/binding.json").read_text())
            if checked["prompt_schedule_sha256"] != base_binding["prompt_schedule_sha256"]:
                raise ValueError("Extension and original control prompt schedules differ")
            rows, summary = verify_predictions(root / "study", "final-dev", source)
            if summary != json.loads((root / "study/completed.json").read_text())["summary"]:
                raise ValueError("Weight completion summary changed")
            predictions.append(rows)
            per_seed[str(seed)][variant] = quality(rows, base["quality_gates"])
        # Common random query draws across weight candidates; the metric/gates stay fixed.
        comparisons = {
            name: paired_quality(rows, predictions, base["quality_gates"],
                                 repetitions, 202610270 + i)
            for i, (name, rows) in enumerate(controls.items())}
        for comparison in comparisons.values():
            comparison["math_regression_noninferior"] = comparison["svamp_regression_noninferior"]
            comparison["math_source_boundary"] = "Observed DEV mix; not new SVAMP confirmation."
        absolute = all(per_seed[str(s)][variant]["absolute_gates_passed"] for s in base["seeds"])
        results[variant] = {
            **spec, "mean_dev_business_macro": sum(
                per_seed[str(s)][variant]["summary"]["business_macro"] for s in base["seeds"]) / 4,
            "candidate_all_seed_absolute_gates_passed": absolute, "comparisons": comparisons,
            "advance_to_fresh_confirmation": advance_rule(comparisons, absolute, base["quality_gates"]),
        }
    ranked, chosen = select_candidate(
        results, {name: r["anchor_ce_weight"] for name, r in results.items()})
    base_cost = json.loads((base_root / "cost.json").read_text())
    weight_cost = json.loads((weight_root / "cost.json").read_text())
    base_pods = {r["trial_pod_sid"] for r in base_cost["records"]}
    if base_pods & {r["trial_pod_sid"] for r in weight_cost["records"]}:
        raise ValueError("Original and extension allocation evidence overlap")
    return {
        "scope": "development_only", "holdout_observed": False,
        "independent_regrading_completed": True, "all_three_weights_and_controls_complete": True,
        "protocol_sha256": file_digest(protocol_path), "extension_sha256": extension_hash,
        "collection_audits": {
            "original": file_digest(base_root / "collection-audit.json"),
            "weights": file_digest(weight_root / "collection-audit.json")},
        "prediction_rows_regraded": base_result["prediction_rows_regraded"] + 8 * 1100,
        "seeds": base["seeds"], "candidates": results, "per_seed": per_seed,
        "eligible_candidates_ranked": ranked, "selected_dev_candidate": chosen,
        "advance_to_fresh_confirmation": chosen is not None,
        "quality_advantage_supported": False, "cost_advantage_supported": False,
        "dev_allocated_gpu_hours": base_cost["allocated_gpu_hours"] + weight_cost["allocated_gpu_hours"],
        "cost_sha256": {"original": file_digest(base_root / "cost.json"),
                        "weights": file_digest(weight_root / "cost.json")},
        "boundary": "All fixed update320 BF16 DEV outputs retained, including the pure AOPD "
                    "ablation. Ranking only follows the frozen extension gate and tie rule. "
                    "An eligible DEV choice does not establish a fresh confirmation advantage. "
                    "Cost covers these twelve runs; historical preparation and profiling remain separate.",
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("base-root", "weight-root", "protocol", "extension", "training", "dev", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    result = analyze(args.base_root, args.weight_root, args.protocol, args.extension,
                     args.training, args.dev, args.repetitions)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "prediction_rows_regraded", "selected_dev_candidate",
        "advance_to_fresh_confirmation", "quality_advantage_supported")}))


if __name__ == "__main__":
    main()
