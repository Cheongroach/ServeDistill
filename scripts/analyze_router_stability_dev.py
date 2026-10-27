"""Regrade every declared BF16 DEV rule and enforce all-seed selection gates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from servedistill.advantage_contracts import file_digest
from servedistill.data import grade, load_rows
from servedistill.runtime import summarize, write_json


def verify_predictions(root: Path, label: str, source: dict) -> tuple[list[dict], dict]:
    rows = load_rows(root / (label + ".jsonl"))
    if (len(rows) != len(source) or len({r["sample_id"] for r in rows}) != len(rows)
            or {r["sample_id"] for r in rows} != set(source)):
        raise ValueError("DEV prediction coverage changed: " + label)
    for row in rows:
        sample = source[row["sample_id"]]
        if (row["input_sha256"] != sample["input_sha256"] or row["domain"] != sample["domain"]
                or any(row.get(k) != v for k, v in grade(sample, row["text"]).items())):
            raise ValueError("Independent DEV regrading differs: " + label)
    summary = summarize(rows)
    if summary != json.loads((root / (label + ".summary.json")).read_text()):
        raise ValueError("Saved DEV summary changed")
    return rows, summary


def quality(rows: list[dict], gate: dict) -> dict:
    summary = summarize(rows)
    checks, classes = [], {}
    for domain, minimum in gate["minimum_metric"].items():
        part = [r for r in rows if r["domain"] == domain]
        value = summary["domains"][domain]["selection_score"]
        checks.append({"criterion": domain + "/metric", "value": value,
                       "minimum": minimum, "passed": value >= minimum})
        valid = sum(r["json_valid"] for r in part) / len(part)
        checks.append({"criterion": domain + "/json_valid", "value": valid,
                       "minimum": gate["minimum_json_valid"],
                       "passed": valid >= gate["minimum_json_valid"]})
        if domain not in gate["minimum_recall"]:
            continue
        target = Counter(r["target_class"] for r in part)
        predicted = Counter(r["predicted_class"] for r in part)
        per_class = {}
        for label, threshold in gate["minimum_recall"][domain].items():
            if not target[label]:
                raise ValueError("Required DEV class has no support")
            tp = sum(r["target_class"] == label and r["predicted_class"] == label for r in part)
            recall = tp / target[label]
            precision = tp / predicted[label] if predicted[label] else 0.0
            f1 = 2 * tp / (target[label] + predicted[label])
            per_class[label] = {"support": target[label], "predicted": predicted[label],
                                "true_positive": tp, "precision": precision,
                                "recall": recall, "f1": f1}
            checks.append({"criterion": domain + "/recall/" + label, "value": recall,
                           "minimum": threshold, "passed": recall >= threshold})
        classes[domain] = per_class
    return {"summary": summary, "classes": classes, "checks": checks,
            "absolute_gates_passed": all(c["passed"] for c in checks)}


def analyze(root: Path, contract_path: Path, dev: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    contract_hash = file_digest(contract_path)
    if (contract["scope"] != "development_only" or contract["holdout_files_packaged"]
            or contract["selection_precision"] != "bf16"
            or file_digest(dev) != contract["dev_sha256"]):
        raise ValueError("Declared DEV-only data binding changed")
    audit_path = root / "collection-audit.json"
    audit = json.loads(audit_path.read_text())
    if (not audit["verified"] or not audit["all_runs_terminal_successful"]
            or audit["scope"] != "development_only" or audit["holdout_observed"]
            or audit["contract_sha256"] != contract_hash):
        raise ValueError("Independent terminal adapter audit is required")
    for record in audit["files"]:
        if "path" in record:
            path = root / record["path"]
            if file_digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError("Collected evidence changed after verification")
    source_rows = load_rows(dev)
    selected = [r for r in source_rows if r["domain"] in ("search", "relevance", "substitute")]
    selected += [r for d in ("math", "instruction")
                 for r in [x for x in source_rows if x["domain"] == d][:100]]
    source = {r["sample_id"]: r for r in selected}
    if len(selected) != contract["dev_rows"] or len(source) != len(selected):
        raise ValueError("DEV workload coverage changed")
    substitute = {k: v for k, v in source.items() if v["domain"] == "substitute"}
    seeds, rules = contract["seeds"], contract["aopd_rules"]
    values, controls = {}, {}
    for seed in seeds:
        path = root / f"seed{seed}"
        completion = json.loads((path / "completed.json").read_text())
        worker = json.loads((path / "worker-completed.json").read_text())
        if (not worker["completed"] or not worker["assets_verified"]
                or any(r["returncode"] for r in worker["tasks"])
                or not completion["completed"] or completion["seed"] != seed
                or completion["contract_sha256"] != contract_hash
                or completion["scope"] != "development_only" or completion["holdout_observed"]
                or completion["precision"] != "bf16"):
            raise ValueError("Closed BF16 DEV worker evidence changed")
        offline, _ = verify_predictions(path, "offline_sft", source)
        oracle, _ = verify_predictions(path, "oracle_sft", source)
        controls[str(seed)] = {
            "offline_sft": quality(offline, contract["quality_gates"]),
            "oracle_sft": quality(oracle, contract["quality_gates"]),
        }
        values[str(seed)] = {}
        for label in rules:
            part, _ = verify_predictions(path, label, substitute)
            by_id = {r["sample_id"]: r for r in part}
            routed = [by_id[r["sample_id"]] if r["domain"] == "substitute" else r for r in offline]
            result = quality(routed, contract["quality_gates"])
            if result["summary"] != completion["routed_summaries"][label]:
                raise ValueError("Saved routed summary changed")
            result["delta_business_vs_offline"] = (
                result["summary"]["business_macro"]
                - controls[str(seed)]["offline_sft"]["summary"]["business_macro"])
            result["delta_business_vs_oracle"] = (
                result["summary"]["business_macro"]
                - controls[str(seed)]["oracle_sft"]["summary"]["business_macro"])
            result["invalid_substitute_ids"] = [r["sample_id"] for r in part if not r["json_valid"]]
            values[str(seed)][label] = result
    comparisons = {}
    for label in rules:
        by_seed = [values[str(seed)][label] for seed in seeds]
        deltas = [r["delta_business_vs_offline"] for r in by_seed]
        comparisons[label] = {
            "all_seed_absolute_gates_passed": all(r["absolute_gates_passed"] for r in by_seed),
            "mean_business_macro": float(np.mean([r["summary"]["business_macro"] for r in by_seed])),
            "mean_delta_business_vs_offline": float(np.mean(deltas)),
            "paired_seed_deltas_vs_offline": deltas,
            "positive_seed_pairs_vs_offline": sum(x > 0 for x in deltas),
            "mean_delta_business_vs_oracle": float(np.mean(
                [r["delta_business_vs_oracle"] for r in by_seed])),
            "substitute_f1_by_seed": [r["summary"]["domains"]["substitute"]["macro_f1"] for r in by_seed],
            "substitute_json_valid_by_seed": [
                r["summary"]["domains"]["substitute"]["json_valid"] for r in by_seed],
            "failed_checks_by_seed": {
                str(seed): [c for c in r["checks"] if not c["passed"]]
                for seed, r in zip(seeds, by_seed)},
        }
    eligible = [label for label in rules if comparisons[label]["all_seed_absolute_gates_passed"]]
    ranked = sorted(eligible, key=lambda label: comparisons[label]["mean_business_macro"], reverse=True)
    return {
        "scope": "exploratory_development_only", "independent_regrading_completed": True,
        "exact_adapter_average_audit_sha256": file_digest(audit_path),
        "contract_sha256": contract_hash, "dev_sha256": file_digest(dev),
        "seeds": seeds, "source_rows": len(selected),
        "prediction_rows": sum(len(source) * 2 + len(substitute) * len(rules) for _ in seeds),
        "rules": comparisons, "per_seed": values, "controls": controls,
        "eligible_rules_ranked_by_declared_dev_metric": ranked,
        "selected_dev_rule": ranked[0] if ranked else None,
        "quality_advantage_supported": False, "cost_advantage_supported": False,
        "boundary": "DEV selection after an observed failed confirmation. Repeated outputs "
                    "are not independent new queries; these results cannot establish a "
                    "holdout advantage. Keep all rules/seeds and the failed confirmation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("root", "contract", "dev", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.contract, args.dev)
    write_json(args.output, result)
    print(json.dumps({"selected_dev_rule": result["selected_dev_rule"],
                      "rules": result["rules"], "quality_advantage_supported": False}))


if __name__ == "__main__":
    main()
