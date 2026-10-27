"""Build and audit the DEV-only AOPD/offline domain router."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys


repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from servedistill.advantage_contracts import file_digest
from servedistill.data import grade, load_rows
from servedistill.runtime import summarize


reports = repo / "reports/advantage"
raw = reports / "raw/aopd-domain-router-v1"
output_predictions = raw / "dev.jsonl"
output_summary = raw / "dev.summary.json"
report_path = reports / "DEVELOPMENT_AOPD_DOMAIN_ROUTER.json"
audit_path = reports / "BINARY_AUDIT_AOPD_DOMAIN_ROUTER.json"

source_paths = {
    "aopd_step320": (
        reports
        / "raw/aopd-v1/aopd_tau0_k32_trajectory_lr5e5/selected-dev.jsonl"
    ),
    "offline_sft_all4": (
        reports
        / "raw/decay640-sft-v1/averaged-development/"
        "decay_offline_trajectory_lr5e5_average_all4/dev.jsonl"
    ),
}
source_summaries = {
    "aopd_step320": source_paths["aopd_step320"].with_suffix(".summary.json"),
    "offline_sft_all4": source_paths["offline_sft_all4"].with_suffix(
        ".summary.json"
    ),
}
source_completions = {
    "aopd_step320": (
        reports
        / "raw/aopd-v1/aopd_tau0_k32_trajectory_lr5e5/completed.json"
    ),
    "offline_sft_all4": source_paths["offline_sft_all4"].with_name(
        "completed.json"
    ),
}
routing = {
    "substitute": "aopd_step320",
    "search": "offline_sft_all4",
    "relevance": "offline_sft_all4",
    "math": "offline_sft_all4",
    "instruction": "offline_sft_all4",
}
thresholds = {
    "offline_sft": 0.7038299984194807,
    "oracle_sft": 0.687469949849396,
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    temporary.replace(path)


def development_source() -> dict[str, dict]:
    rows = load_rows(repo / "data_advantage_screen/dev.jsonl")
    selected = [
        row
        for row in rows
        if row["domain"] in ("search", "relevance", "substitute")
    ] + [
        row
        for domain in ("math", "instruction")
        for row in [value for value in rows if value["domain"] == domain][:100]
    ]
    source = {row["sample_id"]: row for row in selected}
    if len(selected) != 1100 or len(source) != 1100:
        raise ValueError("Router DEV source coverage drift")
    return source


def verify_source(name: str, source: dict[str, dict]) -> dict[str, dict]:
    path = source_paths[name]
    rows = load_rows(path)
    indexed = {row["sample_id"]: row for row in rows}
    if len(rows) != 1100 or len(indexed) != 1100 or set(indexed) != set(source):
        raise ValueError(f"{name} prediction coverage drift")
    for sample_id, row in indexed.items():
        sample = source[sample_id]
        if (
            row["input_sha256"] != sample["input_sha256"]
            or row["domain"] != sample["domain"]
        ):
            raise ValueError(f"{name} prediction identity drift")
        if any(
            row.get(key) != value
            for key, value in grade(sample, row["text"]).items()
        ):
            raise ValueError(f"{name} independent regrading differs")
    summary = summarize(rows)
    if summary != json.loads(source_summaries[name].read_text()):
        raise ValueError(f"{name} summary drift")
    completion = json.loads(source_completions[name].read_text())
    if not completion["completed"]:
        raise ValueError(f"{name} source is incomplete")
    if name == "offline_sft_all4" and (
        completion["rows"] != 1100
        or completion["precision"] != "bf16"
        or completion["holdout_observed"]
    ):
        raise ValueError("Offline SFT completion drift")
    if name == "aopd_step320" and (
        completion["phase"] != "development"
        or completion["selection"]["step"] != 320
        or completion["test_used_for_selection"]
    ):
        raise ValueError("AOPD selected checkpoint drift")
    return indexed


def compose_rows(
    sources: dict[str, dict[str, dict]], route: dict[str, str]
) -> list[dict]:
    if set(route) != {"substitute", "search", "relevance", "math", "instruction"}:
        raise ValueError("Router domain contract drift")
    names = list(sources)
    if set(route.values()) - set(names):
        raise ValueError("Router references an unknown source")
    identities = set(sources[names[0]])
    if any(set(source) != identities for source in sources.values()):
        raise ValueError("Router source coverage differs")
    rows = []
    for sample_id in sorted(identities):
        reference = sources[names[0]][sample_id]
        domain = reference["domain"]
        selected = sources[route[domain]][sample_id]
        if any(
            source[sample_id]["domain"] != domain
            or source[sample_id]["input_sha256"] != reference["input_sha256"]
            for source in sources.values()
        ):
            raise ValueError("Router source identity differs")
        rows.append({**selected, "router_source": route[domain]})
    return rows


def build_candidate() -> tuple[list[dict], dict, dict]:
    source = development_source()
    sources = {
        name: verify_source(name, source)
        for name in source_paths
    }
    rows = compose_rows(sources, routing)
    for row in rows:
        sample = source[row["sample_id"]]
        if any(
            row.get(key) != value
            for key, value in grade(sample, row["text"]).items()
        ):
            raise ValueError("Routed prediction independent regrading differs")
    summary = summarize(rows)
    provenance = {
        name: {
            "predictions_path": str(path.relative_to(repo)),
            "predictions_sha256": file_digest(path),
            "summary_sha256": file_digest(source_summaries[name]),
            "completion_sha256": file_digest(source_completions[name]),
        }
        for name, path in source_paths.items()
    }
    return rows, summary, provenance


def report() -> None:
    rows, summary, provenance = build_candidate()
    atomic_jsonl(output_predictions, rows)
    atomic_json(output_summary, summary)
    deltas = {
        name: summary["business_macro"] - value
        for name, value in thresholds.items()
    }
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate": {
            "name": "aopd_step320_substitute_router_offline_all4",
            "kind": "request_task_domain_adapter_router",
            "routing": routing,
            "routing_key": "request task type; never target, prediction, or grader output",
            "business_macro": summary["business_macro"],
            "general_macro": summary["general_macro"],
            "selection_score": (
                0.8 * summary["business_macro"]
                + 0.2 * summary["general_macro"]
            ),
            "domains": summary["domains"],
            "predictions_path": str(output_predictions.relative_to(repo)),
            "predictions_sha256": file_digest(output_predictions),
        },
        "sources": provenance,
        "retained_thresholds": thresholds,
        "selected_dev_business_deltas": deltas,
        "declared_candidates": 1,
        "completed_candidates": 1,
        "complete_selection_pool": True,
        "independent_regrading_completed": True,
        "quality_advantage_established": all(value > 0 for value in deltas.values()),
        "posthoc_development_design": True,
        "holdout_observed": False,
        "boundary": (
            "Observed DEV only. The fixed task-domain router was designed after "
            "AOPD DEV analysis; independent holdouts remain sealed."
        ),
    }
    atomic_json(report_path, result)
    print(
        json.dumps(
            {
                "business_macro": summary["business_macro"],
                "general_macro": summary["general_macro"],
                "selected_dev_business_deltas": deltas,
                "quality_advantage_established": result[
                    "quality_advantage_established"
                ],
                "holdout_observed": False,
            }
        )
    )


def audit() -> None:
    result = json.loads(report_path.read_text())
    rows, summary, provenance = build_candidate()
    materialized = load_rows(output_predictions)
    if materialized != rows or json.loads(output_summary.read_text()) != summary:
        raise ValueError("Materialized router candidate drift")
    candidate = result["candidate"]
    if (
        result["declared_candidates"] != 1
        or result["completed_candidates"] != 1
        or not result["complete_selection_pool"]
        or not result["independent_regrading_completed"]
        or result["holdout_observed"]
        or result["sources"] != provenance
        or candidate["routing"] != routing
        or candidate["business_macro"] != summary["business_macro"]
        or candidate["general_macro"] != summary["general_macro"]
        or candidate["predictions_sha256"] != file_digest(output_predictions)
    ):
        raise ValueError("Router report contract drift")
    deltas = {
        name: summary["business_macro"] - value
        for name, value in thresholds.items()
    }
    if result["selected_dev_business_deltas"] != deltas:
        raise ValueError("Router threshold comparison drift")
    if result["quality_advantage_established"] != all(
        value > 0 for value in deltas.values()
    ):
        raise ValueError("Router advantage decision drift")
    audit = {
        "verified": True,
        "candidate": candidate["name"],
        "report_sha256": file_digest(report_path),
        "predictions_sha256": file_digest(output_predictions),
        "summary_sha256": file_digest(output_summary),
        "source_prediction_sha256": {
            name: value["predictions_sha256"]
            for name, value in provenance.items()
        },
        "routing": routing,
        "quality_advantage_established": result["quality_advantage_established"],
        "holdout_observed": False,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_json(audit_path, audit)
    print(
        json.dumps(
            {
                "verified": True,
                "quality_advantage_established": result[
                    "quality_advantage_established"
                ],
                "holdout_observed": False,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("report", "audit"))
    args = parser.parse_args()
    if args.action == "report":
        report()
    else:
        audit()


if __name__ == "__main__":
    main()
