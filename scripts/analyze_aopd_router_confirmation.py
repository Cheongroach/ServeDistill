"""Independently regrade the frozen routed candidate on both holdout splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servedistill.advantage_contracts import file_digest, verify_data
from servedistill.advantage_statistics import paired_sft_quality
from servedistill.data import grade, load_rows
from servedistill.runtime import summarize, write_json


def verify_predictions(
    root: Path,
    label: str,
    source: dict[str, dict],
    expected_domains: set[str],
    protocol_hash: str,
    split: str,
) -> tuple[list[dict], dict]:
    provenance_path = root / "evaluation-provenance.json"
    completion_path = root / "completed.json"
    provenance = json.loads(provenance_path.read_text())
    completion = json.loads(completion_path.read_text())
    if (
        provenance["label"] != label
        or provenance["protocol_sha256"] != protocol_hash
        or provenance["test_used_for_selection"]
        or set(provenance["evaluation_domains"]) != expected_domains
        or not completion["completed"]
        or completion["label"] != label
        or completion["protocol_sha256"] != protocol_hash
        or completion["test_used_for_selection"]
        or not completion["holdout_observed"]
    ):
        raise ValueError("Evaluation provenance changed: " + label)
    rows = load_rows(root / (split + ".jsonl"))
    expected = {
        identity for identity, row in source.items()
        if row["domain"] in expected_domains
    }
    if (
        len(rows) != len(expected)
        or len({row["sample_id"] for row in rows}) != len(rows)
        or {row["sample_id"] for row in rows} != expected
    ):
        raise ValueError("Independent prediction coverage changed: " + label)
    for row in rows:
        sample = source[row["sample_id"]]
        if (
            row["input_sha256"] != sample["input_sha256"]
            or row["domain"] != sample["domain"]
            or any(
                row.get(key) != value
                for key, value in grade(sample, row["text"]).items()
            )
        ):
            raise ValueError("Independent regrading differs: " + label)
    summary = summarize(rows)
    if summary != json.loads((root / (split + ".summary.json")).read_text()):
        raise ValueError("Saved split summary changed: " + label)
    return rows, {
        "predictions_sha256": file_digest(root / (split + ".jsonl")),
        "summary_sha256": file_digest(root / (split + ".summary.json")),
        "provenance_sha256": file_digest(provenance_path),
        "completion_sha256": file_digest(completion_path),
        "rows": len(rows),
        "domains": sorted(expected_domains),
    }


def compose_candidate(
    aopd: list[dict], offline: list[dict], route: dict[str, str]
) -> list[dict]:
    if route != {
        "substitute": "aopd",
        "search": "offline_sft",
        "relevance": "offline_sft",
        "math": "offline_sft",
    }:
        raise ValueError("Frozen router changed")
    sources = {
        "aopd": {row["sample_id"]: row for row in aopd},
        "offline_sft": {row["sample_id"]: row for row in offline},
    }
    result = []
    for row in offline:
        selected = sources[route[row["domain"]]].get(row["sample_id"])
        if selected is None:
            raise ValueError("Routed source coverage is incomplete")
        if (
            selected["domain"] != row["domain"]
            or selected["input_sha256"] != row["input_sha256"]
        ):
            raise ValueError("Routed source identity changed")
        result.append({**selected, "router_source": route[row["domain"]]})
    return result


def analyze(root: Path, data: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = file_digest(protocol_path)
    if protocol["test_observed_before_freeze"]:
        raise ValueError("Protocol was not frozen before holdout evaluation")
    verification = verify_data(data, confirmation=True)
    evidence, results, oracle_results = {}, {}, {}
    support = {}
    for split_index, split in enumerate(("test", "ood")):
        source = {
            row["sample_id"]: row for row in load_rows(data / (split + ".jsonl"))
        }
        offline_rows, oracle_rows, candidates = [], [], []
        for seed in protocol["paired_seeds"]:
            values = {}
            for family, domains in (
                ("aopd", {"substitute"}),
                ("offline_sft", {"search", "relevance", "substitute", "math"}),
                ("oracle_sft", {"search", "relevance", "substitute", "math"}),
            ):
                label = family + "_seed" + str(seed)
                rows, record = verify_predictions(
                    root / ("seed" + str(seed)) / label,
                    label, source, domains, protocol_hash, split,
                )
                evidence.setdefault(label, {})[split] = record
                values[family] = rows
            offline_rows.append(values["offline_sft"])
            oracle_rows.append(values["oracle_sft"])
            candidates.append(compose_candidate(
                values["aopd"], values["offline_sft"], protocol["routing"]
            ))
        compared = paired_sft_quality(
            offline_rows,
            oracle_rows,
            candidates,
            protocol["quality_gates"],
            repetitions=protocol["bootstrap_repetitions"],
            seed=protocol["bootstrap_seed"] + split_index,
        )
        results[split] = compared["primary"]
        oracle_results[split] = compared["oracle_guard"]
        support[split] = compared["quality_advantage_supported"]
    return {
        "protocol_sha256": protocol_hash,
        "data_verification": verification,
        "paired_seeds": protocol["paired_seeds"],
        "routing": protocol["routing"],
        "independent_regrading_completed": True,
        "arm_evidence": evidence,
        "primary_test": results["test"],
        "disjoint_replication": results["ood"],
        "oracle_primary_test": oracle_results["test"],
        "oracle_disjoint_replication": oracle_results["ood"],
        "quality_advantage_supported": support["test"] and support["ood"],
        "cost_advantage_supported": False,
        "boundary": (
            "The post-hoc DEV router is tested once on two disjoint query "
            "splits under four paired training seeds. Query intervals condition "
            "on those observed seeds; no universal seed or cost claim is made."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("root", "data", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.data, args.protocol)
    write_json(args.output, report)
    print(json.dumps({
        "independent_regrading_completed": True,
        "quality_advantage_supported": report["quality_advantage_supported"],
        "cost_advantage_supported": False,
    }))


if __name__ == "__main__":
    main()
