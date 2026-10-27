"""Verify a sealed fresh packet before GPU execution or independent regrading."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .advantage_contracts import file_digest
from .data import digest, load_rows
from .sequential_confirmation import attempt_alpha


def verify_packet(packet: Path, family_path: Path, code_root: Path) -> tuple[dict, dict]:
    family = json.loads(family_path.read_text())
    protocol = json.loads((packet / "protocol.json").read_text())
    receipt = json.loads((packet / "reservation.json").read_text())
    expected_alpha = attempt_alpha(family["total_one_sided_alpha"], protocol["attempt"])
    if (protocol["scope"] != "frozen_fresh_confirmation"
            or protocol["family_sha256"] != file_digest(family_path)
            or protocol["model_predictions_observed_before_freeze"]
            or protocol["alpha"] != expected_alpha
            or protocol["paired_training_seeds"] != family["paired_training_seeds"]
            or protocol["required_batches"] != ["confirmation", "replication"]
            or protocol["required_controls"] != ["offline_sft", "oracle_sft"]
            or protocol["required_batches"] != family["required_batches"]
            or protocol["required_controls"] != family["required_controls"]
            or protocol["bootstrap_repetitions"] * expected_alpha <
            family["bootstrap_minimum_expected_tail_draws"]
            or protocol["bootstrap_chunk_size"] != family["bootstrap_chunk_size"]
            or protocol["reservation_sha256"] != file_digest(packet / "reservation.json")
            or receipt["family_sha256"] != protocol["family_sha256"]
            or receipt["attempt"] != protocol["attempt"] or receipt["alpha"] != expected_alpha
            or not receipt["consumed_even_if_failed"]
            or receipt["bindings"]["adapters"] != protocol["adapters"]
            or receipt["bindings"]["selected_dev_candidate"] != protocol["selected_dev_candidate"]):
        raise ValueError("Frozen fresh family, reservation or selection binding changed")
    for name, expected in protocol["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or file_digest(
                code_root / relative) != expected:
            raise ValueError("Frozen fresh execution/analysis source changed")
    if protocol["dev_analysis_sha256"] != file_digest(packet / "dev-analysis.json"):
        raise ValueError("DEV selection evidence changed")
    dev = json.loads((packet / "dev-analysis.json").read_text())
    if (dev["selected_dev_candidate"] != protocol["selected_dev_candidate"]
            or not dev["all_three_weights_and_controls_complete"]
            or not dev["advance_to_fresh_confirmation"]):
        raise ValueError("Complete eligible DEV selection is required")
    manifest_path = packet / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (file_digest(manifest_path) != protocol["data_manifest_sha256"]
            or manifest["splits"] != protocol["data_splits"]
            or manifest["source_lock_sha256"] != receipt["bindings"]["source_lock_sha256"]
            or manifest["business_count_per_domain_per_batch"] !=
            receipt["bindings"]["business_count_per_domain_per_batch"]
            or manifest["math_count_per_batch"] != receipt["bindings"]["math_count_per_batch"]
            or manifest["labels_used_for_selection"] or manifest["model_predictions_observed"]
            or manifest["selection_sha256"] != file_digest(packet / "data/selection.json")):
        raise ValueError("Frozen fresh data selection changed")
    sources, queries, inputs, ids = {}, set(), set(), set()
    for split in protocol["required_batches"]:
        path = packet / "data" / (split + ".jsonl")
        rows = load_rows(path)
        binding = manifest["splits"][split]
        if (file_digest(path) != binding["sha256"] or len(rows) != binding["rows"]
                or dict(Counter(r["domain"] for r in rows)) != binding["domains"]
                or set(binding["domains"]) != {"search", "relevance", "substitute", "math"}):
            raise ValueError("Frozen fresh batch contents or count changed")
        for row in rows:
            if (row["split"] != split or digest(row["prompt"]) != row["input_sha256"]
                    or row["input_sha256"] in inputs or row["sample_id"] in ids):
                raise ValueError("Repeated or altered fresh input identity")
            inputs.add(row["input_sha256"])
            ids.add(row["sample_id"])
            if "query_group_sha256" in row:
                if row["query_group_sha256"] in queries:
                    raise ValueError("Query overlaps across fresh domains or batches")
                queries.add(row["query_group_sha256"])
        sources[split] = {row["sample_id"]: row for row in rows}
    return protocol, sources
