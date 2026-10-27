"""Prospective attempt accounting and paired, fixed-seed query inference."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path

import numpy as np

from .advantage_contracts import file_digest
from .advantage_statistics import class_recall
from .data import canonical, digest
from .expanded_statistics import validate_pairs
from .statistics import bootstrap_metric, metric


def attempt_alpha(total: float, attempt: int) -> float:
    if not math.isfinite(total) or not 0 < total < 0.5:
        raise ValueError("A finite one-sided family alpha below 0.5 is required")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("Attempt index must be a positive integer")
    return total / (attempt * (attempt + 1))


@contextmanager
def locked_ledger(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_attempts(root: Path, family_hash: str) -> list[dict]:
    records, previous = [], None
    for index, path in enumerate(sorted(root.glob("attempt-*.json")), start=1):
        record = json.loads(path.read_text())
        if (path.name != f"attempt-{index:04d}.json" or record["attempt"] != index
                or record["family_sha256"] != family_hash
                or record["previous_receipt_sha256"] != previous):
            raise ValueError("Attempt ledger is missing, reordered or altered")
        records.append(record)
        previous = file_digest(path)
    return records


def reserve_attempt(root: Path, family_path: Path, bindings: dict) -> dict:
    """Call while holding locked_ledger; reservations are never refunded."""
    family = json.loads(family_path.read_text())
    family_hash = file_digest(family_path)
    records = read_attempts(root, family_hash)
    attempt = len(records) + 1
    alpha = attempt_alpha(family["total_one_sided_alpha"], attempt)
    record = {
        "attempt": attempt, "family_sha256": family_hash, "alpha": alpha,
        "previous_receipt_sha256": (
            file_digest(root / f"attempt-{attempt - 1:04d}.json") if records else None),
        "bindings": bindings, "consumed_even_if_failed": True,
    }
    with (root / f"attempt-{attempt:04d}.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    return record


def lower_bound(draws: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 0.5 or not np.isfinite(draws).all():
        raise ValueError("Invalid bootstrap tail or non-finite metric")
    # An observed order statistic avoids interpolation between sparse tail draws.
    index = max(0, math.floor(len(draws) * alpha) - 1)
    return float(np.partition(draws, index)[index])


def compare(baselines: list[list[dict]], candidates: list[list[dict]], gate: dict,
            alpha: float, repetitions: int, seed: int, chunk_size: int = 256,
            minimum_tail_draws: int = 200) -> dict:
    """Resample independent queries jointly across all four paired seed outputs."""
    if (len(baselines) != 4 or type(repetitions) is not int or repetitions < 1
            or type(chunk_size) is not int or chunk_size < 1
            or type(minimum_tail_draws) is not int or minimum_tail_draws < 1
            or not 0 < alpha < 0.5 or repetitions * alpha < minimum_tail_draws):
        raise ValueError("Four seeds and sufficient predeclared bootstrap tail draws required")
    before, after = validate_pairs(baselines, candidates)
    domains = ("search", "relevance", "substitute", "math")
    if {r["domain"] for r in baselines[0]} != set(domains):
        raise ValueError("Exactly the four frozen confirmation domains are required")
    identities = [r.get("query_group_sha256", r["sample_id"]) for r in baselines[0]]
    if len(identities) != len(set(identities)):
        raise ValueError("Query identities repeat within or across domains")
    rng = np.random.default_rng(seed)
    reports, checks = {}, [[] for _ in range(4)]
    macro_draws, macro_seed_deltas = np.zeros(repetitions), np.zeros(4)
    for domain in domains:
        ids = [r["sample_id"] for r in baselines[0] if r["domain"] == domain]
        a = [[group[i] for i in ids] for group in before]
        b = [[group[i] for i in ids] for group in after]
        scores_a, scores_b = np.array([metric(r) for r in a]), np.array([metric(r) for r in b])
        deltas = scores_b - scores_a
        draws = np.empty(repetitions)
        for start in range(0, repetitions, chunk_size):
            stop = min(start + chunk_size, repetitions)
            indices = rng.integers(0, len(ids), (stop - start, len(ids)))
            draws[start:stop] = np.mean([
                bootstrap_metric(y, indices) - bootstrap_metric(x, indices)
                for x, y in zip(a, b)], axis=0)
        margin = gate["math_noninferiority_margin"] if domain == "math" else gate[
            "domain_noninferiority_margin"]
        bound = lower_bound(draws, alpha)
        reports[domain] = {
            "n_independent_inputs": len(ids), "baseline_seed_scores": scores_a.tolist(),
            "candidate_seed_scores": scores_b.tolist(), "paired_seed_deltas": deltas.tolist(),
            "delta": float(deltas.mean()), "one_sided_lower_bound": bound,
            "noninferiority_margin": margin, "noninferior": bound > margin,
        }
        if domain == "math":
            continue
        macro_draws += draws / 3
        macro_seed_deltas += deltas / 3
        for index, rows in enumerate(b):
            valid = float(np.mean([r["json_valid"] for r in rows]))
            checks[index].extend([
                {"criterion": domain + "/metric", "value": float(scores_b[index]),
                 "passed": bool(scores_b[index] >= gate["minimum_metric"][domain])},
                {"criterion": domain + "/json_valid", "value": valid,
                 "passed": valid >= gate["minimum_json_valid"]},
            ])
            for label, minimum in gate.get("minimum_recall", {}).get(domain, {}).items():
                recall = class_recall(rows, label)
                checks[index].append({"criterion": domain + "/recall/" + label,
                                      "value": recall, "passed": recall >= minimum})
    bound = lower_bound(macro_draws, alpha)
    positive = int((macro_seed_deltas > 0).sum())
    passed = (all(c["passed"] for group in checks for c in group)
              and all(r["noninferior"] for r in reports.values())
              and bound > 0 and positive >= gate["minimum_positive_seed_pairs"])
    return {
        "domains": reports, "absolute_quality_checks_by_seed": checks,
        "business_macro": {"delta": float(macro_seed_deltas.mean()),
                           "paired_seed_deltas": macro_seed_deltas.tolist(),
                           "positive_seed_pairs": positive, "one_sided_lower_bound": bound},
        "quality_advantage_supported": bool(passed), "alpha": alpha,
        "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
        "ci_scope": "Percentile query bootstrap conditional on the four observed seed pairs; "
                    "asymptotic coverage only. Training-seed universality is not established.",
        "draws_digest": digest(canonical({
            "business_macro": macro_draws.tolist(), "domain_order": domains})),
    }
