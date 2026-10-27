"""Paired nonlinear business metrics with explicit seed and quality boundaries."""
from __future__ import annotations

import itertools

import numpy as np

from .expanded_statistics import validate_pairs
from .statistics import bootstrap_metric, metric


def ci(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def class_recall(rows, label):
    selected = [row for row in rows if row["target_class"] == label]
    if not selected:
        raise ValueError("Required quality-gate class has no independent test support")
    return sum(row["predicted_class"] == label for row in selected) / len(selected)


def paired_quality(baselines, candidates, gate, repetitions=5000, seed=202610254):
    if len(baselines) != 4:
        raise ValueError("Confirmation requires exactly four paired training seeds")
    if repetitions < 1:
        raise ValueError("Positive bootstrap repetitions required")
    before, after = validate_pairs(baselines, candidates)
    rng = np.random.default_rng(seed)
    seeds = len(before)
    seed_draw = rng.integers(0, seeds, (repetitions, seeds))
    business = ("search", "relevance", "substitute")
    query_macro, joint_macro = np.zeros(repetitions), np.zeros(repetitions)
    business_seed_delta = np.zeros(seeds)
    reports, gate_checks = {}, [[] for _ in range(seeds)]
    for domain in (*business, "math"):
        ids = [row["sample_id"] for row in baselines[0] if row["domain"] == domain]
        if not ids:
            raise ValueError("Missing frozen confirmation domain: " + domain)
        queries = [before[0][identity].get("query_group_sha256", identity) for identity in ids]
        if len(queries) != len(set(queries)):
            raise ValueError("Repeated queries require cluster resampling")
        a = [[group[identity] for identity in ids] for group in before]
        b = [[group[identity] for identity in ids] for group in after]
        indices = rng.integers(0, len(ids), (repetitions, len(ids)))
        differences = np.array([
            bootstrap_metric(y, indices) - bootstrap_metric(x, indices)
            for x, y in zip(a, b)])
        conditional = differences.mean(axis=0)
        joint = differences[seed_draw, np.arange(repetitions)[:, None]].mean(axis=1)
        a_scores, b_scores = np.array([metric(rows) for rows in a]), np.array([metric(rows) for rows in b])
        deltas = b_scores - a_scores
        reports[domain] = {
            "n_independent_queries": len(ids), "n_paired_training_seeds": seeds,
            "baseline_seed_scores": a_scores.tolist(), "candidate_seed_scores": b_scores.tolist(),
            "baseline_mean": float(a_scores.mean()), "candidate_mean": float(b_scores.mean()),
            "paired_seed_deltas": deltas.tolist(), "delta": float(deltas.mean()),
            "paired_query_ci95": ci(conditional),
            "exploratory_seed_query_ci95": ci(joint),
        }
        if domain in business:
            query_macro += conditional / len(business)
            joint_macro += joint / len(business)
            business_seed_delta += deltas / len(business)
            reports[domain]["noninferiority_margin"] = gate["domain_noninferiority_margin"]
            reports[domain]["noninferior"] = ci(conditional)[0] > gate["domain_noninferiority_margin"]
            for index, rows in enumerate(b):
                gate_checks[index].append({
                    "criterion": domain + "/metric", "value": b_scores[index],
                    "minimum": gate["minimum_metric"][domain],
                    "passed": bool(b_scores[index] >= gate["minimum_metric"][domain]),
                })
                valid = np.mean([row["json_valid"] for row in rows]).item()
                gate_checks[index].append({
                    "criterion": domain + "/json_valid", "value": valid,
                    "minimum": gate["minimum_json_valid"],
                    "passed": valid >= gate["minimum_json_valid"],
                })
                for label, minimum in gate.get("minimum_recall", {}).get(domain, {}).items():
                    recall = class_recall(rows, label)
                    gate_checks[index].append({
                        "criterion": domain + "/recall/" + label, "value": recall,
                        "minimum": minimum, "passed": recall >= minimum,
                    })
    query_interval = ci(query_macro)
    quality_passed = all(check["passed"] for checks in gate_checks for check in checks)
    noninferior = all(reports[domain]["noninferior"] for domain in business)
    math_retained = reports["math"]["paired_query_ci95"][0] > gate["math_noninferiority_margin"]
    positive_seeds = int((business_seed_delta > 0).sum())
    seed_indices = np.array(list(itertools.product(range(seeds), repeat=seeds)))
    return {
        "domains": reports, "quality_gate_checks_by_seed": gate_checks,
        "all_seed_quality_gates_passed": quality_passed,
        "business_macro": {
            "delta": float(business_seed_delta.mean()),
            "paired_seed_deltas": business_seed_delta.tolist(),
            "paired_query_ci95": query_interval,
            "exploratory_seed_query_ci95": ci(joint_macro),
            "exploratory_seed_only_ci95": ci(business_seed_delta[seed_indices].mean(axis=1)),
            "positive_seed_pairs": positive_seeds,
        },
        "all_business_domains_noninferior": noninferior,
        "svamp_regression_noninferior": math_retained,
        "quality_advantage_supported": bool(
            quality_passed and noninferior and math_retained and query_interval[0] > 0
            and positive_seeds >= gate["minimum_positive_seed_pairs"]),
        "bootstrap_repetitions": repetitions,
        "ci_scope": "Primary query intervals condition on the observed paired training seeds. "
                    "Four seeds do not establish universal training-seed robustness.",
        "multiplicity": "One frozen business-macro primary comparison; domain intervals "
                        "are pointwise, and every domain must pass the noninferiority gate.",
        "cost_boundary": "Quality advantage alone does not establish cost or throughput advantage.",
    }


def paired_sft_quality(offline, oracle, candidates, gate, repetitions=5000, seed=202610254):
    """Require superiority against both frozen strong SFT controls."""
    primary = paired_quality(offline, candidates, gate, repetitions, seed)
    oracle_guard = paired_quality(oracle, candidates, gate, repetitions, seed + 2)
    return {
        "primary": primary, "oracle_guard": oracle_guard,
        "quality_advantage_supported": bool(primary["quality_advantage_supported"]
                                             and oracle_guard["quality_advantage_supported"]),
        "comparison_rule": "Both predeclared SFT comparisons must pass; a win against "
                           "a weaker SFT alone cannot establish the quality advantage.",
    }
