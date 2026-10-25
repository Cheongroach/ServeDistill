"""Frozen-bank timing and paired training-seed quality statistics."""
from __future__ import annotations

from collections import defaultdict
import itertools

import numpy as np

from .statistics import bootstrap_metric, metric


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def performance(rows, banks_per_worker=16, worker_count=2, repetitions=3,
                bootstrap_repetitions=5000):
    """Use bank medians; resample whole banks, never timing repetitions."""
    groups = defaultdict(dict)
    for row in rows:
        if row["warmup"] or row["profiled"]:
            continue
        key = (row["worker_index"], row["bank"])
        cell = (row["microbatch"], row["repetition"])
        if cell in groups[key]:
            raise ValueError("Duplicate bank/microbatch/repetition")
        seconds = row["learner_update_seconds"]
        if not np.isfinite(seconds) or seconds <= 0 or not row["parent_reset"]:
            raise ValueError("Invalid timing or parent reset")
        groups[key][cell] = row
    expected = {(w, b) for w in range(worker_count) for b in range(banks_per_worker)}
    if set(groups) != expected:
        raise ValueError("Incomplete independent-bank evidence")
    cells = {(m, r) for m in (1, 2, 4, 8) for r in range(repetitions)}
    banks = []
    for key in sorted(groups):
        values = groups[key]
        if set(values) != cells:
            raise ValueError("Incomplete paired timing repetitions")
        tokens = {row["effective_response_tokens"] for row in values.values()}
        widths = {row["response_width"] for row in values.values()}
        if len(tokens) != 1 or len(widths) != 1 or min(tokens) <= 0:
            raise ValueError("Frozen trajectory size changed")
        banks.append({
            "worker_index": key[0], "bank": key[1],
            "effective_response_tokens": tokens.pop(), "response_width": widths.pop(),
            "mean_prompt_token_length": next(iter(values.values())).get("mean_prompt_token_length"),
            "median_seconds": {
                m: float(np.median([values[m, r]["learner_update_seconds"]
                                    for r in range(repetitions)]))
                for m in (1, 2, 4, 8)},
        })
    rng = np.random.default_rng(20261025)
    indices = rng.integers(0, len(banks), (bootstrap_repetitions, len(banks)))
    times = {m: np.array([bank["median_seconds"][m] for bank in banks])
             for m in (1, 2, 4, 8)}
    tokens = sum(bank["effective_response_tokens"] for bank in banks)
    comparisons = {}
    for m in (1, 2, 4, 8):
        ratios = times[1] / times[m]
        # Identical token totals cancel in the paired throughput ratio.
        sampled = times[1][indices].sum(axis=1) / times[m][indices].sum(axis=1)
        ci = interval(sampled)
        retained = [r for r in rows if not r["warmup"] and not r["profiled"]
                    and r["microbatch"] == m]
        comparisons[str(m)] = {
            "response_tokens_per_second": tokens / float(times[m].sum()),
            "paired_throughput_ratio": float(times[1].sum() / times[m].sum()),
            "paired_bank_ci95": ci,
            "stable_improvement": bool(ci[0] > 1),
            "median_bank_speedup": float(np.median(ratios)),
            "median_bank_update_seconds": float(np.median(times[m])),
            "p95_bank_update_seconds": float(np.percentile(times[m], 95)),
            "maximum_peak_allocated_gib": max(r["peak_allocated_bytes"] for r in retained) / 2**30,
            "maximum_peak_reserved_gib": max(r["peak_reserved_bytes"] for r in retained) / 2**30,
            "worker_throughput_ratios": {
                str(w): float(sum(b["median_seconds"][1] for b in banks if b["worker_index"] == w)
                              / sum(b["median_seconds"][m] for b in banks if b["worker_index"] == w))
                for w in range(worker_count)},
        }
    length_slices = {}
    if all(b["mean_prompt_token_length"] is not None for b in banks):
        ranked = np.argsort([b["mean_prompt_token_length"] for b in banks])
        for bucket, selected in enumerate(np.array_split(ranked, 4)):
            local_indices = rng.choice(selected, (bootstrap_repetitions, len(selected)))
            length_slices[str(bucket + 1)] = {
                "n_independent_banks": len(selected),
                "mean_prompt_token_length_range": [
                    min(banks[i]["mean_prompt_token_length"] for i in selected),
                    max(banks[i]["mean_prompt_token_length"] for i in selected)],
                "m4_throughput_ratio": float(times[1][selected].sum() / times[4][selected].sum()),
                "paired_bank_ci95": interval(
                    times[1][local_indices].sum(axis=1) / times[4][local_indices].sum(axis=1)),
            }
    return {
        "n_independent_banks": len(banks),
        "n_formal_measurements": len(banks) * len(cells),
        "n_unique_trajectory_response_tokens": tokens,
        "bootstrap_repetitions": bootstrap_repetitions,
        "ci_scope": "paired independent banks, conditional on fixed parent/workers/workload",
        "estimator": "total response tokens divided by sum of per-bank median update time",
        "phase_boundary": "unprofiled learner update; excludes rollout, state reset, CPU vector export",
        "comparisons": comparisons, "banks": banks, "prompt_length_quartiles": length_slices,
    }


def validate_pairs(baselines, candidates):
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("Equal nonempty paired seed sets required")
    groups = []
    for rows in [*baselines, *candidates]:
        group = {row["sample_id"]: row for row in rows}
        if not rows or len(group) != len(rows):
            raise ValueError("Empty evaluation or duplicate task IDs")
        groups.append(group)
    for group in groups[1:]:
        if set(group) != set(groups[0]):
            raise ValueError("Paired seeds require identical task IDs")
        for identity, row in groups[0].items():
            for key in ("input_sha256", "domain", "task", "target_class",
                        "ndcg_eligible", "query_group_sha256"):
                if row.get(key) != group[identity].get(key):
                    raise ValueError("Task/label changed: " + identity + "/" + key)
    return groups[:len(baselines)], groups[len(baselines):]


def quality(baselines, candidates, repetitions=3000, margin=-0.05):
    """Keep every query's outputs bound across all observed seed pairs."""
    if repetitions < 1:
        raise ValueError("Positive bootstrap repetitions required")
    before, after = validate_pairs(baselines, candidates)
    first = baselines[0]
    slices = {(r["domain"], None) for r in first}
    slices |= {(r["domain"], r.get("locale", "general")) for r in first}
    rng, reports = np.random.default_rng(20261025), {}
    for domain, locale in sorted(slices, key=str):
        ids = [r["sample_id"] for r in first if r["domain"] == domain
               and (locale is None or r.get("locale", "general") == locale)]
        query_ids = [before[0][i].get("query_group_sha256", i) for i in ids]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Repeated queries need cluster resampling")
        a = [[group[i] for i in ids] for group in before]
        b = [[group[i] for i in ids] for group in after]
        a_scores, b_scores = [metric(r) for r in a], [metric(r) for r in b]
        seed_deltas = np.array(b_scores) - np.array(a_scores)
        indices = rng.integers(0, len(ids), (repetitions, len(ids)))
        paired = np.mean([bootstrap_metric(y, indices) - bootstrap_metric(x, indices)
                          for x, y in zip(a, b)], axis=0)
        ci = interval(paired)
        auxiliary = {}
        for key in ("json_valid", "truncated", "mrr"):
            if not all(key in r for part in [*a, *b] for r in part):
                continue
            values_a = np.array([[r[key] for r in part] for part in a], dtype=float)
            values_b = np.array([[r[key] for r in part] for part in b], dtype=float)
            delta_per_query = (values_b - values_a).mean(axis=0)
            auxiliary[key] = {
                "m1": float(values_a.mean()), "m4": float(values_b.mean()),
                "delta": float(delta_per_query.mean()),
                "paired_query_ci95": interval(delta_per_query[indices].mean(axis=1)),
            }
        # Exact empirical seed resampling is exploratory with only four seeds.
        seed_indices = np.array(list(itertools.product(range(len(a)), repeat=len(a))))
        name = domain + ("/" + locale if locale else "")
        reports[name] = {
            "n_independent_queries": len(ids), "n_paired_training_seeds": len(a),
            "n_metric_eligible_queries": sum(
                row.get("ndcg_eligible", True) for row in a[0]),
            "m1_seed_scores": a_scores, "m4_seed_scores": b_scores,
            "m1_mean": float(np.mean(a_scores)), "m4_mean": float(np.mean(b_scores)),
            "paired_seed_deltas": seed_deltas.tolist(),
            "delta": float(seed_deltas.mean()), "paired_query_ci95": ci,
            "noninferiority_margin": margin,
            "noninferiority": "supported" if ci[0] > margin else
                             "inferior" if ci[1] < margin else "undetermined",
            "exploratory_paired_seed_ci95": interval(seed_deltas[seed_indices].mean(axis=1)),
            "auxiliary": auxiliary,
        }
    business = ("search", "relevance", "substitute")
    return {
        "slices": reports, "bootstrap_repetitions": repetitions,
        "ci_scope": "paired independent queries, conditional on the observed training seeds",
        "seed_ci_scope": "exploratory empirical paired seed bootstrap; four seeds do not establish universality",
        "multiplicity": "pointwise intervals; joint business claim requires every primary domain to pass",
        "all_business_domains_noninferior": all(
            d in reports and reports[d]["noninferiority"] == "supported" for d in business),
    }
