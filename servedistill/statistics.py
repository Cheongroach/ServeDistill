"""Paired bootstrap over independent tasks, conditional on observed training seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from .data import load_rows


def metric(rows):
    task = rows[0].get("task", "")
    if task == "esci_ranking":
        values = [r["ndcg"] for r in rows if r["ndcg_eligible"]]
        return float(np.mean(values)) if values else 0.0
    if task in ("esci_classification", "esci_substitute"):
        labels = ("E", "S", "C", "I") if task == "esci_classification" else ("yes", "no")
        scores = []
        for label in labels:
            tp = sum(r["target_class"] == label and r["predicted_class"] == label for r in rows)
            fp = sum(r["target_class"] != label and r["predicted_class"] == label for r in rows)
            fn = sum(r["target_class"] == label and r["predicted_class"] != label for r in rows)
            total = 2 * tp + fp + fn
            scores.append(2 * tp / total if total else 0.0)
        return float(np.mean(scores))
    return float(np.mean([r["success"] for r in rows]))


def bootstrap_metric(rows, indices):
    """Vectorized task resampling, retaining the nonlinear macro-F1 metric."""
    task = rows[0].get("task", "")
    if task == "esci_ranking":
        values = np.array([r["ndcg"] for r in rows])
        eligible = np.array([r["ndcg_eligible"] for r in rows], dtype=float)
        numerator = (values * eligible)[indices].sum(axis=1)
        denominator = eligible[indices].sum(axis=1)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                         where=denominator > 0)
    if task in ("esci_classification", "esci_substitute"):
        labels = ("E", "S", "C", "I") if task == "esci_classification" else ("yes", "no")
        scores = []
        for label in labels:
            truth = np.array([r["target_class"] == label for r in rows])
            predicted = np.array([r["predicted_class"] == label for r in rows])
            tp = (truth & predicted)[indices].sum(axis=1)
            fp = (~truth & predicted)[indices].sum(axis=1)
            fn = (truth & ~predicted)[indices].sum(axis=1)
            denominator = 2 * tp + fp + fn
            scores.append(np.divide(2 * tp, denominator, out=np.zeros(len(indices)),
                                    where=denominator > 0))
        return np.mean(scores, axis=0)
    return np.array([r["success"] for r in rows], dtype=float)[indices].mean(axis=1)


def compare(baseline, candidates, repetitions=2000):
    groups = [{r["sample_id"]: r for r in rows} for rows in [baseline, *candidates]]
    if not candidates or repetitions < 1:
        raise ValueError("Candidates and positive bootstrap repetitions required")
    if any(len(group) != len(rows) for group, rows in zip(groups, [baseline, *candidates])):
        raise ValueError("Duplicate task IDs would inflate the sample size")
    if any(set(group) != set(groups[0]) for group in groups[1:]):
        raise ValueError("Paired comparison requires the same tasks")
    for group in groups[1:]:
        for identity, before in groups[0].items():
            for key in ("input_sha256", "domain", "task", "target_class", "ndcg_eligible"):
                if before.get(key) != group[identity].get(key):
                    raise ValueError("Task/label changed: " + identity + "/" + key)
    slices = {(r["domain"], None) for r in baseline}
    slices |= {(r["domain"], r.get("locale", "general")) for r in baseline}
    rng, reports = np.random.default_rng(20261024), {}
    for domain, locale in sorted(slices, key=str):
        ids = [r["sample_id"] for r in baseline if r["domain"] == domain
               and (locale is None or r.get("locale", "general") == locale)]
        rows = [[group[i] for i in ids] for group in groups]
        scores = [metric(part) for part in rows]
        indices = rng.integers(0, len(ids), (repetitions, len(ids)))
        sampled = [bootstrap_metric(part, indices) for part in rows]
        deltas = np.mean(sampled[1:], axis=0) - sampled[0]
        label = domain + ("/" + locale if locale else "")
        reports[label] = {
            "n_independent_tasks": len(ids), "n_training_seeds": len(candidates),
            "baseline": scores[0], "candidate_seed_scores": scores[1:],
            "candidate_mean": float(np.mean(scores[1:])),
            "delta": float(np.mean(scores[1:])) - scores[0],
            "paired_task_ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
            "small_slice": len(ids) < 20,
        }
    return {"slices": reports, "bootstrap_repetitions": repetitions,
            "ci_scope": "paired tasks, conditional on observed training seeds",
            "multiplicity": "exploratory; no multiple-comparison correction"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    result = compare(load_rows(args.baseline), [load_rows(p) for p in args.candidate], args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
