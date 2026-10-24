"""Freeze a deployment candidate using completed development evaluations only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DOMAINS = ("search", "relevance", "substitute", "math", "instruction")


def score(report):
    return 0.8 * report["business_macro"] + 0.2 * report["general_macro"]


def freeze(snapshot, output):
    """Select a method across two seeds, then its strongest development seed."""
    if output.exists():
        raise FileExistsError("Deployment selection is immutable")
    groups = {}
    for label, run in snapshot.items():
        if label.startswith(("business_sft_", "restore_general_")):
            continue
        if not run.get("completed") or run.get("steps") != 80:
            raise ValueError("Incomplete primary matrix: " + label)
        if run["selection"]["step"] < 1 or run.get("uses_test"):
            raise ValueError("Invalid development provenance: " + label)
        if abs(run["selection"]["dev_score"] - score(run["development"])) > 1e-10:
            raise ValueError("Checkpoint score does not match development evidence: " + label)
        method, seed = label.rsplit("_seed", 1)
        groups.setdefault(method, {})[int(seed)] = (label, run)
    if len(groups) != 12 or any(set(seeds) != {17, 29} for seeds in groups.values()):
        raise ValueError("Expected the frozen 12-method, two-seed primary matrix")
    if len({run["dev_task_sha256"] for seeds in groups.values()
            for _, run in seeds.values()}) != 1:
        raise ValueError("Development tasks differ across methods")
    candidates = []
    for method, seeds in sorted(groups.items()):
        records = list(seeds.values())
        if len({r["dev_task_sha256"] for _, r in records}) != 1:
            raise ValueError("Development tasks differ across seeds")
        drops = {
            domain: sum(r["development"]["domains"][domain]["selection_score"]
                        - r["baseline"]["domains"][domain]["selection_score"]
                        for _, r in records) / 2
            for domain in DOMAINS
        }
        # Thresholds must be recorded before any final test comparison.
        eligible = all(delta >= -0.05 - 1e-12 for delta in drops.values())
        candidates.append({
            "method": method, "dev_score": sum(score(r["development"]) for _, r in records) / 2,
            "domain_delta": drops, "passes_dev_retention_gate": eligible,
            "seed_scores": {str(seed): score(run["development"])
                            for seed, (_, run) in seeds.items()},
        })
    candidates.sort(key=lambda r: (-r["dev_score"], r["method"]))
    eligible = [r for r in candidates if r["passes_dev_retention_gate"]]
    chosen = (eligible or candidates)[0]
    label, run = sorted(groups[chosen["method"]].values(),
                        key=lambda item: (-score(item[1]["development"]), item[0]))[0]
    opd = [r for r in candidates if r["method"] not in ("oracle_sft", "offline_sft", "grpo")]
    opd_eligible = [r for r in opd if r["passes_dev_retention_gate"]]
    opd_method = (opd_eligible or opd)[0]
    opd_label, opd_run = sorted(groups[opd_method["method"]].values(),
                               key=lambda item: (-score(item[1]["development"]), item[0]))[0]
    payload = {
        "label": label, "method": chosen["method"], "selection": run["selection"],
        "adapter_file_sha256": run["adapter_file_sha256"],
        "deployment_gate_passed": bool(eligible),
        "criterion": "mean two-seed 0.8 business macro + 0.2 general macro; <=5pp loss per domain",
        "seed_rule": "highest development score within selected method; label breaks ties",
        "checkpoint_rule": "each run's previously frozen development-selected checkpoint",
        "test_used_for_selection": False,
        "dev_task_sha256": run["dev_task_sha256"],
        "snapshot_sha256": hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        "methods": candidates,
        "opd_candidate": {
            "label": opd_label, "method": opd_method["method"],
            "selection": opd_run["selection"],
            "adapter_file_sha256": opd_run["adapter_file_sha256"],
            "deployment_gate_passed": bool(opd_eligible),
        },
        "interpretation": "research candidate only if no method passes the development gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(json.loads(args.snapshot.read_text()), args.output)))


if __name__ == "__main__":
    main()
