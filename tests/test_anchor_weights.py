"""Exercise terminal evidence rejection and the preregistered joint selection rule."""
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_anchor_weights import select_candidate
from scripts.collect_anchor_weights import (
    load_protocols, verify_collection, verify_weight_seed,
)
from scripts.run_anchor_weight_development import variant_config
from servedistill.advantage_contracts import file_digest
from servedistill.runtime import write_json
from servedistill.scheduling import schedule, update_learning_rate


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"anchor01": 0.1, "aopd_anchor": 0.5, "anchor10": 1.0}


def test_joint_selection_rejects_incomplete_candidates_and_ignores_ineligible_winner():
    result = {name: {"advance_to_fresh_confirmation": True, "mean_dev_business_macro": .8}
              for name in WEIGHTS}
    with pytest.raises(ValueError, match="All three"):
        select_candidate({"anchor01": result["anchor01"]}, WEIGHTS)
    result["anchor10"] = {"advance_to_fresh_confirmation": False, "mean_dev_business_macro": .99}
    assert select_candidate(result, WEIGHTS) == (["anchor01", "aopd_anchor"], "anchor01")
    result["aopd_anchor"]["mean_dev_business_macro"] += 1e-10
    assert select_candidate(result, WEIGHTS)[1] == "aopd_anchor"
    for candidate in result.values():
        candidate["advance_to_fresh_confirmation"] = False
    assert select_candidate(result, WEIGHTS) == ([], None)
    result["anchor10"]["mean_dev_business_macro"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        select_candidate(result, WEIGHTS)


@pytest.fixture
def weight_evidence(tmp_path):
    base, extension = load_protocols(ROOT / "configs/anchor-development-v1.json",
                                     ROOT / "configs/anchor-weight-development-v2.json")
    extension_hash = file_digest(ROOT / "configs/anchor-weight-development-v2.json")
    config = variant_config(base, extension, "anchor01", 101)
    training = [{"sample_id": f"{domain}/{i}", "domain": domain}
                for domain in ("search", "relevance", "substitute", "math", "instruction")
                for i in range(3)]
    batches = [[r["sample_id"] for r in batch] for batch in schedule(training, config)]
    schedule_hash = hashlib.sha256(json.dumps(batches).encode()).hexdigest()
    study = tmp_path / "study"
    checkpoint = study / "train/checkpoint-0320"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"synthetic adapter fixture")
    write_json(checkpoint / "adapter_config.json", {"synthetic": True})
    evaluated = {
        "precision": "bf16", "step": 320, "source": "fixed_final_update", "dev_rows": 1100,
        "adapter_sha256": file_digest(checkpoint / "adapter_model.safetensors"),
        "adapter_config_sha256": file_digest(checkpoint / "adapter_config.json")}
    tasks = [{"returncode": 0, "name": "weight_training_and_dev"}]
    metrics, rollouts, tokens = [], [], 0
    for step, batch in enumerate(batches, 1):
        ids = [identity for identity in batch for _ in range(config.group_size)]
        tokens += len(ids)
        metrics.append({
            "step": step, "learner_version": step, "rollout_version": step - 1,
            "response_tokens": len(ids), "cumulative_response_tokens": tokens,
            "learning_rate_used": update_learning_rate(config, step),
            "parameter_delta_l2": .1, "gradient_norm": 1., "loss": .2,
            "behavior_max_error": .00001})
        rollouts.append({"step": step, "version": step - 1, "sample_ids": ids,
                         "texts": ["{}"] * len(ids), "rewards": [0.] * len(ids)})
    for name, rows in (("metrics", metrics), ("rollouts", rollouts)):
        (study / f"train/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    documents = {
        "worker-completed.json": {"completed": True, "assets_verified": True, "tasks": tasks},
        "task-events.json": tasks,
        "study/completed.json": {
            "completed": True, "scope": "development_only", "holdout_observed": False,
            "quality_advantage_supported": False, "seed": 101, "variant": "anchor01",
            "extension_sha256": extension_hash, "evaluated_adapter": evaluated},
        "study/binding.json": {
            "scope": "development_only", "holdout_observed": False, "seed": 101,
            "variant": "anchor01", "extension_sha256": extension_hash,
            "base_protocol_sha256": extension["base_protocol_sha256"], "fresh_adam": True,
            "common_initial_adapter": base["initial_adapter_sha256"],
            "prompt_schedule_sha256": schedule_hash},
        "study/execution.json": {"started": 10., "ended": 20., "returncode": 0},
        "study/evaluated-adapter.json": evaluated,
        "study/config.json": dataclasses.asdict(config),
        "study/train/config.json": dataclasses.asdict(config),
        "study/train/schedule.json": batches,
        "study/train/checkpoint-0320/state.json": {"step": 320, "tokens_seen": tokens},
        "study/train/completed.json": {
            "completed": True, "steps": 320, "planned_steps": 320,
            "stop_reason": "update_budget", "final_checkpoint": "checkpoint-0320",
            "teacher_unchanged": True, "nonzero_updates": 320, "response_tokens": tokens},
    }
    for name, value in documents.items():
        write_json(tmp_path / name, value)
    return tmp_path, base, extension, extension_hash, training


def verify_fixture(fixture):
    root, base, extension, extension_hash, training = fixture
    return verify_weight_seed(root, base, extension, extension_hash, "anchor01", 101, training)


def test_complete_weight_seed_is_bound_to_all_executed_updates(weight_evidence):
    result = verify_fixture(weight_evidence)
    assert result["verified_optimizer_updates"] == 320
    assert result["verified_response_tokens"] == 6400


@pytest.mark.parametrize("change", [
    "adapter", "schedule", "early_final", "observed_holdout", "wrong_seed",
    "missing_update", "executed_prompt", "nonfinite_update", "token_state", "failed_task",
])
def test_weight_evidence_rejects_broken_bindings(weight_evidence, change):
    root = weight_evidence[0]
    if change == "adapter":
        (root / "study/train/checkpoint-0320/adapter_model.safetensors").write_bytes(b"different")
    elif change in ("missing_update", "executed_prompt", "nonfinite_update"):
        path = root / ("study/train/rollouts.jsonl" if change == "executed_prompt"
                       else "study/train/metrics.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "missing_update":
            rows.pop(10)
        elif change == "executed_prompt":
            rows[10]["sample_ids"][0] = "changed-prompt"
        else:
            rows[10]["parameter_delta_l2"] = float("nan")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        relative, key, value = {
            "early_final": ("study/train/completed.json", "steps", 160),
            "observed_holdout": ("study/completed.json", "holdout_observed", True),
            "wrong_seed": ("study/binding.json", "seed", 211),
            "token_state": ("study/train/checkpoint-0320/state.json", "tokens_seen", 1),
            "failed_task": ("worker-completed.json", "completed", False),
            "schedule": ("study/train/schedule.json", None, None),
        }[change]
        path = root / relative
        doc = json.loads(path.read_text())
        if change == "schedule":
            doc[0][0] = "changed-prompt"
        else:
            doc[key] = value
        write_json(path, doc)
    with pytest.raises(ValueError):
        verify_fixture(weight_evidence)


def test_joint_analysis_rejects_omitted_prediction_files(tmp_path):
    base, extension = load_protocols(ROOT / "configs/anchor-development-v1.json",
                                     ROOT / "configs/anchor-weight-development-v2.json")
    extension_hash = file_digest(ROOT / "configs/anchor-weight-development-v2.json")
    write_json(tmp_path / "collection-audit.json", {
        "verified": True, "all_runs_terminal_successful": True, "scope": "development_only",
        "holdout_observed": False, "extension_sha256": extension_hash,
        "base_protocol_sha256": extension["base_protocol_sha256"], "files": []})
    with pytest.raises(ValueError, match="Complete eight-run"):
        verify_collection(tmp_path, base, extension, extension_hash)


@pytest.fixture
def domain_evidence(weight_evidence):
    from scripts.run_domain_anchor_development import load_protocol, variant_config as domain_config
    root, _, _, _, training = weight_evidence
    base, protocol = load_protocol(ROOT / "configs/anchor-development-v1.json",
                                   ROOT / "configs/domain-anchor-development-v1.json")
    protocol_hash = file_digest(ROOT / "configs/domain-anchor-development-v1.json")
    for relative in ("study/completed.json", "study/binding.json"):
        path = root / relative
        value = json.loads(path.read_text())
        value.pop("extension_sha256")
        value["protocol_sha256"] = protocol_hash
        value["variant"] = "domain01"
        write_json(path, value)
    config = dataclasses.asdict(domain_config(base, protocol, "domain01", 101))
    for relative in ("study/config.json", "study/train/config.json"):
        write_json(root / relative, config)
    return root, base, protocol, protocol_hash, training


def test_domain_candidate_binds_every_executed_update(domain_evidence):
    from scripts.collect_domain_anchor_development import verify_seed_candidate
    root, base, protocol, protocol_hash, training = domain_evidence
    result = verify_seed_candidate(root, base, protocol, protocol_hash, "domain01", 101, training)
    assert result["verified_optimizer_updates"] == 320


@pytest.mark.parametrize("change", ["weight", "schedule", "holdout", "adapter", "task"])
def test_domain_candidate_rejects_broken_execution(domain_evidence, change):
    from scripts.collect_domain_anchor_development import verify_seed_candidate
    root, base, protocol, protocol_hash, training = domain_evidence
    path, key = {
        "weight": ("study/train/config.json", "anchor_ce_domain_weights"),
        "schedule": ("study/train/schedule.json", None),
        "holdout": ("study/binding.json", "holdout_observed"),
        "adapter": ("study/evaluated-adapter.json", "adapter_sha256"),
        "task": ("worker-completed.json", "completed"),
    }[change]
    file = root / path
    doc = json.loads(file.read_text())
    if change == "weight":
        doc[key]["math"] = .1
    elif change == "schedule":
        doc[0][0] = "wrong"
    else:
        doc[key] = {"holdout": True, "adapter": "wrong", "task": False}[change]
    write_json(file, doc)
    with pytest.raises(ValueError):
        verify_seed_candidate(root, base, protocol, protocol_hash, "domain01", 101, training)


def test_domain_selection_cannot_skip_or_accept_ineligible_candidate():
    from scripts.analyze_domain_anchor_development import select_candidate as select_domain
    _, protocol = __import__("scripts.run_domain_anchor_development", fromlist=["load_protocol"]).load_protocol(
        ROOT / "configs/anchor-development-v1.json", ROOT / "configs/domain-anchor-development-v1.json")
    result = {n: {"advance_to_fresh_confirmation": True, "mean_dev_business_macro": .8}
              for n in protocol["variants"]}
    assert select_domain(result, protocol)[1] == "domain01"
    result["domain01"]["advance_to_fresh_confirmation"] = False
    assert select_domain(result, protocol)[1] == "domain05"
    result["domain05"]["advance_to_fresh_confirmation"] = False
    assert select_domain(result, protocol) == ([], None)
    with pytest.raises(ValueError, match="Both domain"):
        select_domain({"domain01": result["domain01"]}, protocol)
