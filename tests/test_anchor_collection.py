import copy

import pytest

from scripts.analyze_anchor_development import advance_rule
from scripts.collect_anchor_development import allocation_cost
from scripts.audit_fresh_pools import available_math, question_identity, SUFFIX
from scripts.run_anchor_weight_development import variant_config
from servedistill.data import digest


def comparison():
    return {"business_macro": {"delta": 0.01, "positive_seed_pairs": 3},
            "all_business_domains_noninferior": True, "svamp_regression_noninferior": True}


def test_anchor_cannot_advance_by_winning_only_against_weak_control():
    controls = {name: comparison() for name in ("offline_sft", "oracle_sft", "initial")}
    controls["oracle_sft"]["business_macro"]["delta"] = -0.001
    assert not advance_rule(controls, True, {"minimum_positive_seed_pairs": 3})
    controls["oracle_sft"] = comparison()
    controls["initial"]["svamp_regression_noninferior"] = False
    assert not advance_rule(controls, True, {"minimum_positive_seed_pairs": 3})
    controls["initial"] = comparison()
    controls["offline_sft"]["business_macro"]["positive_seed_pairs"] = 2
    assert not advance_rule(controls, True, {"minimum_positive_seed_pairs": 3})


def test_cost_retains_restarted_trials_and_counts_preparation():
    pod = {"sid": "p1", "trial_sid": "t1", "job_run_sid": "r", "exit_status": 1,
           "gpuv": "NVIDIA-H20", "schedule_at": "2026-10-27T10:00:00Z",
           "start_at": "2026-10-27T10:01:00Z", "stop_at": "2026-10-27T11:00:00Z"}
    second = {**pod, "sid": "p2", "trial_sid": "t2", "exit_status": 0}
    trials = [{"sid": t, "job_run_sid": "r"} for t in ("t1", "t2")]
    assert allocation_cost([pod, second], trials, {"r"})["allocated_gpu_hours"] == 2
    with pytest.raises(ValueError, match="Every run"):
        allocation_cost([pod], trials, {"r"})
    active = copy.copy(second)
    active["stop_at"] = None
    with pytest.raises(ValueError, match="closed H20"):
        allocation_cost([pod, active], trials, {"r"})


def test_math_exclusion_survives_prompt_and_identifier_variants():
    old_question = "How many apples are left?"
    observed = {"math_questions": {question_identity(old_question)}, "prompts": {
        digest("How many pears are left?" + SUFFIX)}, "ids": set()}
    rows = [{"question": " HOW many apples are left? "},
            {"question": "How many pears are left?"},
            {"question": "How many oranges are left?"}]
    assert [r["question"] for r in available_math(rows, observed)] == [
        "How many oranges are left?"]


def test_weight_sweep_changes_only_coefficient_and_paired_seed():
    import dataclasses
    import json
    from pathlib import Path
    base = json.loads((Path(__file__).resolve().parents[1]
                       / "configs/anchor-development-v1.json").read_text())
    extension = {"scope": "development_only", "holdout_files_packaged": False,
                 "seeds": base["seeds"], "variants": {"low": {"anchor_ce_weight": 0.1}}}
    actual = dataclasses.asdict(variant_config(base, extension, "low", 101))
    expected = {**base["methods"]["aopd_anchor"], "anchor_ce_weight": 0.1, "seed": 101,
                "anchor_ce_domain_weights": {}}
    assert actual == expected
    extension["variants"]["low"]["anchor_ce_weight"] = 0.2
    with pytest.raises(ValueError, match="Undeclared"):
        variant_config(base, extension, "low", 101)
