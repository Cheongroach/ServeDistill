"""CPU gates for the complete pre-test 640-update method selection."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "freeze640", Path(__file__).resolve().parents[1] / "scripts/freeze_advantage_budget640.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def development():
    specs = [
        {"kind": "final_checkpoint"},
        {"kind": "lora_factor_mean_fp32", "steps": [480, 640]},
        {"kind": "lora_factor_mean_fp32", "steps": [160, 320, 480, 640]}]
    rankings = {}
    for objective, business in (("opd", 0.73), ("offline_sft", 0.71), ("oracle_sft", 0.72)):
        rankings[objective] = [
            {"name": f"{objective}_{aggregation}_{rate}_{index}",
             "source_arm": f"{objective}_{aggregation}_{rate}",
             "objective": objective, "business_macro": business,
             "selection_score": 0.8 * business + 0.2 * 0.5,
             "arm_config": {"objective": objective, "steps": 640,
                            "aggregation": aggregation, "learning_rate": rate},
             "postprocessing": rule}
            for aggregation in ("token", "trajectory")
            for rate in (1e-5, 2e-5, 5e-5)
            for index, rule in enumerate(specs)]
    rankings["full_kl"] = [
        {"name": "repaired_full_kl/" + str(i), "source_arm": "deep_repaired_full_kl_lr2e5",
         "objective": "full_kl", "business_macro": 0.70, "selection_score": 0.65,
         "postprocessing": rule}
        for i, rule in enumerate(specs)]
    return {
        "declared_candidates": 60, "completed_candidates": 57, "failed_candidates": 3,
        "complete_selection_pool": True, "independent_regrading_completed": True,
        "holdout_observed": False, "rankings": rankings,
        "failed_controls": {"deep_original_full_kl_lr2e5": {"training_failed": True}}}


def test_complete_positive_pool_keeps_both_strong_sft_controls():
    selected = module.select(development())
    assert set(selected) == {"opd", "offline_sft", "oracle_sft", "full_kl"}
    assert selected["oracle_sft"]["business_macro"] > selected["offline_sft"]["business_macro"]


@pytest.mark.parametrize("change", [
    "missing_candidate", "missing_grid", "weaker_oracle", "tied_oracle",
    "unknown_failure", "holdout_observed", "objective_drift", "duplicate_candidate"])
def test_pre_test_selection_rejects_invalid_or_unconvincing_pool(change):
    value = copy.deepcopy(development())
    if change == "missing_candidate":
        value["completed_candidates"] -= 1
    elif change == "missing_grid":
        value["rankings"]["oracle_sft"][0]["arm_config"]["learning_rate"] = 1e-4
    elif change in ("weaker_oracle", "tied_oracle"):
        score = 0.74 if change == "weaker_oracle" else 0.73
        for row in value["rankings"]["oracle_sft"]:
            row["business_macro"] = score
            row["selection_score"] = 0.8 * score + 0.2 * 0.5
    elif change == "unknown_failure":
        value["failed_controls"]["oracle_sft"] = {}
    elif change == "holdout_observed":
        value["holdout_observed"] = True
    elif change == "objective_drift":
        value["rankings"]["opd"][0]["arm_config"]["objective"] = "oracle_sft"
    else:
        value["rankings"]["offline_sft"][0]["name"] = value["rankings"]["offline_sft"][1]["name"]
    with pytest.raises(ValueError):
        module.select(value)
