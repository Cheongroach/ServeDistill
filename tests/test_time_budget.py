"""CPU evidence for elapsed-time accounting and the actual training stop boundary."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from servedistill.time_budget import TrainingBudget, completed_checkpoint


def test_restore_keeps_consumed_time_and_shared_prior_cost():
    now = [100.0]
    clock = lambda: now[0]
    original = TrainingBudget(10, 3, clock=clock)
    now[0] += 4
    saved = original.snapshot()
    resumed = TrainingBudget(10, 3, clock=clock, restored=saved)
    now[0] += 3
    assert resumed.exhausted()
    assert resumed.snapshot()["charged_method_seconds"] == 10
    with pytest.raises(ValueError, match="differs"):
        TrainingBudget(12, 3, clock=clock, restored=saved)


@pytest.mark.parametrize("limit,prior", [(0, 0), (float("inf"), 0), (10, -1),
                                        (10, 10), (None, 1)])
def test_invalid_budget_is_rejected(limit, prior):
    with pytest.raises(ValueError):
        TrainingBudget(limit, prior)


def test_step_cap_is_not_equal_time_evidence():
    config = {"steps": 100, "method_time_budget_seconds": 10, "prior_method_seconds": 2}
    result = {"completed": True, "steps": 100, "stop_reason": "update_budget",
              "time_budget": {"limit_seconds": 10, "prior_method_seconds": 2,
                              "exhausted": True}}
    with pytest.raises(ValueError, match="did not exhaust"):
        completed_checkpoint(config, result)
    result.update(steps=4, stop_reason="method_time_budget")
    assert completed_checkpoint(config, result) == "checkpoint-0004"


def test_prior_cost_credit_is_bound_to_actual_interval_bytes(tmp_path):
    from servedistill.advantage_contracts import file_digest, verify_cost_credit

    evidence = tmp_path / "teacher-repair.json"
    evidence.write_text(json.dumps({"start_unix_seconds": 100, "end_unix_seconds": 200,
                                    "allocated_gpu_count": 1}))
    manifest = tmp_path / "credits.json"
    credits = {"method_time_budget_seconds": 200, "components": [{
        "kind": "teacher_repair", "amortization_runs": 4,
        "evidence_file": evidence.name, "evidence_sha256": file_digest(evidence)}]}
    manifest.write_text(json.dumps(credits))
    config = {"method_time_budget_seconds": 200, "prior_method_seconds": 25}
    assert verify_cost_credit(manifest, config)["prior_method_seconds"] == 25
    with pytest.raises(ValueError, match="does not match"):
        verify_cost_credit(manifest, {**config, "prior_method_seconds": 0})
    evidence.write_text(json.dumps({"start_unix_seconds": 100, "end_unix_seconds": 101,
                                    "allocated_gpu_count": 1}))
    with pytest.raises(ValueError, match="evidence changed"):
        verify_cost_credit(manifest, config)


def test_actual_training_stops_between_updates_and_saves_final_adapter(tmp_path, monkeypatch):
    from servedistill import train as driver

    now = [0.0]
    monkeypatch.setattr(driver.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(driver, "TrainingBudget",
                        lambda seconds, prior_seconds, **kw:
                        TrainingBudget(seconds, prior_seconds, clock=lambda: now[0], **kw))

    class Model(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.4]))

        def save_pretrained(self, path):
            torch.save(self.state_dict(), path / "fixture-adapter.pt")

    class Runtime:
        contract = {"boundary": "CPU test fixture, no model quality evidence"}
        teacher = None
        eos_ids = [2]
        tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda *a, **kw: [1])

        def __init__(self, *a, **kw):
            self.student = Model()

        def evaluate(self, rows, output, label, **kw):
            now[0] += 1
            return {"business_macro": 0.5, "general_macro": 0.5}

        def score(self, rows, tokens, mask, **kw):
            now[0] += 3
            logp = self.student.weight.log_softmax(-1).expand(*tokens.shape, 3)
            return {"logp": logp.gather(-1, tokens[..., None]).squeeze(-1),
                    "entropy": -(logp.exp() * logp).sum(-1)}

    monkeypatch.setattr(driver, "Runtime", Runtime)
    for name, replacement in (
            ("synchronize", lambda *a: None), ("max_memory_allocated", lambda: 0),
            ("get_rng_state_all", lambda: [])):
        monkeypatch.setattr(torch.cuda, name, replacement)
    data = tmp_path / "data"
    data.mkdir()
    rows = [{"sample_id": domain, "domain": domain, "target": {"answer": "1"}}
            for domain in ("search", "relevance", "substitute", "math", "instruction")]
    for split in ("train", "dev"):
        (data / (split + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {"objective": "oracle_sft", "steps": 100, "prompts_per_step": 1,
              "dev_every": 100, "method_time_budget_seconds": 5, "prior_method_seconds": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = driver.train(SimpleNamespace(
        config=config_path, student="fixture", teacher=None, adapter=None, resume=None,
        teacher_device="cpu", data=data, output=tmp_path / "out", stop_after=None))
    assert result["steps"] == 1
    assert result["nonzero_updates"] == 1
    assert result["time_budget"]["overrun_seconds"] == 1
    checkpoint = completed_checkpoint(config, result)
    assert (tmp_path / "out" / checkpoint / "fixture-adapter.pt").exists()
    saved = json.loads((tmp_path / "out" / checkpoint / "state.json").read_text())
    assert saved["time_budget"]["charged_method_seconds"] == 6
