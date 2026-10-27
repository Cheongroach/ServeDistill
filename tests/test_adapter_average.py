"""CPU evidence for exact factor means and rejection of changed source adapters."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from servedistill.adapter_average import materialize_average, verify_average, source_average


@pytest.fixture
def averages(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "completed.json").write_text(json.dumps({"completed": True, "steps": 320}))
    for step, a, b in ((240, 1., 4.), (320, 3., 2.)):
        path = train / f"checkpoint-{step:04d}"
        path.mkdir()
        save_file({"lora_A.weight": torch.tensor([[a]]),
                   "lora_B.weight": torch.tensor([[b]])}, str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text(json.dumps({"r": 1}))
        (path / "state.json").write_text(json.dumps({"step": step, "adapter_sha256": str(step)}))
        (path / "training.pt").write_bytes(b"CPU source optimizer fingerprint fixture")
    return train, tmp_path / "average", {"kind": "lora_factor_mean_fp32", "steps": [240, 320]}


def test_factor_mean_differs_from_mean_of_weight_deltas(averages):
    train, adapter, spec = averages
    result = materialize_average(train, adapter, spec, 320)
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    assert (tensors["lora_B.weight"] @ tensors["lora_A.weight"]).item() == 6.
    assert (1. * 4. + 3. * 2.) / 2. == 5.
    assert result["postprocessing_only"] and not result["resumable_optimizer"]
    assert set(result["source_checkpoints"]) == {"checkpoint-0240", "checkpoint-0320"}
    assert verify_average(train, adapter, spec, 320) == result


@pytest.mark.parametrize("change", ["source_weights", "evaluated_weights", "optimizer", "rule"])
def test_changed_average_evidence_is_rejected(averages, change):
    train, adapter, spec = averages
    materialize_average(train, adapter, spec, 320)
    if change in ("source_weights", "evaluated_weights"):
        path = (train / "checkpoint-0240" if change == "source_weights" else adapter)
        weight = path / "adapter_model.safetensors"
        tensors = load_file(str(weight))
        tensors["lora_A.weight"].add_(1.)
        save_file(tensors, str(weight))
    elif change == "optimizer":
        (train / "checkpoint-0240/training.pt").write_bytes(b"changed optimizer")
    else:
        spec = {"kind": "lora_factor_mean_fp32", "steps": [160, 320]}
    with pytest.raises(ValueError):
        verify_average(train, adapter, spec, 320)


@pytest.mark.parametrize("change", ["incomplete", "config", "precision", "shape", "nonfinite"])
def test_invalid_source_training_is_rejected(averages, change):
    train, _, spec = averages
    checkpoint = train / "checkpoint-0240"
    if change == "incomplete":
        (train / "completed.json").write_text(json.dumps({"completed": False, "steps": 320}))
    elif change == "config":
        (checkpoint / "adapter_config.json").write_text(json.dumps({"r": 2}))
    else:
        weight = checkpoint / "adapter_model.safetensors"
        tensors = load_file(str(weight))
        if change == "precision":
            tensors["lora_A.weight"] = tensors["lora_A.weight"].bfloat16()
        elif change == "shape":
            tensors["lora_A.weight"] = torch.zeros(2, 1)
        else:
            tensors["lora_A.weight"].fill_(float("nan"))
        save_file(tensors, str(weight))
    with pytest.raises(ValueError):
        source_average(train, spec, 320)


@pytest.mark.parametrize("change", [None, "identity", "weights", "rule"])
def test_confirmation_rebuilds_frozen_average_before_regrading(averages, change):
    import dataclasses
    import shutil
    from scripts.analyze_advantage_confirmation import verified_arm
    from servedistill.data import grade
    from servedistill.runtime import summarize, write_json
    from servedistill.scheduling import Config
    train, _, spec = averages
    root = train.parent
    arm = root / "arm"
    arm.mkdir()
    shutil.move(str(train), str(arm / "train"))
    train = arm / "train"
    config = {"objective": "oracle_sft", "steps": 320, "seed": 29}
    write_json(train / "config.json", dataclasses.asdict(Config(**config)))
    evidence = materialize_average(train, arm / "averaged-adapter", spec, 320)
    sample = {"sample_id": "math/1", "domain": "math", "target": {"answer": "2"},
              "input_sha256": "cpu/math", "math_source": "SVAMP"}
    text = json.dumps(sample["target"])
    record = {**sample, "text": text, "output_tokens": 5, "truncated": False,
              **grade(sample, text)}
    write_json(arm / "start-provenance.json", {
        "phase": "confirmation", "label": "arm", "seed": 29,
        "protocol_sha256": "cpu/protocol", "test_used_for_selection": False})
    write_json(arm / "completed.json", {
        "completed": True, "phase": "confirmation", "checkpoint": "checkpoint-0320",
        "test_used_for_selection": False, "postprocessing": spec,
        "evaluated_adapter_file_sha256": evidence["adapter_file_sha256"]})
    final = arm / "final"
    final.mkdir()
    (final / "test.jsonl").write_text(json.dumps(record) + "\n")
    write_json(final / "test.summary.json", summarize([record]))
    identity = {key: evidence[key] for key in ("adapter_file_sha256", "adapter_config_sha256")}
    if change == "identity":
        identity["adapter_file_sha256"] = "different evaluated adapter"
    write_json(final / "evaluation-adapter.json", identity)
    if change == "weights":
        path = arm / "averaged-adapter/adapter_model.safetensors"
        tensors = load_file(str(path))
        tensors["lora_A.weight"].add_(1.)
        save_file(tensors, str(path))
    elif change == "rule":
        spec = {"kind": "lora_factor_mean_fp32", "steps": [160, 320]}
    def verify():
        return verified_arm(root, "arm", {}, "cpu/protocol", config,
                            {sample["sample_id"]: sample}, "test", postprocessing=spec)
    if change is None:
        rows, verified = verify()
        assert len(rows) == 1 and verified["final_checkpoint"] == "checkpoint-0320"
    else:
        with pytest.raises(ValueError):
            verify()


@pytest.mark.parametrize("steps", [[480, 640], [160, 320, 480, 640]])
def test_640_factor_average_binds_real_sources(tmp_path, steps):
    train = tmp_path / "train"
    train.mkdir()
    (train / "completed.json").write_text(json.dumps({"completed": True, "steps": 640}))
    for step in steps:
        checkpoint = train / f"checkpoint-{step:04d}"
        checkpoint.mkdir()
        save_file({"lora_A.weight": torch.tensor([[float(step)]])},
                  str(checkpoint / "adapter_model.safetensors"))
        (checkpoint / "adapter_config.json").write_text(json.dumps({"r": 1}))
        (checkpoint / "state.json").write_text(json.dumps(
            {"step": step, "adapter_sha256": str(step)}))
        (checkpoint / "training.pt").write_bytes(b"CPU optimizer fixture")
    spec = {"kind": "lora_factor_mean_fp32", "steps": steps}
    adapter = tmp_path / "average"
    evidence = materialize_average(train, adapter, spec, 640)
    actual = load_file(str(adapter / "adapter_model.safetensors"))["lora_A.weight"].item()
    assert actual == sum(steps) / len(steps)
    assert len(evidence["source_checkpoints"]) == len(steps)
    (train / f"checkpoint-{steps[0]:04d}/training.pt").write_bytes(b"changed optimizer")
    with pytest.raises(ValueError):
        verify_average(train, adapter, spec, 640)
