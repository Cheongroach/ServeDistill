"""Measure profiler checkpoint drift without relabeling it as exact equality."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from servedistill.profiling import digest


def compare_tensors(reference: dict, candidate: dict, tolerance: dict) -> dict:
    keys_match = set(reference) == set(candidate)
    records = []
    for key in sorted(set(reference) & set(candidate)):
        left, right = reference[key], candidate[key]
        compatible = left.shape == right.shape and left.dtype == right.dtype
        record = {"key": key, "shape_dtype_match": compatible}
        if compatible:
            left, right = left.double(), right.double()
            finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            difference = right - left
            norm = float(left.norm())
            record.update({
                "finite": finite, "exact": torch.equal(left, right),
                "max_abs_error": float(difference.abs().max()) if difference.numel() else 0.0,
                "difference_l2": float(difference.norm()), "reference_l2": norm,
                "relative_l2": float(difference.norm()) / norm if norm else None,
                "within_tolerance": finite and torch.allclose(left, right, **tolerance),
            })
        records.append(record)
    return {"passed": keys_match and bool(records)
                     and all(row.get("within_tolerance", False) for row in records),
            "keys_match": keys_match, "reference_keys": len(reference),
            "candidate_keys": len(candidate), "tolerance": tolerance, "tensors": records}


def optimizer_tensors(state: dict) -> dict:
    return {str(parameter) + "/" + key: value
            for parameter, values in state["state"].items()
            for key, value in values.items() if isinstance(value, torch.Tensor)}


def compare(reference: Path, candidate: Path, gates: dict) -> dict:
    adapter = compare_tensors(
        load_file(str(reference / "adapter_model.safetensors")),
        load_file(str(candidate / "adapter_model.safetensors")), gates["adapter"])
    # These are locally collected, SHA-verified training checkpoints. Do not load untrusted pickles.
    left = torch.load(reference / "training.pt", map_location="cpu", weights_only=False)
    right = torch.load(candidate / "training.pt", map_location="cpu", weights_only=False)
    optimizer = compare_tensors(optimizer_tensors(left["optimizer"]),
                                optimizer_tensors(right["optimizer"]), gates["optimizer"])
    optimizer["param_groups_equal"] = left["optimizer"]["param_groups"] == right["optimizer"]["param_groups"]
    optimizer["passed"] &= optimizer["param_groups_equal"]
    left_state = json.loads((reference / "state.json").read_text())
    right_state = json.loads((candidate / "state.json").read_text())
    rng_equal = left_state["rng_sha256"] == right_state["rng_sha256"]
    same_step_tokens = (left_state["step"], left_state["tokens_seen"]) == (
        right_state["step"], right_state["tokens_seen"])
    exact = all(left_state[key] == right_state[key]
                for key in ("adapter_sha256", "optimizer_sha256", "rng_sha256"))
    return {"passed": adapter["passed"] and optimizer["passed"] and same_step_tokens
                     and (rng_equal or not gates["require_identical_rng"]),
            "exact_final_state_equivalence": exact,
            "rng_equal": rng_equal, "same_step_and_tokens": same_step_tokens,
            "adapter": adapter, "optimizer": optimizer,
            "reference_state_sha256": digest(reference / "state.json"),
            "candidate_state_sha256": digest(candidate / "state.json"),
            "boundary": gates["boundary"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates", type=Path, default=Path(
        __file__).resolve().parents[1] / "configs/validity/profile-numerics.json")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, json.loads(args.gates.read_text()))
    result["gates_sha256"] = digest(args.gates)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "passed", "exact_final_state_equivalence", "rng_equal", "same_step_and_tokens")}))


if __name__ == "__main__":
    main()
