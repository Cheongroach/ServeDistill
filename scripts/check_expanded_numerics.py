"""Independent FP64 reduction of a fixed-Teacher, four-sequence gradient audit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from servedistill.losses import DevWeightController
from servedistill.runtime import Runtime, write_json
from servedistill.scheduling import Config
from servedistill.train import apply_rollout, load_checkpoint, parameter_state, tensor_fingerprint
from scripts.benchmark_microbatch import PhaseTimer


def stable_comparison(a, b):
    """Compute reductions in FP64 without clipping an invalid cosine."""
    a, b = a.double(), b.double()
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    return {
        "gradient_cosine_fp64": float(torch.dot(a, b) / (norm_a * norm_b)),
        "gradient_relative_l2_fp64": float(torch.linalg.vector_norm(b - a) / norm_a),
        "gradient_max_abs_difference": float((b - a).abs().max()),
        "gradient_vector_elements": a.numel(),
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "checkpoint", "bank", "output", "destination"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from scripts.worker import publish
    runtime = Runtime(str(args.student), str(args.teacher), str(args.checkpoint), train=True)
    state = torch.load(args.checkpoint / "training.pt", map_location="cpu", weights_only=False)
    config = Config(**state["config"])
    parameters = list(parameter_state(runtime).values())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.)
    restored = load_checkpoint(runtime, optimizer, args.checkpoint, config, DevWeightController(1.))
    saved = [p.detach().clone() for p in parameters]
    rollout = torch.load(args.bank, map_location="cpu", weights_only=False)
    subset = {k: v[:4] if isinstance(v, (torch.Tensor, list)) else v for k, v in rollout.items()}
    frozen = runtime.score(subset["rows"], subset["tokens"], subset["mask"],
                           teacher=True, keep_logits=True)
    frozen = {k: v.detach().cpu() for k, v in frozen.items()}
    gradients, updates, audits = {}, {}, {}
    for microbatch in (1, 4):
        with torch.no_grad():
            for p, value in zip(parameters, saved):
                p.copy_(value)
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        for item in optimizer.state.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(runtime.student.device)
        optimizer.zero_grad(set_to_none=True)
        config.microbatch = microbatch
        result = apply_rollout(
            runtime, optimizer, config, DevWeightController(1.), subset,
            torch.zeros(4), parameters, saved, 41, args.output,
            {"teacher": 0., "learner": 0.}, PhaseTimer().span, frozen)
        audits[str(microbatch)] = {
            "loss": result[0]["loss"], "behavior_max_error": result[1],
            "preclip_gradient_norm": result[2], "parameter_delta_l2": result[3]}
        gradients[microbatch] = torch.cat([
            p.grad.detach().cpu().double().reshape(-1)
            for p in parameters if p.grad is not None])
        updates[microbatch] = torch.cat([
            (p.detach().cpu().double() - old.cpu().double()).reshape(-1)
            for p, old in zip(parameters, saved)])
    result = {
        "completed": True, "parent_restoration": restored["restoration"],
        "bank_token_sha256": tensor_fingerprint(rollout["tokens"]),
        "teacher_logits_sha256": tensor_fingerprint(frozen["logits"]),
        "sequences": 4, "audits": audits,
        **stable_comparison(gradients[1], gradients[4]),
        "update_relative_l2_fp64": float(
            torch.linalg.vector_norm(updates[4] - updates[1]) / torch.linalg.vector_norm(updates[1])),
        "boundary": "Independent fixed-Teacher four-sequence post-clip audit; outside formal timing",
        "reduction_dtype": "float64",
    }
    torch.save({"gradient_m1": gradients[1], "gradient_m4": gradients[4],
                "update_m1": updates[1], "update_m4": updates[4]}, args.output / "vectors.pt")
    write_json(args.output / "comparison.json", result)
    publish(args.output, args.destination)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
