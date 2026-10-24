"""Cached generation versus teacher-forced probability recomputation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from .data import load_rows
from .runtime import Runtime, append_json, seed_all, stratified, write_json

class FloatHead(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = torch.nn.Parameter(original.weight.float(), requires_grad=False)
    def forward(self, value):
        return torch.nn.functional.linear(value.float(), self.weight)

def configure(runtime, variant):
    model = runtime.student.get_base_model()
    if variant == "float_head":
        model.lm_head = FloatHead(model.lm_head)
    elif variant == "float32_torch":
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
            torch_causal_conv1d_update)
        model.float()
        for layer in model.model.layers:
            attention = getattr(layer, "linear_attn", None)
            if attention is not None:
                attention.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                attention.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
                attention.causal_conv1d_fn = None
                attention.causal_conv1d_update = torch_causal_conv1d_update

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--variants", nargs="+", default=["bf16", "float_head", "float32_torch"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = stratified(load_rows(Path(args.data) / "train.jsonl"), args.per_domain)
    reports = []
    for variant in args.variants:
        seed_all(17)
        runtime = Runtime(args.model, train=True, train_precision="bf16")
        configure(runtime, variant)
        started = time.monotonic()
        errors, backward_done = [], False
        for row in rows:
            rollout = runtime.rollout([row], args.max_new_tokens)
            current = runtime.score([row], rollout["tokens"], rollout["mask"])
            selected = rollout["mask"].to(current["logp"].device)
            difference = (current["logp"].detach().cpu() - rollout["behavior_logp"])[rollout["mask"]].abs()
            errors.extend(difference.tolist())
            if not backward_done:
                loss = -current["logp"][selected].mean()
                loss.backward()
                gradients = [p.grad for p in runtime.student.parameters() if p.requires_grad and p.grad is not None]
                norm = sum(g.float().square().sum().item() for g in gradients) ** 0.5
                finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
                if not finite or norm <= 0:
                    raise RuntimeError("Diagnostic backward was not finite and nonzero")
                runtime.student.zero_grad(set_to_none=True)
                backward_done = True
            append_json(args.output / "cases.jsonl", {
                "variant": variant, "sample_id": row["sample_id"], "domain": row["domain"],
                "max_error": max(difference.tolist()), "tokens": int(selected.sum()),
                "text": rollout["texts"][0]})
            del current, rollout
        value = {"variant": variant, "n_sequences": len(rows), "n_actions": len(errors),
                 "max_error": max(errors), "mean_error": sum(errors) / len(errors),
                 "p99_error": float(torch.quantile(torch.tensor(errors), 0.99)),
                 "elapsed_seconds": time.monotonic() - started,
                 "backward_finite_nonzero": finite, "gradient_norm": norm,
                 "parameter_dtype": str(next(runtime.student.parameters()).dtype)}
        reports.append(value)
        write_json(args.output / "summary.json", reports)
        print(json.dumps({"event": "probability_probe", **value}), flush=True)
        del runtime
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
