"""Single-owner LoRA experiments: immutable rollouts, one update per batch."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import BUSINESS, canonical, grade, load_rows
from .losses import (
    DevWeightController, full_reverse_kl, grpo_advantages, mix_losses,
    opd_advantages, policy_loss,
)
from .runtime import Runtime, append_json, seed_all, stratified, write_json
from .scheduling import Config, schedule


def tensor_fingerprint(value) -> str:
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.shape, tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def parameter_state(runtime: Runtime) -> dict:
    return {name: parameter for name, parameter in runtime.student.named_parameters()
            if parameter.requires_grad}


def save_checkpoint(runtime, optimizer, config, controller, step, output, best_score,
                    tokens_seen, monitor_history) -> dict:
    path = output / f"checkpoint-{step:04d}"
    path.mkdir(parents=True, exist_ok=True)
    runtime.student.save_pretrained(path)
    state = {
        "optimizer": optimizer.state_dict(), "step": step,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "controller": vars(controller), "best_score": best_score,
        "tokens_seen": tokens_seen, "monitor_history": monitor_history,
        "config": dataclasses.asdict(config),
    }
    temporary = path / "training.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(path / "training.pt")
    metadata = {
        "step": step, "adapter_sha256": tensor_fingerprint(parameter_state(runtime)),
        "optimizer_sha256": tensor_fingerprint(state["optimizer"]),
        "rng_sha256": tensor_fingerprint([
            state["python_rng"], state["numpy_rng"], state["torch_rng"], state["cuda_rng"]]),
        "tokens_seen": tokens_seen,
    }
    write_json(path / "state.json", metadata)
    write_json(output / "latest.json", {"path": path.name, **metadata})
    return metadata


def load_checkpoint(runtime, optimizer, path: Path, config: Config, controller) -> dict:
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if state["config"] != dataclasses.asdict(config):
        raise ValueError("Resume configuration differs from the saved immutable configuration")
    optimizer.load_state_dict(state["optimizer"])
    for item in optimizer.state.values():
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.to(runtime.student.device)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    controller.__dict__.update(state["controller"])
    actual = {
        "adapter_sha256": tensor_fingerprint(parameter_state(runtime)),
        "optimizer_sha256": tensor_fingerprint(optimizer.state_dict()),
        "rng_sha256": tensor_fingerprint([
            random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all()]),
    }
    expected = json.loads((path / "state.json").read_text())
    if any(actual[key] != expected[key] for key in actual):
        raise RuntimeError("Cold-process restoration fingerprint mismatch")
    return {**state, "restoration": {"passed": True, "step": state["step"], **actual}}


def collate_rollouts(runtime: Runtime, prompts: list[dict], config: Config) -> dict:
    rows = [row for row in prompts for _ in range(config.group_size)]
    fragments = [
        runtime.rollout(rows[i:i + config.rollout_batch], config.max_new_tokens)
        for i in range(0, len(rows), config.rollout_batch)
    ]
    width = max(part["tokens"].shape[1] for part in fragments)
    result = {"rows": rows, "texts": [], "elapsed_seconds": 0.0}
    for key in ("tokens", "mask", "behavior_logp"):
        tensors = []
        for part in fragments:
            tensor = part[key]
            pad = width - tensor.shape[1]
            fill = runtime.tokenizer.pad_token_id if key == "tokens" else 0
            tensors.append(torch.nn.functional.pad(tensor, (0, pad), value=fill))
        result[key] = torch.cat(tensors)
    for part in fragments:
        result["texts"].extend(part["texts"])
        result["elapsed_seconds"] += part["elapsed_seconds"]
    result["groups"] = torch.arange(len(prompts)).repeat_interleave(config.group_size)
    return result


def alert(metrics: dict, history: list[dict]) -> list[str]:
    """Numerical contracts stop immediately; trend alarms require dev confirmation."""
    alarms = []
    if metrics["behavior_max_error"] > metrics["behavior_tolerance"]:
        raise RuntimeError("Behavior/learner logprob contract failed")
    if history:
        baseline = history[0]
        if (metrics["reward_mean"] < baseline["reward_mean"] - 0.2
                and metrics["entropy"] < baseline["entropy"] * 0.5
                and (abs(metrics["reference_sampled_kl"]) >
                     abs(baseline["reference_sampled_kl"]) + 0.3
                     or metrics["gradient_norm"] > max(1.0, baseline["gradient_norm"]) * 5)):
            alarms.append("reward_entropy_divergence_joint_alarm")
        if (metrics["truncation_rate"] > 0.5 and metrics["json_valid"] < 0.3
                and metrics["effective_group_fraction"] < 0.1):
            alarms.append("truncation_format_zero_signal_joint_alarm")
    return alarms


def train(args) -> dict:
    config = Config(**json.loads(Path(args.config).read_text()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    teacher_needed = config.objective not in ("oracle_sft", "offline_sft", "grpo")
    runtime = Runtime(
        args.student, args.teacher if teacher_needed else None, args.adapter or args.resume,
        train=True, teacher_device=args.teacher_device, train_precision=config.train_precision,
    )
    write_json(output / "contract.json", runtime.contract)
    write_json(output / "config.json", dataclasses.asdict(config))
    parameters = list(parameter_state(runtime).values())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    controller = DevWeightController(config.opd_weight)
    batches = schedule(load_rows(Path(args.data) / "train.jsonl"), config)
    dev = stratified(load_rows(Path(args.data) / "dev.jsonl"), config.dev_per_domain)
    write_json(output / "schedule.json", [[row["sample_id"] for row in b] for b in batches])
    teacher_labels = {}
    if config.objective == "offline_sft":
        if not args.teacher_labels:
            raise ValueError("Offline SFT requires frozen teacher trajectories")
        teacher_labels = {
            row["sample_id"]: row["text"] for row in load_rows(args.teacher_labels)}
    start_step, tokens_seen, best_score, history = 0, 0, -1.0, []
    if args.resume:
        restored = load_checkpoint(runtime, optimizer, Path(args.resume), config, controller)
        start_step, tokens_seen = restored["step"], restored["tokens_seen"]
        best_score, history = restored["best_score"], restored["monitor_history"]
        write_json(output / f"restoration-{start_step:04d}.json", restored["restoration"])
    frozen_teacher = (
        tensor_fingerprint({name: p.flatten()[:32] for name, p in runtime.teacher.named_parameters()})
        if runtime.teacher else None)
    started = time.monotonic()
    totals = {"rollout": 0.0, "teacher": 0.0, "learner": 0.0, "dev": 0.0}
    if start_step == 0:
        baseline = runtime.evaluate(dev, output, "dev-0000")
        controller.baseline_general = baseline["general_macro"]
        controller.last_business = baseline["business_macro"]
        write_json(output / "baseline.json", baseline)
        totals["dev"] += time.monotonic() - started
    for step in range(start_step, config.steps):
        optimizer.zero_grad(set_to_none=True)
        prompts = batches[step]
        before = [p.detach().clone() for p in parameters]
        version = step
        sft = config.objective in ("oracle_sft", "offline_sft")
        if sft:
            rows = prompts
            texts = [
                teacher_labels[row["sample_id"]] if config.objective == "offline_sft"
                else canonical(row["target"]) for row in rows]
            encoded = [runtime.tokenizer.encode(
                text, add_special_tokens=False) + [runtime.eos_ids[0]] for text in texts]
            width = max(map(len, encoded))
            tokens = torch.full((len(rows), width), runtime.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            for index, values in enumerate(encoded):
                tokens[index, :len(values)] = torch.tensor(values)
                mask[index, :len(values)] = True
            rollout = {"rows": rows, "texts": texts, "tokens": tokens, "mask": mask}
            rl_adv = None
            metrics = {"reward_mean": np.mean([grade(row, text)["reward"]
                                              for row, text in zip(rows, texts)]).item(),
                       "effective_group_fraction": 0.0, "reward_std": 0.0}
        else:
            rollout = collate_rollouts(runtime, prompts, config)
            rewards = torch.tensor([grade(row, text)["reward"]
                                    for row, text in zip(rollout["rows"], rollout["texts"])])
            rl_adv, metrics = grpo_advantages(rewards, rollout["groups"])
            totals["rollout"] += rollout["elapsed_seconds"]
        rows, tokens, mask = rollout["rows"], rollout["tokens"], rollout["mask"]
        total_actions = mask.sum().item()
        write_json(output / "last-rollout-contract.json", {
            "step": step + 1, "version": version,
            "sample_ids": [r["sample_id"] for r in rows],
            "token_sha256": tensor_fingerprint(tokens),
            "mask_sha256": tensor_fingerprint(mask),
            "texts": rollout["texts"],
        })
        audit = defaultdict(float)
        max_error = 0.0
        for start in range(0, len(rows), config.microbatch):
            end = min(start + config.microbatch, len(rows))
            batch, tok, m = rows[start:end], tokens[start:end], mask[start:end]
            learner_started = time.monotonic()
            teacher_seconds = 0.0
            current = runtime.score(batch, tok, m, entropy=True,
                                    keep_logits=config.objective == "full_kl")
            logp = current["logp"]
            m = m.to(logp.device)
            factor = m.sum().item() / total_actions if config.aggregation == "token" else len(batch) / len(rows)
            entropy = current["entropy"].detach()
            if sft:
                from .losses import reduce_actions
                loss = reduce_actions(-logp, m, config.aggregation)
                rl_loss, opd_loss = loss * 0, loss * 0
            else:
                old = rollout["behavior_logp"][start:end].to(logp.device)
                error = (logp.detach() - old)[m].abs().max().item()
                max_error = max(max_error, error)
                if error > config.behavior_tolerance:
                    write_json(output / "probability-failure.json", {
                        "step": step + 1, "version": version, "max_error": error,
                        "sample_ids": [r["sample_id"] for r in batch],
                        "tokens": tok.tolist(), "mask": m.cpu().tolist(),
                        "behavior_logp": old.cpu().tolist(),
                        "recomputed_logp": logp.detach().cpu().tolist()})
                    raise RuntimeError(f"Version {version}: rollout/recompute error {error:.6f}")
                rl_signal = rl_adv[start:end].to(logp.device)[:, None].expand_as(logp)
                rl_loss, ratios = policy_loss(logp, old, rl_signal, m, config.clip, config.aggregation)
                for key, value in ratios.items():
                    audit[key] += value * factor
                opd_loss = logp.sum() * 0
                if runtime.teacher:
                    teacher_started = time.monotonic()
                    teacher = runtime.score(batch, tok, m, teacher=True,
                                            keep_logits=config.objective == "full_kl")
                    torch.cuda.synchronize()
                    teacher_seconds = time.monotonic() - teacher_started
                    totals["teacher"] += teacher_seconds
                    teacher_logp = teacher["logp"].to(logp.device)
                    signal = opd_advantages(old, teacher_logp, m)
                    from .losses import reduce_actions
                    if config.objective == "full_kl":
                        kl = full_reverse_kl(
                            current["logits"], teacher["logits"].to(logp.device))
                        opd_loss = reduce_actions(kl, m, config.aggregation)
                    else:
                        opd_loss, _ = policy_loss(
                            logp, old, signal, m, config.clip, config.aggregation)
                    audit["teacher_sampled_kl"] += reduce_actions(-signal, m, config.aggregation).item() * factor
                    del teacher
                if config.objective == "full_kl":
                    loss = opd_loss
                else:
                    loss = mix_losses(rl_loss, opd_loss, controller.weight,
                                      config.objective, step)
                with torch.no_grad():
                    reference = runtime.score(batch, tok, m, reference=True)["logp"]
                from .losses import reduce_actions
                audit["reference_sampled_kl"] += reduce_actions(
                    old - reference, m, config.aggregation).item() * factor
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite objective")
            audit["entropy"] += (entropy[m].mean().item() * factor)
            audit["loss"] += loss.detach().item() * factor
            audit["rl_loss"] += rl_loss.detach().item() * factor
            audit["opd_loss"] += opd_loss.detach().item() * factor
            # A limited parameter projection, explicitly not a full-gradient claim.
            if not sft and runtime.teacher and step % 10 == 0 and start == 0:
                projection = parameters[-1]
                g_rl = torch.autograd.grad(rl_loss, projection, retain_graph=True,
                                           allow_unused=True)[0]
                g_opd = torch.autograd.grad(opd_loss, projection, retain_graph=True,
                                            allow_unused=True)[0]
                if g_rl is not None and g_opd is not None:
                    audit["gradient_projection_cosine"] = torch.nn.functional.cosine_similarity(
                        g_rl.flatten().float(), g_opd.flatten().float(), dim=0).item()
                    audit["rl_gradient_projection_norm"] = g_rl.float().norm().item()
                    audit["opd_gradient_projection_norm"] = g_opd.float().norm().item()
            (loss * factor).backward()
            del current, logp, loss
            torch.cuda.synchronize()
            totals["learner"] += time.monotonic() - learner_started - teacher_seconds
        gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
        if not gradients_finite:
            raise FloatingPointError("Non-finite gradient; optimizer update refused")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True).item()
        optimizer.step()
        delta = sum((p.detach().float() - old.float()).square().sum().item()
                    for p, old in zip(parameters, before)) ** 0.5
        if not all(bool(torch.isfinite(p).all()) for p in parameters):
            raise FloatingPointError("Non-finite adapter parameters")
        if runtime.teacher:
            if any(p.grad is not None or p.requires_grad for p in runtime.teacher.parameters()):
                raise RuntimeError("Teacher gradient/freeze contract failed")
        grades = [grade(row, text) for row, text in zip(rows, rollout["texts"])]
        domain_metrics = {}
        for domain in set(row["domain"] for row in rows):
            indices = [i for i, row in enumerate(rows) if row["domain"] == domain]
            domain_metrics[domain] = {
                "prompts": sum(row["domain"] == domain for row in prompts),
                "sequences": len(indices), "response_tokens": int(mask[indices].sum()),
                "reward": np.mean([grades[i]["reward"] for i in indices]).item(),
                "success": np.mean([grades[i]["success"] for i in indices]).item(),
            }
        tokens_seen += total_actions
        truncation = sum(not any(int(t) in runtime.eos_ids for t in ts[ms])
                         for ts, ms in zip(tokens, mask)) / len(rows)
        metrics.update({
            **audit, "step": step + 1, "rollout_version": version,
            "learner_version": step + 1, "gradient_norm": norm, "parameter_delta_l2": delta,
            "response_tokens": total_actions, "cumulative_response_tokens": tokens_seen,
            "opd_weight": controller.weight, "behavior_max_error": max_error,
            "behavior_tolerance": config.behavior_tolerance,
            "json_valid": np.mean([g["json_valid"] for g in grades]).item(),
            "truncation_rate": truncation, "domains": domain_metrics,
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        })
        metrics["alarms"] = alert(metrics, history)
        history.append(metrics)
        append_json(output / "metrics.jsonl", metrics)
        append_json(output / "rollouts.jsonl", {
            "step": step + 1, "version": version,
            "sample_ids": [r["sample_id"] for r in rows],
            "texts": rollout["texts"], "rewards": [g["reward"] for g in grades],
        })
        print(json.dumps({"event": "update", **metrics}), flush=True)
        if (step + 1) % config.dev_every == 0 or step + 1 == config.steps:
            dev_started = time.monotonic()
            report = runtime.evaluate(dev, output, f"dev-{step + 1:04d}", version=step + 1)
            totals["dev"] += time.monotonic() - dev_started
            if config.objective == "dynamic":
                append_json(output / "weight_events.jsonl", {
                    "step": step + 1, **controller.update(
                        report["business_macro"], report["general_macro"])})
            score = 0.8 * report["business_macro"] + 0.2 * report["general_macro"]
            checkpoint = save_checkpoint(
                runtime, optimizer, config, controller, step + 1, output,
                max(best_score, score), tokens_seen, history)
            if score > best_score:
                best_score = score
                write_json(output / "selected.json", {
                    "path": f"checkpoint-{step + 1:04d}", "dev_score": score, **checkpoint})
            if metrics["alarms"]:
                baseline = history[0]
                if (report["business_macro"] < baseline["reward_mean"] - 0.2
                        and len(history) >= 3
                        and all(h["alarms"] for h in history[-3:])):
                    raise RuntimeError("Joint training alarm persisted and dev confirmed degradation")
        if args.stop_after and step + 1 >= args.stop_after:
            save_checkpoint(runtime, optimizer, config, controller, step + 1, output,
                            best_score, tokens_seen, history)
            write_json(output / "cold-stop.json", {"step": step + 1, "healthy": True})
            return {"completed": False, "cold_stop": True, "step": step + 1}
    teacher_unchanged = True
    if runtime.teacher:
        teacher_unchanged = frozen_teacher == tensor_fingerprint(
            {name: p.flatten()[:32] for name, p in runtime.teacher.named_parameters()})
        if not teacher_unchanged:
            raise RuntimeError("Frozen teacher fingerprint changed")
    result = {
        "completed": True, "steps": config.steps, "response_tokens": tokens_seen,
        "elapsed_seconds": time.monotonic() - started, "phase_seconds": totals,
        "teacher_unchanged": teacher_unchanged, "trainable_parameters": sum(p.numel() for p in parameters),
        "nonzero_updates": sum(h["parameter_delta_l2"] > 0 for h in history),
        "max_behavior_error": max(h["behavior_max_error"] for h in history),
        "selected": json.loads((output / "selected.json").read_text()),
    }
    write_json(output / "completed.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-labels")
    parser.add_argument("--adapter")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    try:
        print(json.dumps(train(args)), flush=True)
    except Exception as error:
        write_json(Path(args.output) / "failed.json", {
            "error": type(error).__name__, "message": str(error),
            "failed_unix_seconds": time.time()})
        raise


if __name__ == "__main__":
    main()
