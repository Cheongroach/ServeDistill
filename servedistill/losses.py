"""Explicit token-level OPD and trajectory-level GRPO contracts."""
from __future__ import annotations

import torch
from torch import Tensor


def reduce_actions(values: Tensor, mask: Tensor, mode: str = "token") -> Tensor:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("Expected matching [sequences, response_tokens] tensors")
    counts = mask.sum(-1)
    if bool((counts <= 0).any()):
        raise ValueError("Every sequence must contain at least one action")
    if mode == "token":
        return (values * mask).sum() / counts.sum()
    if mode == "trajectory":
        return ((values * mask).sum(-1) / counts).mean()
    raise ValueError("Unknown aggregation: " + mode)


def grpo_advantages(rewards: Tensor, groups: Tensor) -> tuple[Tensor, dict]:
    if rewards.ndim != 1 or rewards.shape != groups.shape:
        raise ValueError("Rewards and group IDs must be matching vectors")
    result = torch.zeros_like(rewards)
    effective = 0
    group_ids = torch.unique(groups)
    for key in group_ids:
        select = groups == key
        values = rewards[select]
        if values.numel() < 2:
            raise ValueError("GRPO requires at least two completions per prompt")
        std = values.std(unbiased=False)
        if std.item() > 1e-8:
            result[select] = (values - values.mean()) / (std + 1e-8)
            effective += 1
    return result.detach(), {
        "effective_group_fraction": effective / len(group_ids),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std(unbiased=False).item(),
    }


def policy_loss(
    current_logp: Tensor, behavior_logp: Tensor, advantages: Tensor, mask: Tensor,
    clip: float = 0.2, aggregation: str = "token",
) -> tuple[Tensor, dict]:
    if not (current_logp.shape == behavior_logp.shape == advantages.shape == mask.shape):
        raise ValueError("Policy tensors and response mask must have identical shapes")
    if not bool(torch.isfinite(current_logp[mask.bool()]).all()):
        raise FloatingPointError("Non-finite learner logprob")
    # Clamping extreme log-ratios is a numerical guard; raw ratios are audited first.
    log_ratio = current_logp - behavior_logp.detach()
    if bool((log_ratio[mask.bool()].abs() > 20).any()):
        raise FloatingPointError("Behavior/current probability ratio exceeds numerical guard")
    ratio = log_ratio.exp()
    advantage = advantages.detach()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    selected = ratio[mask.bool()].detach()
    return reduce_actions(loss, mask, aggregation), {
        "clip_fraction": ((selected - 1).abs() > clip).float().mean().item(),
        "ratio_min": selected.min().item(), "ratio_max": selected.max().item(),
        "ess_fraction": (
            selected.sum().square() / (selected.square().sum() * selected.numel())
        ).item(),
    }


def opd_advantages(behavior_logp: Tensor, teacher_logp: Tensor, mask: Tensor) -> Tensor:
    if behavior_logp.shape != teacher_logp.shape or mask.shape != behavior_logp.shape:
        raise ValueError("Teacher/behavior response positions do not match")
    if not bool(torch.isfinite(teacher_logp[mask.bool()]).all()):
        raise FloatingPointError("Non-finite teacher logprob")
    # Immediate token signal; negative rewards are retained; no group normalization.
    return ((teacher_logp - behavior_logp) * mask).detach()


def asymmetric_opd_loss(
    current_logp: Tensor,
    behavior_logp: Tensor,
    teacher_logp: Tensor,
    current_logits: Tensor,
    teacher_logits: Tensor,
    mask: Tensor,
    top_k: int = 32,
    tau: float = 0.0,
    clip: float = 0.2,
    aggregation: str = "token",
    current_log_normalizer: Tensor | None = None,
    teacher_log_normalizer: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """AOPD: positive sampled-token OPD plus localized teacher top-k forward KL."""
    if not (
        current_logp.shape == behavior_logp.shape == teacher_logp.shape == mask.shape
    ):
        raise ValueError("AOPD sampled-token tensors must have identical shapes")
    if (
        current_logits.shape != teacher_logits.shape
        or current_logits.shape[:-1] != current_logp.shape
    ):
        raise ValueError("AOPD logits must share the sampled-token prefix shape")
    if top_k <= 0 or top_k > current_logits.shape[-1]:
        raise ValueError("AOPD top_k must be within the vocabulary")
    for name, normalizer in (
        ("current", current_log_normalizer),
        ("teacher", teacher_log_normalizer),
    ):
        if normalizer is not None and normalizer.shape != current_logp.shape:
            raise ValueError(f"AOPD {name} log normalizer shape differs")
    selected = mask.bool()

    signal = opd_advantages(behavior_logp, teacher_logp, mask)
    intervention = (
        teacher_logp.detach().float().exp() - behavior_logp.detach().float().exp()
        <= tau
    ) & selected
    positive_signal = signal * (~intervention)
    positive_loss, metrics = policy_loss(
        current_logp, behavior_logp, positive_signal, mask, clip, aggregation
    )

    if bool(intervention.any()):
        student_selected = current_logits[intervention].float()
        teacher_selected = teacher_logits.detach()[intervention].float()
        if not bool(torch.isfinite(teacher_selected).all()):
            raise FloatingPointError("Non-finite AOPD teacher logits")
        if current_log_normalizer is None:
            student_normalizer = student_selected.logsumexp(-1)
        else:
            student_normalizer = current_log_normalizer[intervention].float()
        if teacher_log_normalizer is None:
            teacher_normalizer = teacher_selected.logsumexp(-1)
        else:
            teacher_normalizer = teacher_log_normalizer[intervention].detach().float()
        teacher_top_logits, teacher_top_indices = teacher_selected.topk(top_k, dim=-1)
        teacher_top_logp = teacher_top_logits - teacher_normalizer[:, None]
        student_top_logp = (
            student_selected.gather(-1, teacher_top_indices)
            - student_normalizer[:, None]
        )
        # Eq. 7 uses full teacher probabilities on its selected support without
        # renormalizing. Compute only intervention rows; other rows multiply by zero.
        forward_kl = (
            teacher_top_logp.exp() * (teacher_top_logp - student_top_logp)
        ).sum(-1) / top_k
        guidance = torch.zeros_like(current_logp).masked_scatter(
            intervention, forward_kl
        )
        guidance_loss = reduce_actions(guidance, mask, aggregation)
    else:
        guidance_loss = current_logp.sum() * 0
    response_tokens = selected.sum()
    metrics.update({
        "aopd_intervention_fraction": (
            intervention.sum().float() / response_tokens
        ).item(),
        "aopd_positive_fraction": (
            ((~intervention) & selected).sum().float() / response_tokens
        ).item(),
        "aopd_guidance_loss": guidance_loss.detach().item(),
    })
    return positive_loss + guidance_loss, metrics


def full_reverse_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Full KL requires the same event space")
    logp = student_logits.float().log_softmax(-1)
    logq = teacher_logits.detach().float().log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def mix_losses(rl_loss: Tensor, opd_loss: Tensor, weight: float, objective: str, step: int) -> Tensor:
    if weight < 0:
        raise ValueError("Negative OPD weight")
    if objective == "grpo":
        return rl_loss
    if objective == "opd":
        return opd_loss
    if objective in ("mixed", "dynamic"):
        return rl_loss + weight * opd_loss
    if objective == "alternating":
        return rl_loss if step % 2 == 0 else weight * opd_loss
    raise ValueError("Unknown objective: " + objective)


class DevWeightController:
    """Bounded dev-based control, not a claim of gradient balancing."""

    def __init__(self, weight: float = 1.0, minimum: float = 0.25, maximum: float = 4.0):
        self.weight = weight
        self.minimum = minimum
        self.maximum = maximum
        self.baseline_general = None
        self.last_business = None

    def update(self, business: float, general: float) -> dict:
        old = self.weight
        reason = "stable"
        if self.baseline_general is None:
            self.baseline_general = general
        elif general < self.baseline_general - 0.03:
            self.weight = min(self.maximum, self.weight * 1.25)
            reason = "dev_general_regression"
        elif self.last_business is not None and business <= self.last_business:
            self.weight = max(self.minimum, self.weight / 1.25)
            reason = "dev_business_stalled"
        self.last_business = business
        return {"old_weight": old, "weight": self.weight, "reason": reason}
