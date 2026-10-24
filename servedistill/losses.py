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
