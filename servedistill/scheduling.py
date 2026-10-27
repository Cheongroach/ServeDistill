"""Pure reproducible prompt budgets and immutable training configurations."""
from __future__ import annotations
import dataclasses
import math
import random
from collections import defaultdict

@dataclasses.dataclass
class Config:
    objective: str = "mixed"
    train_precision: str = "float32_torch"
    business_domains: list[str] = dataclasses.field(
        default_factory=lambda: ["search", "relevance", "substitute"])
    seed: int = 17
    steps: int = 80
    prompts_per_step: int = 5
    group_size: int = 4
    microbatch: int = 1
    rollout_batch: int = 8
    max_new_tokens: int = 128
    learning_rate: float = 2e-5
    learning_rate_schedule: str = "constant"
    learning_rate_decay_start: int = 320
    learning_rate_final_ratio: float = 0.10
    opd_weight: float = 1.0
    aggregation: str = "token"
    bucketed: bool = False
    business_only: bool = False
    general_only: bool = False
    dev_every: int = 20
    dev_per_domain: int = 20
    clip: float = 0.2
    behavior_tolerance: float = 0.15
    aopd_top_k: int = 32
    aopd_tau: float = 0.0
    anchor_ce_weight: float = 0.0
    method_time_budget_seconds: float | None = None
    prior_method_seconds: float = 0.0


def update_learning_rate(config: Config, update: int) -> float:
    """Return the rate used by this one-indexed complete optimizer update."""
    if not 1 <= update <= config.steps:
        raise ValueError("Learning-rate update lies outside the declared budget")
    if not math.isfinite(config.anchor_ce_weight) or config.anchor_ce_weight < 0:
        raise ValueError("Anchor CE weight must be finite and nonnegative")
    if config.anchor_ce_weight and config.objective != "aopd":
        raise ValueError("Supervised anchor is defined only for AOPD")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if config.learning_rate_schedule == "constant":
        return config.learning_rate
    if config.learning_rate_schedule != "constant_then_linear":
        raise ValueError("Unknown learning-rate schedule")
    if not 0 < config.learning_rate_decay_start < config.steps:
        raise ValueError("Decay must start strictly inside the update budget")
    if (
        not math.isfinite(config.learning_rate_final_ratio)
        or not 0 < config.learning_rate_final_ratio <= 1
    ):
        raise ValueError("Final learning-rate ratio must lie in (0, 1]")
    fraction = max(0, update - config.learning_rate_decay_start) / (
        config.steps - config.learning_rate_decay_start
    )
    return config.learning_rate * (
        1.0 - (1.0 - config.learning_rate_final_ratio) * fraction
    )


def schedule(rows: list[dict], config: Config) -> list[list[dict]]:
    rng = random.Random(config.seed)
    pools = defaultdict(list)
    for row in rows:
        pools[row["domain"]].append(row)
    if config.business_only and config.general_only:
        raise ValueError("business_only and general_only are mutually exclusive")
    required = (("math", "instruction") if config.general_only
                else tuple(config.business_domains) + (() if config.business_only
                                                       else ("math", "instruction")))
    if any(not pools[d] for d in required):
        raise ValueError("Dataset is missing a configured training domain")
    sequence = []
    for index in range(config.steps * config.prompts_per_step):
        if config.general_only:
            domain = ("math", "instruction")[index % 2]
        elif config.business_only:
            domain = config.business_domains[index % len(config.business_domains)]
        elif index % 5 != 4:
            domain = config.business_domains[
                (index - index // 5) % len(config.business_domains)]
        else:
            domain = ("math", "instruction")[(index // 5) % 2]
        sequence.append(rng.choice(pools[domain]))
    # Same sequence and total domain budget, reorganized in bounded windows.
    if config.bucketed:
        window = config.prompts_per_step * 10
        sequence = [
            row for start in range(0, len(sequence), window)
            for row in sorted(sequence[start:start + window], key=lambda r: r["domain"])
        ]
    return [sequence[i:i + config.prompts_per_step]
            for i in range(0, len(sequence), config.prompts_per_step)]
