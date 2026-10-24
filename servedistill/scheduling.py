"""Pure reproducible prompt budgets and immutable training configurations."""
from __future__ import annotations
import dataclasses
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
    opd_weight: float = 1.0
    aggregation: str = "token"
    bucketed: bool = False
    business_only: bool = False
    general_only: bool = False
    dev_every: int = 20
    dev_per_domain: int = 20
    clip: float = 0.2
    behavior_tolerance: float = 0.15


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
