"""Gold answer supervision from training rows with the response/EOS mask contract."""
import torch

from .data import canonical
from .losses import reduce_actions


def gold_responses(rows, tokenizer, eos_id):
    encoded = [tokenizer.encode(canonical(row["target"]), add_special_tokens=False) + [eos_id]
               for row in rows]
    width = max(map(len, encoded))
    tokens = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, values in enumerate(encoded):
        tokens[index, :len(values)] = torch.tensor(values)
        mask[index, :len(values)] = True
    return tokens, mask


def gold_ce(runtime, rows, aggregation, weights=None):
    """Weight each row without renormalizing by the sum of its coefficients."""
    tokens, mask = gold_responses(rows, runtime.tokenizer, runtime.eos_ids[0])
    scores = runtime.score(rows, tokens, mask)
    losses = -scores["logp"]
    if weights is not None:
        coefficients = torch.as_tensor(weights, dtype=losses.dtype, device=losses.device)
        if (coefficients.shape != (len(rows),)
                or not bool(torch.isfinite(coefficients).all())
                or bool((coefficients < 0).any())):
            raise ValueError("Require one finite nonnegative CE coefficient per gold row")
        losses = losses * coefficients[:, None]
    return reduce_actions(losses, mask.to(losses.device), aggregation)
