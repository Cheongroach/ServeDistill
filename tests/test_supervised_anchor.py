import unittest
from contextlib import nullcontext
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from servedistill.scheduling import Config, update_learning_rate
from servedistill.supervision import gold_ce, gold_responses
from servedistill.train import apply_rollout
from servedistill.losses import DevWeightController


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [len(text)] if "yes" in text else [1, 2]


class SupervisedAnchorTests(unittest.TestCase):
    def test_eos_included_padding_excluded_and_student_gradient(self):
        rows = [{"target": {"answer": "yes"}}, {"target": {"answer": "no"}}]
        tokens, mask = gold_responses(rows, Tokenizer(), 9)
        self.assertEqual(tokens.tolist()[0][-1], 0)
        self.assertEqual(mask.tolist(), [[True, True, False], [True, True, True]])
        parameter = torch.tensor(0.2, requires_grad=True)

        class Runtime:
            tokenizer = Tokenizer()
            eos_ids = [9]

            def score(self, got_rows, response, got_mask):
                self.assertions = (got_rows == rows and response.tolist() == tokens.tolist())
                return {"logp": parameter.expand_as(response)}

        runtime = Runtime()
        loss = gold_ce(runtime, rows, "token")
        loss.backward()
        self.assertTrue(runtime.assertions)
        self.assertAlmostEqual(parameter.grad.item(), -1.0)

    def test_invalid_anchor_cannot_silently_change_other_methods(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                update_learning_rate(Config(objective="aopd", anchor_ce_weight=weight), 1)
        with self.assertRaises(ValueError):
            update_learning_rate(Config(objective="oracle_sft", anchor_ce_weight=0.5), 1)
        self.assertGreater(update_learning_rate(Config(objective="aopd", anchor_ce_weight=0.5), 1), 0)

    def test_full_update_is_invariant_to_microbatch_partition(self):
        rows = [{"target": {"answer": "yes"}}, {"target": {"answer": "no"}},
                {"target": {"answer": "no"}}]

        def update(microbatch, aggregation):
            torch.manual_seed(37)
            parameter = torch.nn.Parameter(torch.randn(64) * 0.1)

            class Runtime:
                tokenizer = Tokenizer()
                eos_ids = [9]
                teacher = torch.nn.Linear(1, 1).requires_grad_(False)

                def score(self, batch, response, mask, **options):
                    logits = parameter.expand(len(batch), response.shape[1], -1)
                    if options.get("teacher") or options.get("reference"):
                        logits = logits.detach()
                    logp_all = logits.log_softmax(-1)
                    return {
                        "logp": logp_all.gather(-1, response[..., None]).squeeze(-1),
                        "logits": logits, "log_normalizer": logits.logsumexp(-1),
                        "entropy": -(logp_all.exp() * logp_all).sum(-1),
                    }

            runtime = Runtime()
            tokens = torch.tensor([[1, 2, 9], [3, 9, 0], [4, 9, 0]])
            mask = torch.tensor([[True, True, True], [True, True, False], [True, True, False]])
            old = runtime.score(rows, tokens, mask)["logp"].detach()
            rollout = {"rows": rows, "tokens": tokens, "mask": mask, "behavior_logp": old}
            optimizer = torch.optim.SGD([parameter], lr=0.01)
            config = Config(objective="aopd", anchor_ce_weight=0.5, microbatch=microbatch,
                            aggregation=aggregation)
            with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.synchronize"):
                result = apply_rollout(
                    runtime, optimizer, config, DevWeightController(1), rollout,
                    torch.zeros(3), [parameter], [parameter.detach().clone()], 1,
                    Path(temporary), {"teacher": 0.0, "learner": 0.0}, lambda _: nullcontext())
            return parameter.detach(), result[0]

        for aggregation in ("token", "trajectory"):
            reference, audit = update(3, aggregation)
            for microbatch in (1, 2):
                actual, got = update(microbatch, aggregation)
                torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
                self.assertAlmostEqual(got["anchor_ce_loss"], audit["anchor_ce_loss"], places=6)


if __name__ == "__main__":
    unittest.main()
