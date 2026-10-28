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

    def test_domain_weights_reject_typos_nonfinite_and_other_objectives(self):
        for weights in ({"mat": 1.0}, {"math": -1}, {"math": float("nan")},
                        {"math": float("inf")}, {"math": "1"}):
            with self.assertRaises(ValueError):
                update_learning_rate(Config(objective="aopd", anchor_ce_domain_weights=weights), 1)
        with self.assertRaises(ValueError):
            update_learning_rate(Config(objective="offline_sft",
                                        anchor_ce_domain_weights={"math": 1.0}), 1)

    def test_weighted_ce_gradient_matches_explicit_gold_objective(self):
        rows = [{"target": {"answer": "yes"}}, {"target": {"answer": "no"}},
                {"target": {"answer": "no"}}]
        tokens, mask = gold_responses(rows, Tokenizer(), 9)
        for aggregation in ("token", "trajectory"):
            parameter = torch.randn(3, 64, dtype=torch.float64, requires_grad=True)

            class Runtime:
                tokenizer = Tokenizer()
                eos_ids = [9]

                def score(self, batch, response, response_mask):
                    logits = parameter[:, None, :].expand(-1, response.shape[1], -1)
                    return {"logp": logits.log_softmax(-1).gather(
                        -1, response[..., None]).squeeze(-1)}

            actual = gold_ce(Runtime(), rows, aggregation, [.1, 1.0, 0.0])
            explicit = []
            for i, coefficient in enumerate((.1, 1.0, 0.0)):
                valid = tokens[i, mask[i]]
                nll = torch.nn.functional.cross_entropy(
                    parameter[i].expand(len(valid), -1), valid, reduction="sum")
                explicit.append(coefficient * nll / (
                    mask.sum() if aggregation == "token" else len(valid) * len(rows)))
            expected = sum(explicit)
            torch.testing.assert_close(actual, expected)
            got_gradient = torch.autograd.grad(actual, parameter)[0]
            expected_gradient = torch.autograd.grad(expected, parameter)[0]
            torch.testing.assert_close(got_gradient, expected_gradient)
            self.assertTrue(bool((got_gradient[2] == 0).all()))

    def test_full_update_is_invariant_to_microbatch_partition(self):
        rows = [{"domain": "search", "target": {"answer": "yes"}},
                {"domain": "math", "target": {"answer": "no"}},
                {"domain": "instruction", "target": {"answer": "no"}}]

        def update(microbatch, aggregation, weights=None):
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
                            aggregation=aggregation, anchor_ce_domain_weights=weights or {})
            with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.synchronize"):
                result = apply_rollout(
                    runtime, optimizer, config, DevWeightController(1), rollout,
                    torch.zeros(3), [parameter], [parameter.detach().clone()], 1,
                    Path(temporary), {"teacher": 0.0, "learner": 0.0}, lambda _: nullcontext())
            return parameter.detach(), result[0]

        for aggregation in ("token", "trajectory"):
            for weights in (None, {"search": .1, "math": 1.0, "instruction": 0.0},
                            {"math": 1.0}, {"search": .5, "math": .5, "instruction": .5}):
                reference, audit = update(3, aggregation, weights)
                for microbatch in (1, 2):
                    actual, got = update(microbatch, aggregation, weights)
                    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
                    self.assertAlmostEqual(got["anchor_weighted_ce_loss"],
                                           audit["anchor_weighted_ce_loss"], places=6)
            scalar, audit = update(3, aggregation)
            mapped, got = update(3, aggregation, {"search": .5, "math": .5, "instruction": .5})
            torch.testing.assert_close(mapped, scalar, rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(got["loss"], audit["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
