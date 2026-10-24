import unittest

import torch

from servedistill.data import business_oracle, grade, make_business
from servedistill.losses import (
    DevWeightController, full_reverse_kl, grpo_advantages,
    opd_advantages, policy_loss, reduce_actions,
)


class ProbabilityContractTests(unittest.TestCase):
    def test_expected_sampled_gradient_matches_fixed_state_reverse_kl(self):
        logits = torch.tensor([0.4, -0.2, 1.1], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([-0.2, 0.8, 0.1], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(-1)
        logq = teacher.log_softmax(-1)
        advantage = (logq - logp).detach()
        expected_pg = -(logp.exp().detach() * advantage * logp).sum()
        actual_grad = torch.autograd.grad(expected_pg, logits, retain_graph=True)[0]
        exact = (logp.exp() * (logp - logq.detach())).sum()
        exact_grad = torch.autograd.grad(exact, logits)[0]
        torch.testing.assert_close(actual_grad, exact_grad)

    def test_full_kl_teacher_is_frozen(self):
        student = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        teacher = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        loss = full_reverse_kl(student, teacher).sum()
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertGreaterEqual(loss.item(), 0)

    def test_negative_token_logratio_is_preserved_and_detached(self):
        behavior = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        teacher = torch.tensor([[-3.0, -1.0]], requires_grad=True)
        result = opd_advantages(behavior, teacher, torch.ones_like(behavior))
        self.assertEqual(result.tolist(), [[-2.0, 1.0]])
        self.assertFalse(result.requires_grad)

    def test_zero_variance_grpo_does_not_create_updates(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 1.0])
        advantages, metrics = grpo_advantages(rewards, torch.tensor([0, 0, 1, 1]))
        self.assertEqual(advantages[:2].tolist(), [0.0, 0.0])
        self.assertEqual(metrics["effective_group_fraction"], 0.5)
        self.assertAlmostEqual(advantages[2:].mean().item(), 0.0)

    def test_prompt_and_padding_are_not_actions(self):
        current = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
        behavior = current.detach().clone()
        mask = torch.tensor([[0.0, 1.0, 0.0]])
        loss, _ = policy_loss(current, behavior, torch.ones_like(current), mask)
        loss.backward()
        self.assertEqual(current.grad.tolist(), [[0.0, -1.0, 0.0]])

    def test_token_and_trajectory_aggregation_are_distinct(self):
        values = torch.tensor([[1.0, 0.0], [3.0, 3.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        self.assertAlmostEqual(reduce_actions(values, mask, "token").item(), 7 / 3, places=6)
        self.assertAlmostEqual(reduce_actions(values, mask, "trajectory").item(), 2.0)

    def test_bad_alignment_fails(self):
        with self.assertRaises(ValueError):
            opd_advantages(torch.ones(2, 4), torch.ones(2, 3), torch.ones(2, 4))

    def test_dynamic_control_is_bounded_and_uses_dev(self):
        controller = DevWeightController()
        controller.update(0.6, 0.9)
        for _ in range(20):
            event = controller.update(0.6, 0.8)
        self.assertEqual(event["weight"], 4.0)
        self.assertEqual(event["reason"], "dev_general_regression")


class BusinessOracleTests(unittest.TestCase):
    def test_missing_information_clarifies(self):
        row = make_business("dining", "train", 10, 17)
        self.assertEqual(row["target"], {"choice": "NONE", "action": "clarify"})

    def test_coupon_threshold_and_ranking(self):
        merchant = {
            "id": "a", "unit_price": 60, "fee": 10, "coupon": 50,
            "coupon_threshold": 150, "capacity": 3, "distance": 1,
            "rating": 50, "available": True, "refundable": True, "certified": True,
        }
        spec = {
            "domain": "dining", "ranking": "rating",
            "need": {"budget": 100, "quantity": 2, "radius": 2, "refundable": True},
            "candidates": [merchant],
        }
        self.assertEqual(business_oracle(spec)["action"], "reject")
        merchant["coupon_threshold"] = 100
        self.assertEqual(business_oracle(spec), {"choice": "a", "action": "recommend"})

    def test_grader_rejects_extra_fields_and_prose(self):
        row = {"domain": "instruction", "target": {"answer": "42"}}
        self.assertTrue(grade(row, '{"answer":"42"}')["success"])
        self.assertFalse(grade(row, '{"answer":"42","explanation":"x"}')["success"])
        self.assertFalse(grade(row, 'Sure: {"answer":"42"}')["success"])

    def test_candidate_order_does_not_change_oracle(self):
        for domain in ("dining", "hotel", "service"):
            for number in range(20):
                row = make_business(domain, "test", number, 17)
                row["spec"]["candidates"].reverse()
                self.assertEqual(business_oracle(row["spec"]), row["target"])


class SearchLabelTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "domain": "search", "task": "esci_ranking", "locale": "us",
            "labels": {"P0": "E", "P1": "S", "P2": "I"},
            "target": {"ranking": ["P0", "P1", "P2"]},
        }

    def test_real_labels_reward_order_and_reject_duplicate_ids(self):
        best = grade(self.row, '{"ranking":["P0","P1","P2"]}')
        reversed_result = grade(self.row, '{"ranking":["P2","P1","P0"]}')
        duplicate = grade(self.row, '{"ranking":["P0","P0","P2"]}')
        self.assertEqual(best["ndcg"], 1)
        self.assertLess(reversed_result["ndcg"], best["ndcg"])
        self.assertFalse(duplicate["json_valid"])
        self.assertEqual(duplicate["reward"], 0)

    def test_zero_relevance_is_flagged_for_metric_exclusion(self):
        self.row["labels"] = {key: "I" for key in self.row["labels"]}
        result = grade(self.row, '{"ranking":["P0","P1","P2"]}')
        self.assertFalse(result["ndcg_eligible"])
        self.assertEqual(result["reward"], 0)

    def test_relevance_labels_are_not_teacher_judgements(self):
        row = {
            "domain": "relevance", "task": "esci_classification", "locale": "es",
            "target": {"label": "C"},
        }
        self.assertTrue(grade(row, '{"label":"C"}')["success"])
        self.assertFalse(grade(row, '{"label":"S"}')["success"])
        self.assertFalse(grade(row, '{"label":"unknown"}')["json_valid"])


if __name__ == "__main__":
    unittest.main()
