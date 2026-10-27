import unittest

import torch

from servedistill.data import business_oracle, grade, make_business
from servedistill.losses import (
    DevWeightController, asymmetric_opd_loss, full_reverse_kl,
    grpo_advantages, opd_advantages, policy_loss, reduce_actions,
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

    def test_aopd_intervenes_with_teacher_topk_forward_kl(self):
        student = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
        teacher = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)
        current = student.log_softmax(-1)[..., 1]
        behavior = current.detach().clone()
        teacher_sampled = teacher.log_softmax(-1)[..., 1]
        mask = torch.ones_like(current)
        loss, metrics = asymmetric_opd_loss(
            current, behavior, teacher_sampled, student, teacher, mask, top_k=2)
        teacher_logp = teacher.detach().log_softmax(-1)
        indices = teacher_logp.topk(2, dim=-1).indices
        expected = (
            teacher_logp.gather(-1, indices).exp()
            * (teacher_logp.gather(-1, indices)
               - student.log_softmax(-1).gather(-1, indices))
        ).sum() / 2
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics["aopd_intervention_fraction"], 1.0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_aopd_keeps_positive_sampled_token_opd(self):
        student = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
        teacher = torch.tensor([[[2.0, 0.0, -1.0]]])
        current = student.log_softmax(-1)[..., 0]
        behavior = current.detach().clone()
        teacher_sampled = teacher.log_softmax(-1)[..., 0]
        mask = torch.ones_like(current)
        loss, metrics = asymmetric_opd_loss(
            current, behavior, teacher_sampled, student, teacher, mask, top_k=2)
        expected, _ = policy_loss(
            current, behavior, teacher_sampled - behavior, mask)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(metrics["aopd_intervention_fraction"], 0.0)

    def test_aopd_zero_probability_difference_intervenes(self):
        logits = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        sampled = logits.log_softmax(-1)[..., 0]
        _, metrics = asymmetric_opd_loss(
            sampled, sampled.detach(), sampled.detach(), logits, logits.detach(),
            torch.ones_like(sampled), top_k=2, tau=0.0)
        self.assertEqual(metrics["aopd_intervention_fraction"], 1.0)

    def test_aopd_rejects_topk_larger_than_vocabulary(self):
        logits = torch.zeros(1, 1, 2)
        sampled = logits.log_softmax(-1)[..., 0]
        with self.assertRaises(ValueError):
            asymmetric_opd_loss(
                sampled, sampled, sampled, logits, logits,
                torch.ones_like(sampled), top_k=3)

    def test_aopd_sparse_guidance_matches_dense_formula_and_gradient(self):
        torch.manual_seed(7)
        student = torch.randn(2, 3, 5, requires_grad=True)
        teacher = torch.randn(2, 3, 5)
        sampled_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        current = student.log_softmax(-1).gather(
            -1, sampled_ids[..., None]).squeeze(-1)
        teacher_sampled = teacher.log_softmax(-1).gather(
            -1, sampled_ids[..., None]).squeeze(-1)
        offsets = torch.tensor([[0.2, -0.2, 0.2], [-0.2, 0.2, -0.2]])
        behavior = teacher_sampled + offsets

        actual, _ = asymmetric_opd_loss(
            current, behavior, teacher_sampled, student, teacher, mask,
            top_k=3, aggregation="trajectory",
            current_log_normalizer=student.logsumexp(-1),
            teacher_log_normalizer=teacher.logsumexp(-1),
        )
        signal = opd_advantages(behavior, teacher_sampled, mask)
        intervention = (
            teacher_sampled.exp() - behavior.exp() <= 0
        ) & mask
        positive, _ = policy_loss(
            current, behavior, signal * (~intervention), mask,
            aggregation="trajectory",
        )
        teacher_log_distribution = teacher.log_softmax(-1)
        teacher_top_logp, indices = teacher_log_distribution.topk(3, dim=-1)
        student_top_logp = student.log_softmax(-1).gather(-1, indices)
        dense_kl = (
            teacher_top_logp.exp() * (teacher_top_logp - student_top_logp)
        ).sum(-1) / 3
        expected = positive + reduce_actions(
            dense_kl * intervention, mask, "trajectory")
        actual_gradient = torch.autograd.grad(actual, student, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, student)[0]

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)

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
