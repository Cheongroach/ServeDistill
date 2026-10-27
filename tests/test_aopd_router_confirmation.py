"""CPU contracts for composing the frozen route-aware confirmation."""
import unittest

from scripts.analyze_aopd_router_confirmation import compose_candidate


def row(identity, domain, digest, text):
    return {
        "sample_id": identity,
        "domain": domain,
        "input_sha256": digest,
        "text": text,
    }


class RouterConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.route = {
            "substitute": "aopd",
            "search": "offline_sft",
            "relevance": "offline_sft",
            "math": "offline_sft",
        }

    def test_candidate_uses_aopd_only_for_substitute(self):
        offline = [
            row("s", "substitute", "x", "offline"),
            row("q", "search", "y", "offline"),
        ]
        aopd = [row("s", "substitute", "x", "aopd")]
        candidate = compose_candidate(aopd, offline, self.route)
        self.assertEqual(
            [(value["sample_id"], value["text"]) for value in candidate],
            [("s", "aopd"), ("q", "offline")],
        )

    def test_candidate_rejects_missing_substitute_prediction(self):
        offline = [row("s", "substitute", "x", "offline")]
        with self.assertRaisesRegex(ValueError, "coverage"):
            compose_candidate([], offline, self.route)

    def test_candidate_rejects_route_revision(self):
        with self.assertRaisesRegex(ValueError, "changed"):
            compose_candidate([], [], {**self.route, "math": "aopd"})


if __name__ == "__main__":
    unittest.main()
