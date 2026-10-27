"""CPU contracts for the task-domain adapter router."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "aopd_domain_router",
    Path(__file__).resolve().parents[1] / "scripts/aopd_domain_router_workflow.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def prediction(sample_id, domain, digest, text):
    return {
        "sample_id": sample_id,
        "domain": domain,
        "input_sha256": digest,
        "text": text,
    }


class DomainRouterTests(unittest.TestCase):
    def setUp(self):
        self.route = {
            "substitute": "aopd",
            "search": "offline",
            "relevance": "offline",
            "math": "offline",
            "instruction": "offline",
        }

    def test_router_uses_only_declared_task_domain(self):
        sources = {
            "aopd": {
                "a": prediction("a", "substitute", "x", "aopd"),
                "b": prediction("b", "search", "y", "aopd"),
            },
            "offline": {
                "a": prediction("a", "substitute", "x", "offline"),
                "b": prediction("b", "search", "y", "offline"),
            },
        }
        rows = module.compose_rows(sources, self.route)
        self.assertEqual(
            [
                (row["sample_id"], row["text"], row["router_source"])
                for row in rows
            ],
            [("a", "aopd", "aopd"), ("b", "offline", "offline")],
        )

    def test_router_rejects_source_identity_drift(self):
        sources = {
            "aopd": {"a": prediction("a", "substitute", "x", "aopd")},
            "offline": {
                "a": prediction("a", "substitute", "changed", "offline")
            },
        }
        with self.assertRaisesRegex(ValueError, "identity"):
            module.compose_rows(sources, self.route)

    def test_router_requires_every_declared_domain(self):
        sources = {
            "aopd": {"a": prediction("a", "unknown", "x", "aopd")},
            "offline": {"a": prediction("a", "unknown", "x", "offline")},
        }
        with self.assertRaises(KeyError):
            module.compose_rows(sources, self.route)


if __name__ == "__main__":
    unittest.main()
