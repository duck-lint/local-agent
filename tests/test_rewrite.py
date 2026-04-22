from __future__ import annotations

import unittest

from agent.rewrite import RewrittenQuery, rule_based_rewrite


class RewriteTests(unittest.TestCase):
    def test_rule_based_rewrite_acronym_expansion(self) -> None:
        result = rule_based_rewrite(
            "what is an API?",
            acronyms={"api": "application programming interface"},
        )
        self.assertEqual(result.original, "what is an API?")
        self.assertIn("application programming interface", result.rewritten)
        self.assertIn("acronym_expansion", result.transforms_applied)
        self.assertEqual(result.acronyms_expanded, [("API", "application programming interface")])
        self.assertFalse(result.is_identity())

    def test_rule_based_rewrite_identity_when_no_match(self) -> None:
        result = rule_based_rewrite(
            "completely unrelated content",
            acronyms={"api": "application programming interface"},
            synonyms={"bug": "defect"},
        )
        self.assertEqual(result.rewritten, result.original)
        self.assertEqual(result.transforms_applied, [])
        self.assertEqual(result.acronyms_expanded, [])
        self.assertEqual(result.synonyms_injected, [])
        self.assertTrue(result.is_identity())

    def test_rule_based_rewrite_no_maps_is_identity(self) -> None:
        result = rule_based_rewrite("anything API")
        self.assertEqual(result.rewritten, "anything API")
        self.assertTrue(result.is_identity())

    def test_rule_based_rewrite_dedups_repeated_token(self) -> None:
        result = rule_based_rewrite(
            "API and another API",
            acronyms={"api": "application programming interface"},
        )
        self.assertEqual(len(result.acronyms_expanded), 1)
        # Suffix appended exactly once.
        self.assertEqual(result.rewritten.count("application programming interface"), 1)

    def test_rule_based_rewrite_acronyms_then_synonyms_order(self) -> None:
        result = rule_based_rewrite(
            "API bug",
            acronyms={"api": "application programming interface"},
            synonyms={"bug": "defect"},
        )
        self.assertEqual(result.transforms_applied, ["acronym_expansion", "synonym_injection"])
        self.assertTrue(
            result.rewritten.endswith("application programming interface defect"),
            f"unexpected suffix order in {result.rewritten!r}",
        )


if __name__ == "__main__":
    unittest.main()
