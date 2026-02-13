from __future__ import annotations

import unittest

from agent.__main__ import (
    READ_TEXT_FILE_HARD_CAP,
    classify_truncated_evidence_issue,
    compute_initial_read_max_chars,
    compute_reread_max_chars,
)


class EvidenceReadPolicyTests(unittest.TestCase):
    def test_compute_initial_read_max_chars_uses_config(self) -> None:
        cfg = {"max_chars_full_read": 12345}
        self.assertEqual(compute_initial_read_max_chars(cfg), 12345)

    def test_compute_initial_read_max_chars_clamps_low_and_high(self) -> None:
        self.assertEqual(compute_initial_read_max_chars({"max_chars_full_read": 100}), 200)
        self.assertEqual(
            compute_initial_read_max_chars({"max_chars_full_read": 999999}),
            READ_TEXT_FILE_HARD_CAP,
        )

    def test_compute_reread_max_chars_none_when_not_truncated(self) -> None:
        evidence = {"truncated": False, "chars_full": 1000, "chars_returned": 1000}
        self.assertIsNone(compute_reread_max_chars(evidence, initial_max_chars=50000))

    def test_compute_reread_max_chars_one_reread_when_improvable(self) -> None:
        evidence = {"truncated": True, "chars_full": 120000, "chars_returned": 50000}
        self.assertEqual(compute_reread_max_chars(evidence, initial_max_chars=50000), 120000)

    def test_compute_reread_max_chars_none_when_hard_cap_reached(self) -> None:
        evidence = {"truncated": True, "chars_full": 350000, "chars_returned": 200000}
        self.assertIsNone(
            compute_reread_max_chars(evidence, initial_max_chars=READ_TEXT_FILE_HARD_CAP)
        )

    def test_anomalous_truncated_metadata_classified(self) -> None:
        evidence = {"truncated": True, "chars_full": 1000, "chars_returned": 1000}
        issue = classify_truncated_evidence_issue(evidence, initial_max_chars=500)
        self.assertIsNotNone(issue)
        self.assertIn("Anomalous evidence metadata", issue)

        evidence2 = {"truncated": True, "chars_full": 800, "chars_returned": 600}
        issue2 = classify_truncated_evidence_issue(evidence2, initial_max_chars=1000)
        self.assertIsNotNone(issue2)
        self.assertIn("requested max_chars", issue2)


if __name__ == "__main__":
    unittest.main()
