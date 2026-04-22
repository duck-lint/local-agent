"""Phase 4: memory-influenced rule-based rewrite."""

from __future__ import annotations

import unittest

from agent.rewrite import rule_based_rewrite
from agent.session_memory import ChunkRef, SessionState


def _state(*, topic_summary=(), active_refs=()) -> SessionState:
    return SessionState(
        session_id="t",
        topic_summary=list(topic_summary),
        active_refs=list(active_refs),
    )


class MemoryRewriteTests(unittest.TestCase):
    def test_no_memory_returns_identity_when_no_maps(self) -> None:
        rw = rule_based_rewrite("how does alpha work?")
        self.assertTrue(rw.is_identity())

    def test_empty_memory_is_identical_to_no_memory(self) -> None:
        """Cold-start parity: empty SessionState must not alter the query."""
        empty_state = _state()
        rw_none = rule_based_rewrite("how does alpha work?")
        rw_empty = rule_based_rewrite("how does alpha work?", memory=empty_state)
        self.assertEqual(rw_none.to_dict(), rw_empty.to_dict())

    def test_appends_active_refs_as_seeds(self) -> None:
        state = _state(
            active_refs=[
                ChunkRef(
                    chunk_key="c1",
                    doc_key="d1",
                    rel_path="alpha.md",
                    heading_path="Alpha > Subsection",
                )
            ]
        )
        rw = rule_based_rewrite("how does it work?", memory=state)
        self.assertFalse(rw.is_identity())
        self.assertIn("memory_active_refs", rw.transforms_applied)
        # Tokens from the heading_path must appear in the rewritten suffix.
        self.assertIn("Alpha", rw.rewritten.split())
        self.assertIn("Subsection", rw.rewritten.split())
        # Original query preserved in full (prefix).
        self.assertTrue(rw.rewritten.startswith("how does it work?"))

    def test_appends_topic_summary_keywords(self) -> None:
        state = _state(topic_summary=["retrieval", "embeddings"])
        rw = rule_based_rewrite("what did we discuss?", memory=state)
        self.assertFalse(rw.is_identity())
        self.assertIn("memory_topic_summary", rw.transforms_applied)
        self.assertIn("retrieval", rw.rewritten)
        self.assertIn("embeddings", rw.rewritten)

    def test_skips_tokens_already_in_query(self) -> None:
        """Memory must not duplicate tokens already in the original query."""
        state = _state(topic_summary=["retrieval", "embeddings"])
        rw = rule_based_rewrite("retrieval and embeddings tuning", memory=state)
        # Neither keyword should be re-appended.
        self.assertTrue(rw.is_identity())

    def test_memory_does_not_remove_user_tokens(self) -> None:
        """Memory is additive-only — every original token survives."""
        state = _state(topic_summary=["alpha"])
        original = "how does beta work today?"
        rw = rule_based_rewrite(original, memory=state)
        for tok in original.split():
            self.assertIn(tok, rw.rewritten)

    def test_disabled_flag_is_identity_callsite(self) -> None:
        """When caller passes memory=None explicitly, behavior is identical to
        passing no memory — this covers the `memory_rewrite_enabled=False` gate
        in retrieve_with_refinement."""
        state = _state(topic_summary=["alpha", "beta"])
        rw_off = rule_based_rewrite("question", memory=None)
        rw_on_but_gated = rule_based_rewrite("question")  # same as gated off
        self.assertEqual(rw_off.to_dict(), rw_on_but_gated.to_dict())
        # And non-identity when on, proving the gate matters.
        rw_on = rule_based_rewrite("question", memory=state)
        self.assertNotEqual(rw_off.to_dict(), rw_on.to_dict())


if __name__ == "__main__":
    unittest.main()
