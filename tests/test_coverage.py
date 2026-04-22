from __future__ import annotations

import unittest

from agent.coverage import CoverageScore, compute_coverage
from agent.retrieval import RetrievedChunk


def _chunk(text: str, *, vector_score: float = 0.0, heading_path: str = "", chunk_title: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_key="ck",
        doc_key="dk",
        chunk_kind="content",
        rel_path="r.md",
        heading_path=heading_path,
        chunk_anchor="",
        chunk_title=chunk_title,
        text=text,
        score=0.0,
        method="lexical",
        lexical_score=0.0,
        vector_score=vector_score,
    )


class CoverageTests(unittest.TestCase):
    def test_coverage_predicate_deterministic(self) -> None:
        chunks = [_chunk("alpha beta gamma", vector_score=0.7)]
        a = compute_coverage("alpha beta", chunks, lexical_threshold=0.5, vector_threshold=0.5)
        b = compute_coverage("alpha beta", chunks, lexical_threshold=0.5, vector_threshold=0.5)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_coverage_with_no_memory_zeros_memory_component(self) -> None:
        chunks = [_chunk("foo bar baz", vector_score=0.9)]
        score = compute_coverage("foo bar", chunks, lexical_threshold=0.5, vector_threshold=0.5)
        self.assertEqual(score.memory_overlap, 0.0)
        self.assertNotIn("memory_provided_but_unused_in_phase_2", score.notes)

    def test_coverage_passes_when_lexical_and_vector_above_thresholds(self) -> None:
        chunks = [_chunk("alpha beta gamma delta", vector_score=0.8)]
        score = compute_coverage(
            "alpha beta",
            chunks,
            lexical_threshold=0.5,
            vector_threshold=0.5,
        )
        self.assertEqual(score.lexical_token_coverage, 1.0)
        self.assertEqual(score.vector_top_score, 0.8)
        self.assertFalse(score.should_refine)

    def test_coverage_fails_when_vector_below_threshold(self) -> None:
        chunks = [_chunk("alpha beta gamma", vector_score=0.1)]
        score = compute_coverage(
            "alpha beta",
            chunks,
            lexical_threshold=0.5,
            vector_threshold=0.5,
        )
        self.assertTrue(score.should_refine)

    def test_coverage_fails_when_lexical_below_threshold(self) -> None:
        chunks = [_chunk("totally unrelated content", vector_score=0.9)]
        score = compute_coverage(
            "alpha beta",
            chunks,
            lexical_threshold=0.5,
            vector_threshold=0.5,
        )
        self.assertEqual(score.lexical_token_coverage, 0.0)
        self.assertTrue(score.should_refine)

    def test_coverage_empty_chunks_triggers_refine(self) -> None:
        score = compute_coverage("anything at all", [], lexical_threshold=0.1, vector_threshold=0.1)
        self.assertEqual(score.vector_top_score, 0.0)
        self.assertEqual(score.lexical_token_coverage, 0.0)
        self.assertTrue(score.should_refine)


if __name__ == "__main__":
    unittest.main()
