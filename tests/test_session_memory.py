from __future__ import annotations

import json
import unittest

from agent.session_memory import (
    ChunkRef,
    FileSessionStore,
    SessionState,
    SessionStoreError,
    TopicTerm,
    compute_state_update,
    compute_topic_summary,
    extract_topic_terms,
    make_bundle_id,
    validate_session_id,
)
from tests.support import AppFixture


class SessionIdValidationTests(unittest.TestCase):
    def test_accepts_alnum_dot_dash_underscore(self) -> None:
        for sid in ("a", "alpha-1", "team_42", "session.A", "x" * 64):
            self.assertEqual(validate_session_id(sid), sid)

    def test_rejects_path_traversal_and_empty(self) -> None:
        for bad in ("", " ", "a/b", "../etc", "x" * 65, "has space", "foo$"):
            with self.assertRaises(ValueError):
                validate_session_id(bad)


class TopicSummaryTests(unittest.TestCase):
    def test_deterministic_top_k(self) -> None:
        s1 = compute_topic_summary(query="alpha beta gamma", answer_text="alpha alpha beta", top_k=3)
        s2 = compute_topic_summary(query="alpha beta gamma", answer_text="alpha alpha beta", top_k=3)
        self.assertEqual(s1, s2)
        self.assertEqual(s1[0], "alpha")  # weighted by query x2 + answer counts
        self.assertIn("beta", s1)

    def test_stopwords_dropped_and_short_tokens(self) -> None:
        out = compute_topic_summary(query="the and is for", answer_text="x y z", top_k=5)
        self.assertEqual(out, [])


class TopicTermExtractionTests(unittest.TestCase):
    """Polarity + source preservation in extract_topic_terms."""

    def test_query_terms_tagged_query_asserted(self) -> None:
        terms = extract_topic_terms(query="alpha bravo", answer_text="")
        by_text = {t.text: t for t in terms}
        self.assertEqual(by_text["alpha"].source, "query")
        self.assertEqual(by_text["alpha"].polarity, "asserted")
        self.assertEqual(by_text["bravo"].source, "query")

    def test_answer_terms_tagged_answer_source(self) -> None:
        terms = extract_topic_terms(query="", answer_text="alpha bravo")
        by_text = {t.text: t for t in terms}
        self.assertEqual(by_text["alpha"].source, "answer")
        self.assertEqual(by_text["bravo"].source, "answer")

    def test_negation_window_marks_polarity(self) -> None:
        # "alpha does not contain bravo": the 3-word window before "contain"
        # is ["alpha","does","not"] -> negated. Window before "bravo" is
        # ["does","not","contain"] -> negated.
        terms = extract_topic_terms(query="", answer_text="alpha does not contain bravo")
        by_text = {t.text: t for t in terms}
        self.assertEqual(by_text["alpha"].polarity, "asserted")
        self.assertEqual(by_text["contain"].polarity, "negated")
        self.assertEqual(by_text["bravo"].polarity, "negated")

    def test_negation_window_only_three_words(self) -> None:
        # "not" is 4 words upstream of "delta" -> outside window -> asserted.
        terms = extract_topic_terms(
            query="", answer_text="alpha not bravo charlie omega delta"
        )
        by_text = {t.text: t for t in terms}
        self.assertEqual(by_text["bravo"].polarity, "negated")
        self.assertEqual(by_text["delta"].polarity, "asserted")


class TopicSummaryRefusalTests(unittest.TestCase):
    """Regression: an INSUFFICIENT_EVIDENCE-style refusal must not pollute
    topic_summary with refusal meta-vocabulary (the bug observed in run
    20260420_175343 where 7 of 8 topic tokens came from the refusal phrasing,
    not from the actual question subject)."""

    REFUSAL_QUERY = "what are the pillars"
    REFUSAL_ANSWER = (
        'The provided evidence does not contain a clear definition or list of '
        '"pillars." The documents are primarily personal journals that discuss '
        'various themes and reflections but do not explicitly outline what the '
        'pillars are in relation to any structured framework or concept. '
        'Therefore, there is insufficient information to accurately describe '
        'the pillars based on the given data.\n\nINSUFFICIENT_EVIDENCE'
    )

    def test_refusal_does_not_pollute_topic_summary(self) -> None:
        out = compute_topic_summary(
            query=self.REFUSAL_QUERY,
            answer_text=self.REFUSAL_ANSWER,
            top_k=8,
        )
        # Real subject must lead.
        self.assertEqual(out[0], "pillars")
        # None of the single-occurrence refusal meta-vocabulary tokens
        # observed in the failure run should survive.
        for polluter in (
            "provided", "contain", "clear", "definition",
            "list", "documents", "evidence",
        ):
            self.assertNotIn(
                polluter, out,
                f"{polluter!r} should not be in topic_summary; got {out}",
            )

    def test_negated_term_excluded_from_summary(self) -> None:
        # "present" appears twice but always negated -> excluded.
        # "alpha" appears twice, asserted (no negation cue in its 3-word
        # upstream window) -> included.
        out = compute_topic_summary(
            query="",
            answer_text="alpha shines bright. alpha rises. it was not present and was not present",
            top_k=5,
        )
        self.assertNotIn("present", out)
        self.assertIn("alpha", out)

    def test_answer_only_single_occurrence_dropped(self) -> None:
        # No query, every answer token appears once -> none survive the
        # count >= 2 threshold for answer-only tokens.
        out = compute_topic_summary(
            query="",
            answer_text="alpha bravo charlie delta echo",
            top_k=5,
        )
        self.assertEqual(out, [])

    def test_query_token_survives_without_repetition(self) -> None:
        # Query tokens bypass the answer-only threshold.
        out = compute_topic_summary(query="alpha", answer_text="bravo", top_k=5)
        self.assertEqual(out, ["alpha"])

    def test_previous_summary_carry_forward_unaffected(self) -> None:
        # Previous-summary tokens are already-curated; threshold doesn't apply.
        out = compute_topic_summary(
            query="alpha",
            answer_text="",
            previous_summary=("bravo", "charlie"),
            top_k=5,
        )
        self.assertIn("alpha", out)
        self.assertIn("bravo", out)
        self.assertIn("charlie", out)


class FileSessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.store = FileSessionStore(self.fx.workroot)

    def tearDown(self) -> None:
        self.fx.close()

    def test_get_returns_empty_for_unknown(self) -> None:
        s = self.store.get("brand-new")
        self.assertEqual(s.session_id, "brand-new")
        self.assertEqual(s.turn_count, 0)
        self.assertEqual(s.topic_summary, [])

    def test_round_trip_save_get(self) -> None:
        state = SessionState(
            session_id="abc",
            topic_summary=["alpha", "beta"],
            active_refs=[ChunkRef(chunk_key="k1", doc_key="d1", rel_path="a.md", heading_path="H")],
            last_evidence_bundle_ids=["bundle:run-1"],
            last_query="hello",
            updated_unix=1.0,
            created_unix=1.0,
            turn_count=1,
        )
        self.store.save(state)
        loaded = self.store.get("abc")
        self.assertEqual(loaded.topic_summary, ["alpha", "beta"])
        self.assertEqual(loaded.active_refs[0].chunk_key, "k1")
        self.assertEqual(loaded.last_evidence_bundle_ids, ["bundle:run-1"])

    def test_list_and_clear(self) -> None:
        self.store.save(SessionState(session_id="alpha"))
        self.store.save(SessionState(session_id="beta"))
        self.assertEqual(self.store.list(), ["alpha", "beta"])
        self.assertTrue(self.store.clear("alpha"))
        self.assertFalse(self.store.clear("alpha"))
        self.assertEqual(self.store.list(), ["beta"])

    def test_save_rejects_bad_id(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save(SessionState(session_id="../escape"))


class StateUpdateTests(unittest.TestCase):
    def test_compute_state_update_advances_turn_and_dedupes_refs(self) -> None:
        prev = SessionState(
            session_id="s",
            topic_summary=["alpha"],
            active_refs=[ChunkRef(chunk_key="k1", doc_key="d1")],
            last_evidence_bundle_ids=["bundle:run-0"],
            turn_count=1,
            created_unix=1.0,
            updated_unix=1.0,
        )
        new = compute_state_update(
            previous=prev,
            query="alpha gamma evidence",
            answer_text="alpha alpha gamma",
            final_chunk_refs=[
                ChunkRef(chunk_key="k1", doc_key="d1"),  # dupe
                ChunkRef(chunk_key="k2", doc_key="d1"),
            ],
            bundle_id="bundle:run-1",
            now_unix=2.0,
        )
        self.assertEqual(new.turn_count, 2)
        self.assertEqual([r.chunk_key for r in new.active_refs[:2]], ["k1", "k2"])
        # k1 only once, even though present in both prev and new.
        chunk_keys = [r.chunk_key for r in new.active_refs]
        self.assertEqual(len(chunk_keys), len(set(chunk_keys)))
        self.assertEqual(new.last_evidence_bundle_ids[:2], ["bundle:run-1", "bundle:run-0"])
        self.assertIn("alpha", new.topic_summary)

    def test_make_bundle_id_is_deterministic(self) -> None:
        b1 = make_bundle_id(run_id="r-1", candidates=[])
        b2 = make_bundle_id(run_id="r-1", candidates=[])
        self.assertEqual(b1, b2)
        self.assertTrue(b1.startswith("bundle:"))


if __name__ == "__main__":
    unittest.main()
