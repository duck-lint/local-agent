from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
import unittest
from unittest.mock import patch

from agent.corpus_db import _query_chunk_search_fallback
from agent.embeddings import sync_embeddings
from agent.retrieval import RetrievedChunk, _apply_bounded_rerank
from tests.support import AppFixture, dummy_embedder_factory


class RetrievalContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note(
            "atlas.md",
            "---\n"
            "uuid: atlas-doc\n"
            "title: Atlas Document\n"
            "canonical_name: atlas-canonical\n"
            "aliases:\n"
            "  - atlas-alias\n"
            "tags:\n"
            "  - lookup\n"
            "note_type: atlas-note\n"
            "layer: reference-layer\n"
            "register: engineering-register\n"
            "journal_entry_date: 2026-03-19\n"
            "---\n"
            "\n"
            "## Alpha Heading\n"
            "body-only-marker-qz7 lives in the atlas body.\n",
        )
        self.fx.write_corpus_note(
            "journal-old.md",
            "---\n"
            "uuid: journal-old\n"
            "title: Sprint Review Old\n"
            "note_type: journal\n"
            "journal_entry_date: 2026-03-20\n"
            "---\n"
            "\n"
            "## Entry\n"
            "journal planning note old.\n",
        )
        self.fx.write_corpus_note(
            "journal-new.md",
            "---\n"
            "uuid: journal-new\n"
            "title: Sprint Review New\n"
            "note_type: journal\n"
            "journal_entry_date: 2026-03-22\n"
            "---\n"
            "\n"
            "## Entry\n"
            "journal planning note new.\n",
        )
        self.fx.write_corpus_note(
            "journal-title-confounder.md",
            "---\n"
            "uuid: journal-title-confounder\n"
            "title: Journal\n"
            "note_type: knowledge\n"
            "---\n"
            "\n"
            "## Entry\n"
            "knowledge note that happens to say journal.\n",
        )

    def tearDown(self) -> None:
        self.fx.close()

    def _ingested_app(self):
        app = self.fx.build_app()
        ingest = app.ingest_corpus()
        self.assertEqual(ingest.errors, [])
        return app

    def _embedded_app(self):
        app = self._ingested_app()
        embed = sync_embeddings(
            app_config=app.config,
            security_root=app.roots.security_root,
            corpus_db_path=app.corpus_db_path(),
            embedder_factory=dummy_embedder_factory,
        )
        self.assertEqual(embed.errors, [])
        return app

    def _make_candidate(self, *, chunk_key: str, doc_key: str, rel_path: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_key=chunk_key,
            doc_key=doc_key,
            chunk_kind="content",
            rel_path=rel_path,
            heading_path="H2: Entry",
            chunk_anchor="entry",
            chunk_title="Entry",
            text="placeholder text",
            score=1.0,
            method="both",
            lexical_score=1.0,
            vector_score=1.0,
        )

    def test_locked_metadata_fields_are_searchable_via_metadata_chunk(self) -> None:
        app = self._ingested_app()
        expectations = {
            "atlas-canonical": "canonical_name",
            "atlas-alias": "aliases",
            "lookup": "tags",
            "atlas-note": "note_type",
            "2026-03-19": "journal_entry_date",
            "reference-layer": "layer",
            "engineering-register": "register",
        }

        for query, expected_field in expectations.items():
            with self.subTest(query=query):
                rows = app.lexical_query(query, limit=5)
                self.assertGreaterEqual(len(rows), 1)
                self.assertEqual(str(rows[0]["chunk_kind"]), "metadata")
                self.assertEqual(str(rows[0]["rel_path"]), "atlas.md")
                self.assertEqual(str(rows[0]["lexical_exact_match_field"]), expected_field)

    def test_heading_lookup_resolves_to_content_chunk(self) -> None:
        app = self._ingested_app()
        rows = app.lexical_query("H2: Alpha Heading", limit=5)

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["chunk_kind"]), "content")
        self.assertEqual(str(rows[0]["rel_path"]), "atlas.md")
        self.assertEqual(str(rows[0]["lexical_exact_match_field"]), "heading_path")

    def test_body_phrase_lookup_resolves_to_content_chunk(self) -> None:
        app = self._ingested_app()
        rows = app.lexical_query("body-only-marker-qz7", limit=5)

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["chunk_kind"]), "content")
        self.assertEqual(str(rows[0]["rel_path"]), "atlas.md")
        self.assertIn("body-only-marker-qz7", str(rows[0]["chunk_text"]))

    def test_projection_substring_fallback_is_visible_in_retrieval_and_doctor(self) -> None:
        with patch("agent.corpus_db._ensure_chunk_search_fts", return_value=False):
            app = self._ingested_app()

        rows = app.lexical_query("atlas-alias", limit=5)
        report = app.doctor(check_ollama=False, require_grounding=False)

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["lexical_backend_mode"]), "projection_substring")
        self.assertEqual(str(report.summary["lexical_backend_mode"]), "projection_substring")
        self.assertTrue(str(report.summary["lexical_backend_warning"]))
        self.assertTrue(any(check.code == "DOCTOR_LEXICAL_BACKEND_FALLBACK" for check in report.checks))

    def test_retrieve_reports_journal_rerank_diagnostics(self) -> None:
        app = self._embedded_app()

        with patch("agent.app.create_embedder", side_effect=dummy_embedder_factory):
            result = app.retrieve("journal")

        self.assertTrue(result.rerank_applied)
        self.assertEqual(result.rerank_intent, "journal")
        self.assertTrue(result.rerank_signals_available)
        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertIn(result.candidates[0].rel_path, {"journal-old.md", "journal-new.md"})
        self.assertIn(result.lexical_backend_mode, {"fts5", "projection_substring"})

    def test_bounded_rerank_prefers_newer_journal_docs_for_recent_intent(self) -> None:
        candidates = [
            self._make_candidate(chunk_key="journal-old-chunk", doc_key="journal-old", rel_path="journal-old.md"),
            self._make_candidate(chunk_key="journal-new-chunk", doc_key="journal-new", rel_path="journal-new.md"),
        ]
        chunk_meta = {
            "journal-old-chunk": {"note_type": "journal", "journal_entry_date": "2026-03-20", "mtime": 0.0},
            "journal-new-chunk": {"note_type": "journal", "journal_entry_date": "2026-03-22", "mtime": 0.0},
        }

        reranked, applied, intent, signals_available = _apply_bounded_rerank(
            query="latest journal entry",
            candidates=candidates,
            chunk_meta=chunk_meta,
        )

        self.assertTrue(applied)
        self.assertEqual(intent, "journal_recent")
        self.assertTrue(signals_available)
        self.assertEqual(reranked[0].chunk_key, "journal-new-chunk")

    def test_bounded_rerank_prefers_journal_class_for_explicit_class_intent(self) -> None:
        candidates = [
            self._make_candidate(chunk_key="knowledge-chunk", doc_key="knowledge-doc", rel_path="knowledge.md"),
            self._make_candidate(chunk_key="journal-chunk", doc_key="journal-doc", rel_path="journal.md"),
        ]
        chunk_meta = {
            "knowledge-chunk": {"note_type": "knowledge", "journal_entry_date": "", "mtime": 0.0},
            "journal-chunk": {"note_type": "journal", "journal_entry_date": "", "mtime": 0.0},
        }

        reranked, applied, intent, signals_available = _apply_bounded_rerank(
            query="journal entries",
            candidates=candidates,
            chunk_meta=chunk_meta,
        )

        self.assertTrue(applied)
        self.assertEqual(intent, "journal")
        self.assertTrue(signals_available)
        self.assertEqual(reranked[0].chunk_key, "journal-chunk")

    def test_bounded_rerank_does_not_change_results_without_supported_intent(self) -> None:
        candidates = [
            self._make_candidate(chunk_key="knowledge-chunk", doc_key="knowledge-doc", rel_path="knowledge.md"),
            self._make_candidate(chunk_key="journal-chunk", doc_key="journal-doc", rel_path="journal.md"),
        ]
        chunk_meta = {
            "knowledge-chunk": {"note_type": "knowledge", "journal_entry_date": "", "mtime": 0.0},
            "journal-chunk": {"note_type": "journal", "journal_entry_date": "2026-03-22", "mtime": 0.0},
        }

        reranked, applied, intent, signals_available = _apply_bounded_rerank(
            query="project checkpoint",
            candidates=candidates,
            chunk_meta=chunk_meta,
        )

        self.assertFalse(applied)
        self.assertEqual(intent, "")
        self.assertFalse(signals_available)
        self.assertEqual([item.chunk_key for item in reranked], [item.chunk_key for item in candidates])

    def test_bounded_rerank_falls_back_to_mtime_when_journal_entry_date_is_missing(self) -> None:
        candidates = [
            self._make_candidate(chunk_key="older-journal-date", doc_key="older-date", rel_path="older-date.md"),
            self._make_candidate(chunk_key="newer-mtime", doc_key="newer-mtime", rel_path="newer-mtime.md"),
        ]
        chunk_meta = {
            "older-journal-date": {
                "note_type": "knowledge",
                "journal_entry_date": "2026-03-20",
                "mtime": datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp(),
                "source_date": "2027-01-01",
            },
            "newer-mtime": {
                "note_type": "knowledge",
                "journal_entry_date": "",
                "mtime": datetime(2026, 3, 21, tzinfo=timezone.utc).timestamp(),
                "source_date": "2000-01-01",
            },
        }

        reranked, applied, intent, signals_available = _apply_bounded_rerank(
            query="latest notes",
            candidates=candidates,
            chunk_meta=chunk_meta,
        )

        self.assertTrue(applied)
        self.assertEqual(intent, "recent")
        self.assertTrue(signals_available)
        self.assertEqual(reranked[0].chunk_key, "newer-mtime")


class FusionStrategyTests(unittest.TestCase):
    """Unit tests for _fuse_candidates: simple_union vs rrf."""

    def _meta(self, key: str) -> dict[str, object]:
        return {
            "doc_key": f"doc-{key}",
            "chunk_kind": "content",
            "rel_path": f"{key}.md",
            "heading_path": "",
            "chunk_anchor": "",
            "chunk_title": key.title(),
            "text": f"text for {key}",
            "note_type": "",
            "journal_entry_date": "",
            "mtime": 0.0,
        }

    def test_simple_union_orders_by_average_score_with_both_first(self) -> None:
        from agent.retrieval import _fuse_candidates

        # A: lex only (high). B: vec only (high). C: in both (mid+mid).
        lexical_ranked = {"A": 1.0, "C": 0.6}
        vector_ranked = {"B": 1.0, "C": 0.7}
        lexical_meta = {k: self._meta(k) for k in ("A", "B", "C")}

        with patch("agent.retrieval._fetch_chunk_metadata", return_value={}):
            out = _fuse_candidates(
                corpus_db_path=__import__("pathlib").Path("/dev/null"),
                lexical_ranked=lexical_ranked,
                lexical_meta=lexical_meta,
                vector_ranked=vector_ranked,
                lexical_ranks={"A": 1, "C": 2},
                vector_ranks={"B": 1, "C": 2},
                strategy="simple_union",
            )

        # "both" sorts ahead of single-source regardless of score.
        self.assertEqual(out[0].chunk_key, "C")
        self.assertEqual(out[0].method, "both")
        self.assertAlmostEqual(out[0].score, (0.6 + 0.7) / 2.0)
        single = {c.chunk_key: c for c in out[1:]}
        self.assertEqual(set(single.keys()), {"A", "B"})
        self.assertEqual(single["A"].method, "lexical")
        self.assertAlmostEqual(single["A"].score, 1.0)
        self.assertEqual(single["B"].method, "vector")
        self.assertAlmostEqual(single["B"].score, 1.0)

    def test_rrf_uses_rank_positions_not_raw_scores(self) -> None:
        from agent.retrieval import _fuse_candidates

        # A is rank-1 in BOTH lists, even though its mock scores are lower.
        # B is only in lex (rank 2). C is only in vec (rank 2).
        lexical_ranked = {"A": 0.1, "B": 0.05}
        vector_ranked = {"A": 0.1, "C": 0.05}
        lexical_ranks = {"A": 1, "B": 2}
        vector_ranks = {"A": 1, "C": 2}
        lexical_meta = {k: self._meta(k) for k in ("A", "B", "C")}

        with patch("agent.retrieval._fetch_chunk_metadata", return_value={}):
            out = _fuse_candidates(
                corpus_db_path=__import__("pathlib").Path("/dev/null"),
                lexical_ranked=lexical_ranked,
                lexical_meta=lexical_meta,
                vector_ranked=vector_ranked,
                lexical_ranks=lexical_ranks,
                vector_ranks=vector_ranks,
                strategy="rrf",
                rrf_k=60,
            )

        scores = {c.chunk_key: c.score for c in out}
        # A appears in both -> 2 * 1/(60+1) = 2/61
        self.assertAlmostEqual(scores["A"], 2.0 / 61.0)
        # B only in lex at rank 2 -> 1/(60+2) = 1/62
        self.assertAlmostEqual(scores["B"], 1.0 / 62.0)
        # C only in vec at rank 2 -> 1/62
        self.assertAlmostEqual(scores["C"], 1.0 / 62.0)
        # Top by sort key (method=='both' first, then -score) is A.
        self.assertEqual(out[0].chunk_key, "A")
        self.assertEqual(out[0].method, "both")

    def test_unsupported_strategy_raises(self) -> None:
        from agent.retrieval import _fuse_candidates

        with self.assertRaises(ValueError):
            _fuse_candidates(
                corpus_db_path=__import__("pathlib").Path("/dev/null"),
                lexical_ranked={},
                lexical_meta={},
                vector_ranked={},
                strategy="rerank",
            )


class FallbackLimitRegressionTests(unittest.TestCase):
    """Regression tests asserting that _query_chunk_search_fallback respects its LIMIT."""

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = OFF")  # no documents/sources tables in this minimal fixture
        conn.execute(
            """
            CREATE TABLE chunk_search (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                chunk_key TEXT NOT NULL UNIQUE,
                chunk_kind TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                body_text TEXT NOT NULL,
                chunk_title TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                aliases_text TEXT NOT NULL,
                tags_text TEXT NOT NULL,
                note_type TEXT NOT NULL,
                journal_entry_date TEXT,
                layer TEXT NOT NULL,
                register TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        return conn

    def _insert_rows(self, conn: sqlite3.Connection, count: int, *, query_word: str) -> None:
        for i in range(count):
            conn.execute(
                """
                INSERT INTO chunk_search
                    (doc_id, chunk_key, chunk_kind, rel_path, body_text,
                     chunk_title, heading_path, canonical_name, aliases_text,
                     tags_text, note_type, journal_entry_date, layer, register, updated_at)
                VALUES (?, ?, 'content', ?, ?, ?, '', '', '', '', 'knowledge', NULL, '', '', 0.0)
                """,
                (
                    i,
                    f"chunk-{i}",
                    f"doc-{i}.md",
                    f"{query_word} body text for row {i}",
                    f"{query_word} title {i}",
                ),
            )
        conn.commit()

    def test_fallback_candidate_count_is_capped_at_limit(self) -> None:
        conn = self._make_conn()
        total_rows = 200
        limit = 10
        self._insert_rows(conn, total_rows, query_word="common")

        results = _query_chunk_search_fallback(conn, query_text="common", limit=limit)

        self.assertLessEqual(
            len(results),
            limit,
            f"Expected at most {limit} rows but got {len(results)}",
        )

    def test_fallback_returns_all_matching_when_below_limit(self) -> None:
        conn = self._make_conn()
        self._insert_rows(conn, 5, query_word="raretoken")

        results = _query_chunk_search_fallback(conn, query_text="raretoken", limit=50)

        self.assertEqual(len(results), 5)

    def test_fallback_rows_include_backend_score(self) -> None:
        conn = self._make_conn()
        self._insert_rows(conn, 3, query_word="scorecheck")

        results = _query_chunk_search_fallback(conn, query_text="scorecheck", limit=10)

        for row in results:
            self.assertIn("backend_score", row)
            self.assertIsInstance(row["backend_score"], float)


if __name__ == "__main__":
    unittest.main()
