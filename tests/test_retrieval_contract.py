from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
