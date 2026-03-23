from __future__ import annotations

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
            "aliases:\n"
            "  - atlas-alias\n"
            "tags:\n"
            "  - lookup\n"
            "doc_type: knowledge\n"
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
            "doc_type: journal\n"
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
            "doc_type: journal\n"
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
            "doc_type: knowledge\n"
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

    def test_title_lookup_resolves_to_metadata_chunk(self) -> None:
        app = self._ingested_app()
        rows = app.lexical_query("Atlas Document", limit=5)

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["chunk_kind"]), "metadata")
        self.assertEqual(str(rows[0]["rel_path"]), "atlas.md")
        self.assertEqual(str(rows[0]["lexical_exact_match_field"]), "document_title")

    def test_alias_lookup_resolves_to_metadata_chunk(self) -> None:
        app = self._ingested_app()
        rows = app.lexical_query("atlas-alias", limit=5)

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["chunk_kind"]), "metadata")
        self.assertEqual(str(rows[0]["rel_path"]), "atlas.md")
        self.assertEqual(str(rows[0]["lexical_exact_match_field"]), "aliases")

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
            "journal-old-chunk": {"doc_type": "journal", "entry_date": "2026-03-20", "source_date": ""},
            "journal-new-chunk": {"doc_type": "journal", "entry_date": "2026-03-22", "source_date": ""},
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
            "knowledge-chunk": {"doc_type": "knowledge", "entry_date": "", "source_date": ""},
            "journal-chunk": {"doc_type": "journal", "entry_date": "", "source_date": ""},
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

    def test_bounded_rerank_falls_back_to_source_date_when_entry_date_absent(self) -> None:
        # Recency policy (metadata_v1): source_date is the explicit fallback when
        # entry_date is absent or empty. No filesystem mtime is consulted.
        candidates = [
            self._make_candidate(chunk_key="old-chunk", doc_key="old-doc", rel_path="old.md"),
            self._make_candidate(chunk_key="new-chunk", doc_key="new-doc", rel_path="new.md"),
        ]
        chunk_meta = {
            "old-chunk": {"doc_type": "note", "entry_date": "", "source_date": "2026-03-10"},
            "new-chunk": {"doc_type": "note", "entry_date": "", "source_date": "2026-03-22"},
        }

        reranked, applied, intent, signals_available = _apply_bounded_rerank(
            query="most recent",
            candidates=candidates,
            chunk_meta=chunk_meta,
        )

        self.assertTrue(signals_available)
        self.assertEqual(intent, "recent")
        # Newer source_date wins when entry_date is absent
        self.assertEqual(reranked[0].chunk_key, "new-chunk")

    def test_bounded_rerank_entry_date_preferred_over_source_date_for_recency(self) -> None:
        # Recency policy (metadata_v1): entry_date is the primary date signal and wins
        # over source_date when both are present.
        candidates = [
            self._make_candidate(chunk_key="chunk-a", doc_key="doc-a", rel_path="a.md"),
            self._make_candidate(chunk_key="chunk-b", doc_key="doc-b", rel_path="b.md"),
        ]
        # chunk-a has an older entry_date but newer source_date — entry_date should govern
        # chunk-b has a newer entry_date but older source_date — entry_date should govern
        chunk_meta = {
            "chunk-a": {"doc_type": "note", "entry_date": "2026-03-10", "source_date": "2026-03-22"},
            "chunk-b": {"doc_type": "note", "entry_date": "2026-03-20", "source_date": "2026-03-01"},
        }

        reranked, applied, intent, signals_available = _apply_bounded_rerank(
            query="most recent",
            candidates=candidates,
            chunk_meta=chunk_meta,
        )

        self.assertTrue(signals_available)
        self.assertEqual(intent, "recent")
        # entry_date governs; chunk-b has newer entry_date (2026-03-20) so it comes first
        self.assertEqual(reranked[0].chunk_key, "chunk-b")

    def test_bounded_rerank_does_not_change_results_without_supported_intent(self) -> None:
        candidates = [
            self._make_candidate(chunk_key="knowledge-chunk", doc_key="knowledge-doc", rel_path="knowledge.md"),
            self._make_candidate(chunk_key="journal-chunk", doc_key="journal-doc", rel_path="journal.md"),
        ]
        chunk_meta = {
            "knowledge-chunk": {"doc_type": "knowledge", "entry_date": "", "source_date": ""},
            "journal-chunk": {"doc_type": "journal", "entry_date": "2026-03-22", "source_date": ""},
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


if __name__ == "__main__":
    unittest.main()
