from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.corpus import sync_corpus
from agent.corpus_db import connect_db
from tests.support import AppFixture


class CorpusContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note(
            "typed.md",
            "---\n"
            "uuid: typed-doc\n"
            "doc_type: knowledge\n"
            "sensitivity: internal\n"
            "journal_entry_date: 2026-03-20\n"
            "note_creation_date: 2026-03-10T12:00:00\n"
            "---\n"
            "\n"
            "## Alpha Section\n"
            "Alpha paragraph with a [[Target Note|nice alias]].\n\n"
            "## Beta Section\n"
            "Beta paragraph.\n",
        )

    def tearDown(self) -> None:
        self.fx.close()

    def test_corpus_sync_persists_document_and_chunk_contract(self) -> None:
        result = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=self.fx.app_config.corpus,
            force_rebuild=False,
        )
        self.assertEqual(result.errors, [])
        self.assertGreaterEqual(result.total_chunks, 3)

        with connect_db(self.fx.build_app().corpus_db_path()) as conn:
            doc = conn.execute(
                """
                SELECT doc_key, rel_path, source_uri, title, folder, doc_type, sensitivity,
                       entry_date, source_date, frontmatter_json
                FROM documents
                WHERE rel_path = 'typed.md'
                """
            ).fetchone()
            self.assertIsNotNone(doc)
            self.assertEqual(str(doc["doc_key"]), "typed-doc")
            self.assertEqual(str(doc["source_uri"]), "typed.md")
            self.assertEqual(str(doc["doc_type"]), "knowledge")
            self.assertEqual(str(doc["sensitivity"]), "internal")
            self.assertEqual(str(doc["entry_date"]), "2026-03-20")
            self.assertEqual(str(doc["source_date"]), "2026-03-10")
            self.assertIn("uuid", str(doc["frontmatter_json"]))

            metadata_chunk = conn.execute(
                """
                SELECT chunk_key, chunk_kind, heading_path, chunk_anchor, chunk_title, text
                FROM chunks
                WHERE chunk_kind = 'metadata'
                LIMIT 1
                """
            ).fetchone()
            self.assertIsNotNone(metadata_chunk)
            self.assertEqual(str(metadata_chunk["chunk_kind"]), "metadata")
            self.assertEqual(str(metadata_chunk["heading_path"]), "META: frontmatter")
            self.assertEqual(str(metadata_chunk["chunk_anchor"]), "frontmatter")
            self.assertEqual(str(metadata_chunk["chunk_title"]), "frontmatter")
            self.assertIn("title: Alpha Section", str(metadata_chunk["text"]))
            self.assertIn("doc_type: knowledge", str(metadata_chunk["text"]))

            content_chunk = conn.execute(
                """
                SELECT chunk_key, chunk_kind, doc_key, section_index, heading_path, chunk_anchor, chunk_title, out_links_json
                FROM chunks
                WHERE chunk_kind = 'content'
                ORDER BY chunk_index
                LIMIT 1
                """
            ).fetchone()
            self.assertIsNotNone(content_chunk)
            self.assertEqual(len(str(content_chunk["chunk_key"])), 32)
            self.assertEqual(str(content_chunk["chunk_kind"]), "content")
            self.assertEqual(str(content_chunk["doc_key"]), "typed-doc")
            self.assertTrue(str(content_chunk["heading_path"]).startswith("H2: "))
            self.assertTrue(str(content_chunk["chunk_anchor"]))
            self.assertTrue(str(content_chunk["chunk_title"]))
            out_links = json.loads(str(content_chunk["out_links_json"]))
            self.assertIsInstance(out_links, list)

    def test_corpus_sync_is_stable_on_second_run(self) -> None:
        first = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=self.fx.app_config.corpus,
            force_rebuild=False,
        )
        second = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=self.fx.app_config.corpus,
            force_rebuild=False,
        )
        self.assertEqual(first.errors, [])
        self.assertEqual(second.errors, [])
        self.assertEqual(second.docs_changed, 0)
        self.assertEqual(second.docs_unchanged, 1)

    def test_frontmatter_only_note_yields_only_metadata_chunk(self) -> None:
        self.fx.write_corpus_note(
            "frontmatter-only.md",
            "---\n"
            "uuid: frontmatter-doc\n"
            "title: Metadata Only\n"
            "aliases:\n"
            "  - meta-only\n"
            "tags:\n"
            "  - retrieval\n"
            "---\n",
        )

        result = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=self.fx.app_config.corpus,
            force_rebuild=False,
        )
        self.assertEqual(result.errors, [])

        with connect_db(self.fx.build_app().corpus_db_path()) as conn:
            rows = conn.execute(
                """
                SELECT chunk_kind, chunk_index, heading_path, text
                FROM chunks
                INNER JOIN documents ON documents.id = chunks.doc_id
                WHERE documents.rel_path = 'frontmatter-only.md'
                ORDER BY chunk_index
                """
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["chunk_kind"]), "metadata")
        self.assertEqual(int(rows[0]["chunk_index"]), -1)
        self.assertEqual(str(rows[0]["heading_path"]), "META: frontmatter")
        self.assertIn("aliases: meta-only", str(rows[0]["text"]))

    def test_metadata_chunk_is_stable_across_body_transitions(self) -> None:
        app = self.fx.build_app()
        first = app.ingest_corpus()
        self.assertEqual(first.errors, [])

        with connect_db(app.corpus_db_path()) as conn:
            first_rows = conn.execute(
                """
                SELECT chunk_key, chunk_kind
                FROM chunks
                INNER JOIN documents ON documents.id = chunks.doc_id
                WHERE documents.rel_path = 'typed.md'
                ORDER BY chunk_index
                """
            ).fetchall()
            doc_row = conn.execute(
                "SELECT doc_key FROM documents WHERE rel_path = 'typed.md'"
            ).fetchone()
        self.assertIsNotNone(doc_row)
        first_doc_key = str(doc_row["doc_key"])
        first_metadata_key = next(str(row["chunk_key"]) for row in first_rows if str(row["chunk_kind"]) == "metadata")
        first_content_keys = {str(row["chunk_key"]) for row in first_rows if str(row["chunk_kind"]) == "content"}
        self.assertTrue(first_content_keys)

        self.fx.write_corpus_note(
            "typed.md",
            "---\n"
            "uuid: typed-doc\n"
            "doc_type: knowledge\n"
            "aliases:\n"
            "  - typed-alias\n"
            "---\n",
        )
        second = app.ingest_corpus()
        self.assertEqual(second.errors, [])

        with connect_db(app.corpus_db_path()) as conn:
            second_rows = conn.execute(
                """
                SELECT chunk_key, chunk_kind
                FROM chunks
                INNER JOIN documents ON documents.id = chunks.doc_id
                WHERE documents.rel_path = 'typed.md'
                ORDER BY chunk_index
                """
            ).fetchall()
            second_doc_row = conn.execute(
                "SELECT doc_key FROM documents WHERE rel_path = 'typed.md'"
            ).fetchone()
        self.assertIsNotNone(second_doc_row)
        self.assertEqual(str(second_doc_row["doc_key"]), first_doc_key)
        second_metadata_key = next(str(row["chunk_key"]) for row in second_rows if str(row["chunk_kind"]) == "metadata")
        second_content_keys = {str(row["chunk_key"]) for row in second_rows if str(row["chunk_kind"]) == "content"}
        self.assertEqual(second_metadata_key, first_metadata_key)
        self.assertEqual(second_content_keys, set())

        self.fx.write_corpus_note(
            "typed.md",
            "---\n"
            "uuid: typed-doc\n"
            "doc_type: knowledge\n"
            "---\n"
            "\n"
            "## Gamma Section\n"
            "Gamma paragraph.\n",
        )
        third = app.ingest_corpus()
        self.assertEqual(third.errors, [])
        with connect_db(app.corpus_db_path()) as conn:
            third_rows = conn.execute(
                """
                SELECT chunk_key, chunk_kind
                FROM chunks
                INNER JOIN documents ON documents.id = chunks.doc_id
                WHERE documents.rel_path = 'typed.md'
                ORDER BY chunk_index
                """
            ).fetchall()
        third_metadata_key = next(str(row["chunk_key"]) for row in third_rows if str(row["chunk_kind"]) == "metadata")
        third_content_keys = {str(row["chunk_key"]) for row in third_rows if str(row["chunk_kind"]) == "content"}
        self.assertEqual(third_metadata_key, first_metadata_key)
        self.assertTrue(third_content_keys)
        self.assertNotEqual(third_content_keys, first_content_keys)

    def test_duplicate_cross_source_fallback_identity_fails_with_operator_guidance(self) -> None:
        self.fx.write_corpus_note("shared.md", "# Corpus copy\n")
        scratch_note = self.fx.workroot / "allowed" / "scratch" / "shared.md"
        scratch_note.parent.mkdir(parents=True, exist_ok=True)
        scratch_note.write_text("# Scratch copy\n", encoding="utf-8")

        result = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=self.fx.app_config.corpus,
            force_rebuild=False,
        )

        self.assertEqual(len(result.errors), 1)
        error = result.errors[0]
        self.assertIn("DUPLICATE_DOCUMENT_IDENTITY", error)
        self.assertIn("scratch:shared.md", error)
        self.assertIn("corpus:shared.md", error)
        self.assertIn("fallback doc_key", error)
        self.assertIn("Cross-source duplicate fallback identities are invalid corpus input", error)
        self.assertIn("Add explicit uuid frontmatter", error)
        self.assertIn("rename one note", error)

    def test_explicit_uuids_disambiguate_cross_source_duplicate_rel_paths(self) -> None:
        self.fx.write_corpus_note("shared.md", "---\nuuid: corpus-shared\n---\n# Corpus copy\n")
        scratch_note = self.fx.workroot / "allowed" / "scratch" / "shared.md"
        scratch_note.parent.mkdir(parents=True, exist_ok=True)
        scratch_note.write_text("---\nuuid: scratch-shared\n---\n# Scratch copy\n", encoding="utf-8")

        result = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=self.fx.app_config.corpus,
            force_rebuild=False,
        )

        self.assertEqual(result.errors, [])
        with connect_db(self.fx.build_app().corpus_db_path()) as conn:
            rows = conn.execute(
                """
                SELECT sources.name AS source_name, documents.rel_path, documents.doc_key
                FROM documents
                INNER JOIN sources ON sources.id = documents.source_id
                WHERE documents.rel_path = 'shared.md'
                ORDER BY sources.name
                """
            ).fetchall()
            chunk_keys = conn.execute(
                """
                SELECT chunks.chunk_key
                FROM chunks
                INNER JOIN documents ON documents.id = chunks.doc_id
                WHERE documents.rel_path = 'shared.md' AND chunks.chunk_kind = 'metadata'
                ORDER BY chunks.chunk_key
                """
            ).fetchall()
        self.assertEqual(
            [(str(row["source_name"]), str(row["rel_path"]), str(row["doc_key"])) for row in rows],
            [
                ("corpus", "shared.md", "corpus-shared"),
                ("scratch", "shared.md", "scratch-shared"),
            ],
        )
        self.assertEqual(len({str(row["chunk_key"]) for row in chunk_keys}), 2)

    def test_explicit_uuid_chunk_keys_do_not_collapse_under_source_uri_canonicalization(self) -> None:
        self.fx.write_corpus_note("shared.md", "---\nuuid: folder//shared\n---\n# Corpus copy\n")
        scratch_note = self.fx.workroot / "allowed" / "scratch" / "shared.md"
        scratch_note.parent.mkdir(parents=True, exist_ok=True)
        scratch_note.write_text("---\nuuid: folder/shared\n---\n# Scratch copy\n", encoding="utf-8")

        result = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=self.fx.app_config.corpus,
            force_rebuild=False,
        )

        self.assertEqual(result.errors, [])
        with connect_db(self.fx.build_app().corpus_db_path()) as conn:
            rows = conn.execute(
                """
                SELECT sources.name AS source_name, documents.doc_key, chunks.chunk_key
                FROM chunks
                INNER JOIN documents ON documents.id = chunks.doc_id
                INNER JOIN sources ON sources.id = documents.source_id
                WHERE documents.rel_path = 'shared.md' AND chunks.chunk_kind = 'metadata'
                ORDER BY sources.name
                """
            ).fetchall()
        self.assertEqual(
            [(str(row["source_name"]), str(row["doc_key"])) for row in rows],
            [
                ("corpus", "folder//shared"),
                ("scratch", "folder/shared"),
            ],
        )
        self.assertEqual(len({str(row["chunk_key"]) for row in rows}), 2)

    def test_corpus_contract_changes_when_metadata_projection_version_changes(self) -> None:
        original = self.fx.app_config.corpus
        baseline = sync_corpus(
            db_path=self.fx.build_app().corpus_db_path(),
            source_specs=self.fx.app_config.corpus.sources,
            security_root=self.fx.roots.security_root,
            corpus_config=original,
            force_rebuild=False,
        )
        self.assertEqual(baseline.errors, [])

        with patch("agent.corpus.METADATA_PROJECTION_VERSION", "metadata_v2_test"):
            changed = sync_corpus(
                db_path=self.fx.build_app().corpus_db_path(),
                source_specs=self.fx.app_config.corpus.sources,
                security_root=self.fx.roots.security_root,
                corpus_config=original,
                force_rebuild=False,
            )
        self.assertNotEqual(baseline.corpus_contract_sig, changed.corpus_contract_sig)


if __name__ == "__main__":
    unittest.main()
