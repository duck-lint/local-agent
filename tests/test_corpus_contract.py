from __future__ import annotations

import json
import unittest

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
        self.assertGreaterEqual(result.total_chunks, 2)

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

            chunk = conn.execute(
                """
                SELECT chunk_key, doc_key, section_index, heading_path, chunk_anchor, chunk_title, out_links_json
                FROM chunks
                ORDER BY chunk_index
                LIMIT 1
                """
            ).fetchone()
            self.assertIsNotNone(chunk)
            self.assertEqual(len(str(chunk["chunk_key"])), 32)
            self.assertEqual(str(chunk["doc_key"]), "typed-doc")
            self.assertTrue(str(chunk["heading_path"]).startswith("H2: "))
            self.assertTrue(str(chunk["chunk_anchor"]))
            self.assertTrue(str(chunk["chunk_title"]))
            out_links = json.loads(str(chunk["out_links_json"]))
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


if __name__ == "__main__":
    unittest.main()
