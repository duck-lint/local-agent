from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.index_db import connect_db, init_db
from agent.indexer import SourceSpec, index_sources
from agent.tools import configure_tool_security


class Phase2IndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.workroot = self.tmp_path / "workroot"
        self.corpus = self.workroot / "allowed" / "corpus"
        self.scratch = self.workroot / "allowed" / "scratch"
        self.corpus.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workroot / "index" / "index.sqlite"

        configure_tool_security(
            {
                "allowed_roots": ["allowed/corpus/", "allowed/scratch/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": True,
            },
            workspace_root=self.workroot,
        )

        (self.corpus / "typed.md").write_text(
            "---\n"
            "uuid: 123e4567-e89b-12d3-a456-426614174000\n"
            "note_version: v0.1.3\n"
            "---\n"
            "\n"
            "alpha body text for retrieval.\n",
            encoding="utf-8",
        )
        (self.scratch / "no_frontmatter.md").write_text(
            "this document has no frontmatter and stays untyped.\n",
            encoding="utf-8",
        )
        (self.corpus / "broken.md").write_text(
            "---\n"
            "uuid: \"not-closed\n"
            "note_version: v0.1.3\n"
            "---\n"
            "\n"
            "broken yaml body with searchable text.\n",
            encoding="utf-8",
        )

        self.sources = [
            SourceSpec(name="corpus", root="allowed/corpus/", kind="corpus"),
            SourceSpec(name="scratch", root="allowed/scratch/", kind="scratch"),
        ]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_indexing_flags_chunks_incremental_and_prune(self) -> None:
        summary1 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary1.errors, [])
        self.assertEqual(summary1.docs_scanned, 3)
        self.assertEqual(summary1.docs_changed, 3)
        self.assertEqual(summary1.docs_pruned, 0)

        with connect_db(self.db_path) as conn:
            docs = list(
                conn.execute(
                    """
                    SELECT docs.rel_path, docs.yaml_present, docs.yaml_parse_ok,
                           docs.required_keys_present, docs.yaml_error, docs.frontmatter_json,
                           sources.name AS source_name
                    FROM docs
                    INNER JOIN sources ON sources.id = docs.source_id
                    ORDER BY docs.rel_path
                    """
                )
            )
            self.assertEqual(len(docs), 3)

            by_rel_path = {str(row["rel_path"]): row for row in docs}
            typed = by_rel_path["typed.md"]
            self.assertEqual(int(typed["yaml_present"]), 1)
            self.assertEqual(int(typed["yaml_parse_ok"]), 1)
            self.assertEqual(int(typed["required_keys_present"]), 1)
            self.assertEqual(str(typed["source_name"]), "corpus")
            self.assertIn("uuid", str(typed["frontmatter_json"]))

            no_frontmatter = by_rel_path["no_frontmatter.md"]
            self.assertEqual(int(no_frontmatter["yaml_present"]), 0)
            self.assertIsNone(no_frontmatter["yaml_parse_ok"])
            self.assertIsNone(no_frontmatter["required_keys_present"])
            self.assertEqual(str(no_frontmatter["frontmatter_json"]), "{}")
            self.assertEqual(str(no_frontmatter["source_name"]), "scratch")

            broken = by_rel_path["broken.md"]
            self.assertEqual(int(broken["yaml_present"]), 1)
            self.assertEqual(int(broken["yaml_parse_ok"]), 0)
            self.assertIsNone(broken["required_keys_present"])
            self.assertIsInstance(broken["yaml_error"], str)
            self.assertNotEqual(str(broken["yaml_error"]).strip(), "")

            typed_chunks = list(
                conn.execute(
                    """
                    SELECT chunks.text, chunks.scheme, chunks.heading_path, chunks.chunk_key
                    FROM chunks
                    INNER JOIN docs ON docs.id = chunks.doc_id
                    WHERE docs.rel_path = 'typed.md'
                    ORDER BY chunks.chunk_index
                    """
                )
            )
            self.assertGreaterEqual(len(typed_chunks), 1)
            for row in typed_chunks:
                chunk_text = str(row["text"])
                self.assertNotIn("uuid:", chunk_text)
                self.assertNotIn("---", chunk_text)
                self.assertEqual(str(row["scheme"]), "obsidian_v1")
                self.assertIsInstance(row["chunk_key"], str)
                self.assertEqual(len(str(row["chunk_key"])), 32)

        summary2 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary2.errors, [])
        self.assertEqual(summary2.docs_changed, 0)
        self.assertEqual(summary2.docs_unchanged, 3)

        (self.corpus / "typed.md").write_text(
            "---\n"
            "uuid: 123e4567-e89b-12d3-a456-426614174000\n"
            "note_version: v0.1.3\n"
            "---\n"
            "\n"
            "alpha body text for retrieval with a changed suffix token.\n",
            encoding="utf-8",
        )
        summary3 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary3.errors, [])
        self.assertGreaterEqual(summary3.docs_changed, 1)

        with connect_db(self.db_path) as conn:
            changed_chunks = list(
                conn.execute(
                    """
                    SELECT chunks.text
                    FROM chunks
                    INNER JOIN docs ON docs.id = chunks.doc_id
                    WHERE docs.rel_path = 'typed.md'
                    ORDER BY chunks.chunk_index
                    """
                )
            )
            self.assertTrue(
                any("changed suffix token" in str(row["text"]) for row in changed_chunks),
                "Expected updated chunk text after file content change.",
            )

        (self.corpus / "broken.md").unlink()
        summary4 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary4.errors, [])
        self.assertGreaterEqual(summary4.docs_pruned, 1)

        with connect_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM docs WHERE rel_path = 'broken.md'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(int(row["c"]), 0)

    def test_db_migration_v3_adds_chunk_columns_and_meta_table(self) -> None:
        legacy_db = self.workroot / "index" / "legacy.sqlite"
        legacy_db.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(legacy_db))
        try:
            conn.executescript(
                """
                CREATE TABLE sources (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    root TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE docs (
                    id INTEGER PRIMARY KEY,
                    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    rel_path TEXT NOT NULL,
                    abs_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    is_markdown INTEGER NOT NULL,
                    yaml_present INTEGER,
                    yaml_parse_ok INTEGER,
                    yaml_error TEXT,
                    required_keys_present INTEGER,
                    frontmatter_json TEXT NOT NULL,
                    discovered_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(source_id, rel_path)
                );
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(doc_id, chunk_index)
                );
                """
            )
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
        finally:
            conn.close()

        init_db(legacy_db)
        with connect_db(legacy_db) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()
            self.assertIsNotNone(version)
            self.assertEqual(int(version[0]), 3)
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(chunks)")
            }
            self.assertIn("scheme", columns)
            self.assertIn("heading_path", columns)
            self.assertIn("chunk_key", columns)
            meta_row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
            ).fetchone()
            self.assertIsNotNone(meta_row)

    def test_scheme_change_rechunks_even_when_sha_unchanged(self) -> None:
        summary1 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="fixed_window_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary1.errors, [])
        self.assertEqual(summary1.docs_changed, 3)

        summary2 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary2.errors, [])
        self.assertEqual(summary2.docs_changed, 3)
        self.assertEqual(summary2.docs_unchanged, 0)

        with connect_db(self.db_path) as conn:
            schemes = {
                str(row["scheme"])
                for row in conn.execute("SELECT DISTINCT scheme FROM chunks")
            }
            self.assertEqual(schemes, {"obsidian_v1"})

    def test_null_chunk_key_forces_rechunk_even_when_sha_unchanged(self) -> None:
        summary1 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary1.errors, [])
        self.assertEqual(summary1.docs_changed, 3)

        with connect_db(self.db_path) as conn:
            conn.execute("UPDATE chunks SET chunk_key = NULL")
            conn.commit()
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE chunk_key IS NULL OR trim(chunk_key) = ''"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertGreater(int(row["c"]), 0)

        summary2 = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary2.errors, [])
        self.assertGreater(summary2.docs_changed, 0)

        with connect_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE chunk_key IS NULL OR trim(chunk_key) = ''"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(int(row["c"]), 0)

    def test_outside_root_indexing_works_when_containment_disabled(self) -> None:
        outside = self.tmp_path / "outside_root"
        outside.mkdir(parents=True, exist_ok=True)
        (outside / "outside.md").write_text("outside text for indexing", encoding="utf-8")

        configure_tool_security(
            {
                "allowed_roots": [str(outside)],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": False,
            },
            workspace_root=self.workroot,
        )

        summary = index_sources(
            db_path=self.db_path,
            source_specs=[SourceSpec(name="outside", root=str(outside), kind="corpus")],
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=80,
            overlap=20,
        )
        self.assertEqual(summary.errors, [])
        self.assertEqual(summary.docs_scanned, 1)
        self.assertEqual(summary.docs_changed, 1)


if __name__ == "__main__":
    unittest.main()
