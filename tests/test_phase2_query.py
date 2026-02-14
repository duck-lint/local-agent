from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.__main__ import render_query_results
from agent.index_db import connect_db, query_chunks_lexical
from agent.indexer import SourceSpec, index_sources
from agent.tools import configure_tool_security


class Phase2QueryTests(unittest.TestCase):
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
            "---\n"
            "\n"
            "## Typed Section\n"
            "typed_token appears in this body.\n",
            encoding="utf-8",
        )
        (self.scratch / "absent.md").write_text(
            "absent_token appears in a doc with no frontmatter.\n",
            encoding="utf-8",
        )
        (self.corpus / "broken.md").write_text(
            "---\n"
            "uuid: \"oops\n"
            "---\n"
            "\n"
            "unknown_token appears in broken yaml doc.\n",
            encoding="utf-8",
        )

        self.sources = [
            SourceSpec(name="corpus", root="allowed/corpus/", kind="corpus"),
            SourceSpec(name="scratch", root="allowed/scratch/", kind="scratch"),
        ]
        summary = index_sources(
            db_path=self.db_path,
            source_specs=self.sources,
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=120,
            overlap=20,
        )
        self.assertEqual(summary.errors, [])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _render_for_query(self, query: str) -> str:
        with connect_db(self.db_path) as conn:
            rows = query_chunks_lexical(conn, query_text=query, limit=3)
        return render_query_results([dict(row) for row in rows], query_text=query)

    def test_query_includes_provenance_and_present_typed_status(self) -> None:
        rendered = self._render_for_query("typed_token")
        self.assertIn("provenance: source=corpus kind=corpus", rendered)
        self.assertIn("chunk: scheme=obsidian_v1", rendered)
        self.assertIn("heading_path: H2: Typed Section", rendered)
        self.assertIn("metadata=present", rendered)
        self.assertIn("yaml_present=1", rendered)
        self.assertIn("yaml_parse_ok=1", rendered)
        self.assertIn("required_keys_present=1", rendered)

    def test_query_marks_absent_metadata_explicitly(self) -> None:
        rendered = self._render_for_query("absent_token")
        self.assertIn("provenance: source=scratch kind=scratch", rendered)
        self.assertIn("metadata=absent", rendered)
        self.assertIn("yaml_present=0", rendered)
        self.assertIn("yaml_parse_ok=unknown", rendered)
        self.assertIn("required_keys_present=unknown", rendered)

    def test_query_marks_unknown_metadata_explicitly(self) -> None:
        rendered = self._render_for_query("unknown_token")
        self.assertIn("provenance: source=corpus kind=corpus", rendered)
        self.assertIn("metadata=unknown", rendered)
        self.assertIn("yaml_present=1", rendered)
        self.assertIn("yaml_parse_ok=0", rendered)
        self.assertIn("required_keys_present=unknown", rendered)
        self.assertIn("yaml_error:", rendered)


if __name__ == "__main__":
    unittest.main()
