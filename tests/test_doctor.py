from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.__main__ import collect_doctor_checks, deep_merge_config
from agent.index_db import connect_db
from agent.indexer import SourceSpec, index_sources
from agent.tools import configure_tool_security


class DoctorChecksTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.workroot = self.tmp_path / "workroot"
        self.allowed = self.workroot / "allowed"
        self.corpus = self.allowed / "corpus"
        self.scratch = self.allowed / "scratch"
        self.runs = self.workroot / "runs"
        self.corpus.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)

        (self.corpus / "a.md").write_text(
            "---\n"
            "uuid: a\n"
            "---\n"
            "\n"
            "alpha text\n",
            encoding="utf-8",
        )
        self.db_path = self.workroot / "index" / "index.sqlite"
        self.config_path = self.tmp_path / "repo" / "configs" / "default.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text("model: test\n", encoding="utf-8")

        configure_tool_security(
            {
                "allowed_roots": ["allowed/corpus/", "allowed/scratch/", "runs/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": True,
            },
            workspace_root=self.workroot,
            resolved_config_path=self.config_path,
        )

        self.cfg = {
            "ollama_base_url": "http://127.0.0.1:11434",
            "timeout_s": 1,
            "security": {
                "allowed_roots": ["allowed/corpus/", "allowed/scratch/", "runs/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": True,
            },
            "phase2": {
                "index_db_path": "index/index.sqlite",
                "sources": [
                    {"name": "corpus", "root": "allowed/corpus/", "kind": "corpus"},
                    {"name": "scratch", "root": "allowed/scratch/", "kind": "scratch"},
                ],
                "chunking": {
                    "scheme": "obsidian_v1",
                    "max_chars": 120,
                    "overlap": 20,
                },
            },
        }
        self.roots = {"security_root": self.workroot}

        summary = index_sources(
            db_path=self.db_path,
            source_specs=[
                SourceSpec(name="corpus", root="allowed/corpus/", kind="corpus"),
                SourceSpec(name="scratch", root="allowed/scratch/", kind="scratch"),
            ],
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=120,
            overlap=20,
        )
        self.assertEqual(summary.errors, [])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_collect_doctor_checks_happy_path_without_ollama(self) -> None:
        checks = collect_doctor_checks(
            deep_merge_config({}, self.cfg),
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
        )
        failures = [c for c in checks if not c.ok]
        self.assertEqual(failures, [])

    def test_collect_doctor_checks_detects_missing_chunk_key(self) -> None:
        with connect_db(self.db_path) as conn:
            conn.execute("UPDATE chunks SET chunk_key = NULL")
            conn.commit()

        checks = collect_doctor_checks(
            deep_merge_config({}, self.cfg),
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
        )
        failure_codes = {c.error_code for c in checks if not c.ok}
        self.assertIn("DOCTOR_CHUNK_KEY_MISSING", failure_codes)

    def test_collect_doctor_checks_detects_scheme_mismatch(self) -> None:
        summary = index_sources(
            db_path=self.db_path,
            source_specs=[
                SourceSpec(name="corpus", root="allowed/corpus/", kind="corpus"),
                SourceSpec(name="scratch", root="allowed/scratch/", kind="scratch"),
            ],
            security_root=self.workroot,
            scheme="fixed_window_v1",
            max_chars=120,
            overlap=20,
        )
        self.assertEqual(summary.errors, [])

        checks = collect_doctor_checks(
            deep_merge_config({}, self.cfg),
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
        )
        failure_codes = {c.error_code for c in checks if not c.ok}
        self.assertIn("DOCTOR_CHUNK_SCHEME_MISMATCH", failure_codes)

    def test_collect_doctor_checks_detects_blank_scheme(self) -> None:
        with connect_db(self.db_path) as conn:
            conn.execute(
                "UPDATE chunks SET scheme = '   ' WHERE id IN (SELECT id FROM chunks ORDER BY id LIMIT 1)"
            )
            conn.commit()

        checks = collect_doctor_checks(
            deep_merge_config({}, self.cfg),
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
        )
        failure_codes = {c.error_code for c in checks if not c.ok}
        self.assertIn("DOCTOR_CHUNK_SCHEME_MISMATCH", failure_codes)

    def test_collect_doctor_checks_detects_docs_without_chunks(self) -> None:
        with connect_db(self.db_path) as conn:
            row = conn.execute("SELECT id FROM docs ORDER BY rel_path LIMIT 1").fetchone()
            self.assertIsNotNone(row)
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (int(row["id"]),))
            conn.commit()

        checks = collect_doctor_checks(
            deep_merge_config({}, self.cfg),
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
        )
        failure_codes = {c.error_code for c in checks if not c.ok}
        self.assertIn("DOCTOR_DOCS_WITHOUT_CHUNKS", failure_codes)

    def test_collect_doctor_checks_detects_chunker_sig_mismatch(self) -> None:
        with connect_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO meta(key, value)
                VALUES ('chunker_sig', 'deadbeef')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )
            conn.commit()

        checks = collect_doctor_checks(
            deep_merge_config({}, self.cfg),
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
        )
        failure_codes = {c.error_code for c in checks if not c.ok}
        self.assertIn("DOCTOR_CHUNKER_SIG_MISMATCH", failure_codes)

    def test_collect_doctor_checks_detects_missing_db(self) -> None:
        cfg = deep_merge_config({}, self.cfg)
        cfg["phase2"]["index_db_path"] = "index/missing.sqlite"
        missing_path = self.workroot / "index" / "missing.sqlite"
        self.assertFalse(missing_path.exists())

        checks = collect_doctor_checks(
            cfg,
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
        )
        failure_codes = {c.error_code for c in checks if not c.ok}
        self.assertIn("DOCTOR_INDEX_DB_MISSING", failure_codes)


if __name__ == "__main__":
    unittest.main()
