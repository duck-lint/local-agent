from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

from agent.embeddings import sync_embeddings
from tests.support import AppFixture, dummy_embedder_factory


class DoctorRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note(
            "doctor.md",
            "---\n"
            "uuid: doctor-doc\n"
            "---\n"
            "\n"
            "## Ready\n"
            "doctor retrieval evidence.\n",
        )

    def tearDown(self) -> None:
        self.fx.close()

    def test_doctor_require_grounding_passes_when_runtime_is_ready(self) -> None:
        app = self.fx.build_app()
        ingest = app.ingest_corpus()
        self.assertEqual(ingest.errors, [])
        embed = sync_embeddings(
            app_config=app.config,
            security_root=app.roots.security_root,
            corpus_db_path=app.corpus_db_path(),
            embedder_factory=dummy_embedder_factory,
        )
        self.assertEqual(embed.errors, [])

        with patch("agent.doctor.ensure_ollama_up"):
            with patch("agent.doctor.create_embedder", return_value=dummy_embedder_factory(None, "", 0)):
                report = app.doctor(check_ollama=True, require_grounding=True)

        self.assertTrue(report.ok)
        self.assertTrue(any(check.code == "DOCTOR_RETRIEVAL_READY" for check in report.checks))

    def test_doctor_require_grounding_fails_when_embeddings_are_missing(self) -> None:
        app = self.fx.build_app()
        ingest = app.ingest_corpus()
        self.assertEqual(ingest.errors, [])

        report = app.doctor(check_ollama=False, require_grounding=True)
        self.assertFalse(report.ok)
        self.assertTrue(any(check.code == "DOCTOR_EMBEDDINGS_MISSING" for check in report.checks))

    def test_doctor_reports_invalid_corpus_schema_without_crashing(self) -> None:
        app = self.fx.build_app()
        db_path = app.corpus_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE old_docs(id INTEGER PRIMARY KEY)")
            conn.commit()

        report = app.doctor(check_ollama=False, require_grounding=False)
        self.assertFalse(report.ok)
        self.assertTrue(any(check.code == "DOCTOR_CORPUS_DB_INVALID" for check in report.checks))


if __name__ == "__main__":
    unittest.main()
