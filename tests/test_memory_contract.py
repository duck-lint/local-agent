from __future__ import annotations

import json
import unittest

from agent.corpus_db import connect_db as connect_corpus_db
from agent.memory_db import add_memory, connect_db as connect_memory_db, init_db as init_memory_db
from tests.support import AppFixture


class MemoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note(
            "memory.md",
            "---\n"
            "uuid: memory-doc\n"
            "---\n"
            "\n"
            "## Evidence\n"
            "memory evidence lives here.\n",
        )

    def tearDown(self) -> None:
        self.fx.close()

    def test_add_memory_rejects_chunk_keys_not_in_current_corpus(self) -> None:
        app = self.fx.build_app()
        ingest = app.ingest_corpus()
        self.assertEqual(ingest.errors, [])

        with self.assertRaisesRegex(ValueError, "not present in the current corpus"):
            app.add_memory(
                memory_type="user_fact",
                source="derived_from_evidence",
                content="remember this",
                chunk_keys=["missing-chunk-key"],
            )

    def test_export_memory_reports_provenance_and_dangling_evidence(self) -> None:
        app = self.fx.build_app()
        ingest = app.ingest_corpus()
        self.assertEqual(ingest.errors, [])

        with connect_corpus_db(app.corpus_db_path()) as corpus_conn:
            row = corpus_conn.execute("SELECT chunk_key FROM chunks ORDER BY chunk_index LIMIT 1").fetchone()
            self.assertIsNotNone(row)
            chunk_key = str(row["chunk_key"])

        init_db_path = app.memory_db_path()
        init_memory_db(init_db_path)
        with connect_memory_db(init_db_path) as mem_conn:
            add_memory(
                mem_conn,
                memory_type="user_fact",
                source="derived_from_evidence",
                content="good evidence",
                chunk_keys=[chunk_key],
            )
            add_memory(
                mem_conn,
                memory_type="user_fact",
                source="derived_from_evidence",
                content="stale evidence",
                chunk_keys=["missing-chunk-key"],
            )
            mem_conn.commit()

        target = self.fx.workroot / "allowed" / "scratch" / "memory-export.json"
        payload = app.export_memory(str(target))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["provenance"]["corpus_contract_sig"], ingest.corpus_contract_sig)
        self.assertTrue(payload["validation"]["checked_against_current_corpus"])
        self.assertEqual(payload["validation"]["dangling_evidence_chunk_keys"], ["missing-chunk-key"])

        written = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(written["validation"]["dangling_evidence_chunk_keys"], ["missing-chunk-key"])


if __name__ == "__main__":
    unittest.main()
