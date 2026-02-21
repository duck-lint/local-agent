from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.embedding_fingerprint import pack_vector_f32_le
from agent.embeddings_db import connect_db, fetch_embeddings_map, init_db, upsert_embedding


class EmbeddingsDbTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db_path = self.tmp_path / "embeddings.sqlite"
        init_db(self.db_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fetch_embeddings_map_handles_more_than_999_keys(self) -> None:
        with connect_db(self.db_path) as conn:
            upsert_embedding(
                conn,
                chunk_key="00000000000000000000000000000001",
                embed_sig="sig-1",
                model_id="m",
                dim=2,
                preprocess_sig="p",
                vector_blob=pack_vector_f32_le([1.0, 0.0]),
            )
            upsert_embedding(
                conn,
                chunk_key="00000000000000000000000000000002",
                embed_sig="sig-2",
                model_id="m",
                dim=2,
                preprocess_sig="p",
                vector_blob=pack_vector_f32_le([0.0, 1.0]),
            )
            conn.commit()

            many_keys = [f"{i:032x}" for i in range(1200)]
            rows = fetch_embeddings_map(conn, many_keys)

        self.assertIn("00000000000000000000000000000001", rows)
        self.assertIn("00000000000000000000000000000002", rows)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
