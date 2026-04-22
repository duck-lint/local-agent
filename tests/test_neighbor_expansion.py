from __future__ import annotations

import unittest

from agent.corpus_db import connect_db, fetch_neighbor_chunks
from agent.retrieval import RetrievedChunk, RetrievalResult, expand_neighbors
from tests.support import AppFixture


def _doc_with_n_paragraphs(n: int) -> str:
    head = (
        "---\n"
        "uuid: test-doc\n"
        "title: Test Document\n"
        "---\n\n"
        "## Main Heading\n"
    )
    body = "\n\n".join(f"Paragraph {i} content here." for i in range(n))
    return head + body + "\n"


def _make_result(candidates: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        corpus_contract_sig="sig",
        embed_model_id="m",
        chunk_preprocess_sig="cp",
        query_preprocess_sig="qp",
        embed_db_schema_version=1,
        lexical_backend_mode="projection_substring",
        lexical_backend_warning="",
        vector_fetch_k_used=0,
        vector_candidates_scored=0,
        vector_candidates_prefilter=0,
        vector_candidates_postfilter=0,
        rel_path_prefix_applied=False,
        vector_filter_warning="",
        rerank_applied=False,
        rerank_intent="",
        rerank_signals_available=False,
        candidates=candidates,
    )


def _row_to_chunk(row: dict, *, rel_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_key=str(row["chunk_key"]),
        doc_key=str(row["doc_key"]),
        chunk_kind=str(row["chunk_kind"]),
        rel_path=rel_path,
        heading_path=str(row["heading_path"]),
        chunk_anchor=str(row["chunk_anchor"]),
        chunk_title=str(row["chunk_title"]),
        text=str(row["text"]),
        score=1.0,
        method="lexical",
        lexical_score=1.0,
        vector_score=0.0,
    )


class NeighborExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def _ingest(self, doc_text: str):
        self.fx.write_corpus_note("doc.md", doc_text)
        app = self.fx.build_app()
        ingest = app.ingest_corpus()
        self.assertEqual(ingest.errors, [])
        return app

    def test_adjacent_only_scope_returns_plus_minus_one(self) -> None:
        app = self._ingest(_doc_with_n_paragraphs(4))
        with connect_db(app.corpus_db_path()) as conn:
            rows = conn.execute(
                "SELECT chunks.chunk_key AS chunk_key, chunks.chunk_index AS chunk_index, "
                "documents.rel_path AS rel_path "
                "FROM chunks JOIN documents ON documents.id = chunks.doc_id "
                "WHERE documents.rel_path = 'doc.md' "
                "ORDER BY chunk_index ASC"
            ).fetchall()
            self.assertGreaterEqual(len(rows), 3)
            middle = rows[len(rows) // 2]
            middle_idx = int(middle["chunk_index"])
            neighbors = fetch_neighbor_chunks(
                conn,
                chunk_keys=[str(middle["chunk_key"])],
                scope="adjacent_only",
            )
        # Middle chunk is interior, so we expect both ±1 neighbors.
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(
            {n["chunk_index"] for n in neighbors},
            {middle_idx - 1, middle_idx + 1},
        )
        self.assertNotIn(str(middle["chunk_key"]), {n["chunk_key"] for n in neighbors})

    def test_dedup_against_initial_candidates(self) -> None:
        # Pre-load every chunk of the doc as a candidate. Adjacent neighbors of
        # any chunk are necessarily already in the candidate set, so expansion
        # must add zero new chunks.
        app = self._ingest(_doc_with_n_paragraphs(4))
        with connect_db(app.corpus_db_path()) as conn:
            rows = conn.execute(
                "SELECT chunks.chunk_key, chunks.doc_key, chunks.chunk_kind, "
                "chunks.heading_path, chunks.chunk_anchor, chunks.chunk_title, "
                "chunks.text, chunks.chunk_index, documents.rel_path "
                "FROM chunks JOIN documents ON documents.id = chunks.doc_id "
                "WHERE documents.rel_path = 'doc.md' "
                "ORDER BY chunk_index ASC"
            ).fetchall()
        self.assertGreaterEqual(len(rows), 3)
        candidates = [
            _row_to_chunk(dict(r), rel_path=str(r["rel_path"])) for r in rows
        ]
        result = _make_result(candidates)
        expanded = expand_neighbors(
            result,
            corpus_db_path=app.corpus_db_path(),
            scope="adjacent_only",
        )
        self.assertTrue(expanded.neighbor_expansion_applied)
        self.assertEqual(expanded.neighbor_scope, "adjacent_only")
        self.assertEqual(expanded.neighbor_chunks_added, 0)
        self.assertEqual(len(expanded.candidates), len(candidates))

    def test_unknown_scope_raises(self) -> None:
        app = self._ingest(_doc_with_n_paragraphs(2))
        result = _make_result([])
        with self.assertRaises(ValueError):
            expand_neighbors(
                result,
                corpus_db_path=app.corpus_db_path(),
                scope="bogus",
            )


if __name__ == "__main__":
    unittest.main()
