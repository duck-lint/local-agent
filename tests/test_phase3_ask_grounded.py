from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.__main__ import run_ask_grounded
from agent.retrieval import RetrievedChunk, RetrievalResult


class Phase3AskGroundedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.workroot = self.tmp_path / "workroot"
        (self.workroot / "runs").mkdir(parents=True, exist_ok=True)
        (self.workroot / "index").mkdir(parents=True, exist_ok=True)

        self.cfg = {
            "ollama_base_url": "http://127.0.0.1:11434",
            "timeout_s": 1,
            "timeout_s_big_second": 1,
            "max_tokens_big_second": 128,
            "temperature": 0.0,
            "model": "test-model",
            "model_fast": "test-model",
            "model_big": "test-model",
            "prefer_fast": True,
            "big_triggers": [],
            "phase2": {
                "index_db_path": "index/index.sqlite",
                "sources": [{"name": "corpus", "root": "allowed/corpus/", "kind": "corpus"}],
                "chunking": {"scheme": "obsidian_v1", "max_chars": 120, "overlap": 20},
            },
            "phase3": {
                "embeddings_db_path": "embeddings/db/embeddings.sqlite",
                "embed": {
                    "provider": "ollama",
                    "model_id": "nomic-embed-text-v1.5",
                    "preprocess": "obsidian_v1",
                    "chunk_preprocess_sig": "",
                    "query_preprocess_sig": "",
                    "batch_size": 8,
                },
                "retrieve": {
                    "lexical_k": 5,
                    "vector_k": 5,
                    "vector_fetch_k": 0,
                    "rel_path_prefix": "",
                    "fusion": "simple_union",
                },
                "memory": {"durable_db_path": "memory/durable.sqlite", "enabled": True},
            },
        }
        self.roots = {"security_root": self.workroot}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_missing_citations_triggers_fallback(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key="0123456789abcdef0123456789abcdef",
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": "Answer without citations"}}

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("Insufficient citation-grounded answer", output)
        self.assertIn("[source: a.md#H2: Alpha | 0123456789abcdef0123456789abcdef]", output)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    def test_no_candidates_returns_insufficient_evidence(
        self,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=0,
            vector_candidates_prefilter=0,
            vector_candidates_postfilter=0,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[],
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("Insufficient public evidence", output)


if __name__ == "__main__":
    unittest.main()
