from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent import grounding as grounding_module
from agent.tools import ToolError
from agent.embeddings import sync_embeddings
from tests.support import AppFixture, dummy_embedder_factory


class RuntimeAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note(
            "note.md",
            "---\n"
            "uuid: app-doc\n"
            "---\n"
            "\n"
            "## Alpha\n"
            "alpha evidence lives here.\n",
        )

    def tearDown(self) -> None:
        self.fx.close()

    @patch("agent.app.ensure_ollama_up")
    @patch("agent.app.ollama_chat")
    def test_chat_returns_structured_result_without_printing(self, mock_chat, mock_ensure) -> None:
        _ = mock_ensure
        mock_chat.return_value = {"message": {"content": "pong"}}
        app = self.fx.build_app()
        result = app.chat("ping")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "pong")
        self.assertEqual(result.model_used, "test-model")
        self.assertTrue((result.run_dir / "run.json").exists())

    def test_answer_grounded_uses_real_retrieval_and_structured_result(self) -> None:
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
        with patch("agent.app.create_embedder", side_effect=dummy_embedder_factory):
            seed_retrieval = app.retrieve("alpha evidence")
            self.assertGreaterEqual(len(seed_retrieval.candidates), 1)
            chunk = seed_retrieval.candidates[0]
            answer_text = f"alpha evidence lives here. [source: {chunk.rel_path}#{chunk.heading_path} | {chunk.chunk_key}]"
            grounded_retrieval = {}
            original_retrieve = grounding_module.retrieve

            with patch("agent.grounding.ensure_ollama_up"):
                with patch("agent.grounding.create_embedder", side_effect=dummy_embedder_factory):
                    def capture_grounded_retrieval(*args, **kwargs):
                        result = original_retrieve(*args, **kwargs)
                        grounded_retrieval["result"] = result
                        return result

                    with patch("agent.grounding.retrieve", side_effect=capture_grounded_retrieval):
                        with patch("agent.grounding.ollama_chat", return_value={"message": {"content": answer_text}}):
                            grounded = app.answer_grounded("Where is alpha evidence?")

        self.assertTrue(grounded.ok)
        self.assertIn(chunk.chunk_key, grounded.text)
        self.assertIn("result", grounded_retrieval)
        retrieval = grounded_retrieval["result"]
        run_path = grounded.run_dir / "run.json"
        self.assertTrue(run_path.exists())
        persisted = json.loads(run_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["retrieval"]["lexical_backend_mode"], retrieval.lexical_backend_mode)
        self.assertEqual(persisted["retrieval"]["lexical_backend_warning"], retrieval.lexical_backend_warning)
        self.assertEqual(persisted["retrieval"]["rerank_applied"], retrieval.rerank_applied)
        self.assertEqual(persisted["retrieval"]["rerank_intent"], retrieval.rerank_intent)
        self.assertEqual(
            persisted["retrieval"]["rerank_signals_available"],
            retrieval.rerank_signals_available,
        )

    def test_sync_embeddings_accepts_keyword_only_embedder_factory(self) -> None:
        app = self.fx.build_app()
        ingest = app.ingest_corpus()
        self.assertEqual(ingest.errors, [])

        def keyword_only_factory(*, embeddings_cfg, base_url: str, timeout_s: int):
            _ = embeddings_cfg, base_url, timeout_s
            return dummy_embedder_factory(None, "", 0)

        embed = sync_embeddings(
            app_config=app.config,
            security_root=app.roots.security_root,
            corpus_db_path=app.corpus_db_path(),
            embedder_factory=keyword_only_factory,
        )

        self.assertEqual(embed.errors, [])
        self.assertEqual(embed.embedded_written, 2)

    def test_export_memory_writes_json_under_security_root(self) -> None:
        app = self.fx.build_app()
        app.add_memory(
            memory_type="user_fact",
            source="manual",
            content="alpha is important",
            chunk_keys=[],
        )

        payload = app.export_memory("runs/memory-export.json")

        export_path = self.fx.workroot / "runs" / "memory-export.json"
        self.assertTrue(export_path.exists())
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["content"], "alpha is important")

    def test_export_memory_denies_path_outside_security_root(self) -> None:
        app = self.fx.build_app()
        with self.assertRaises(ToolError) as ctx:
            app.export_memory("../memory-export.json")
        self.assertEqual(ctx.exception.code, "PATH_DENIED")


if __name__ == "__main__":
    unittest.main()
