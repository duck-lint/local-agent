from __future__ import annotations

import unittest
from unittest.mock import patch

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
            retrieval = app.retrieve("alpha evidence")
            self.assertGreaterEqual(len(retrieval.candidates), 1)
            chunk = retrieval.candidates[0]
            answer_text = f"alpha evidence lives here. [source: {chunk.rel_path}#{chunk.heading_path} | {chunk.chunk_key}]"

            with patch("agent.grounding.ensure_ollama_up"):
                with patch("agent.grounding.create_embedder", side_effect=dummy_embedder_factory):
                    with patch("agent.grounding.ollama_chat", return_value={"message": {"content": answer_text}}):
                        grounded = app.answer_grounded("Where is alpha evidence?")

        self.assertTrue(grounded.ok)
        self.assertIn(chunk.chunk_key, grounded.text)
        self.assertTrue((grounded.run_dir / "run.json").exists())

    def test_answer_grounded_strict_validation_rejects_unparseable_citation_marker(self) -> None:
        app = self.fx.build_app(
            config_override={
                "grounding": {
                    "citation_validation": {
                        "strict": True,
                    }
                }
            }
        )
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
            retrieval = app.retrieve("alpha evidence")
            self.assertGreaterEqual(len(retrieval.candidates), 1)
            chunk = retrieval.candidates[0]
            # Deliberately omit the `|` separator so the citation marker is counted but not parseable.
            answer_text = f"alpha evidence lives here. [source: {chunk.rel_path}#{chunk.heading_path} {chunk.chunk_key}]"

            with patch("agent.grounding.ensure_ollama_up"):
                with patch("agent.grounding.create_embedder", side_effect=dummy_embedder_factory):
                    with patch("agent.grounding.ollama_chat", return_value={"message": {"content": answer_text}}):
                        grounded = app.answer_grounded("Where is alpha evidence?")

        self.assertFalse(grounded.ok)
        self.assertEqual(grounded.error_code, "ASK_CITATION_INVALID")
        self.assertIn("Citation validation failed", str(grounded.error_message))
        citation_report = grounded.record["citation_validation"]
        self.assertEqual(citation_report["citation_markers_found"], 1)
        self.assertEqual(citation_report["parsed_citations_count"], 0)
        self.assertEqual(citation_report["unparseable_citations_count"], 1)
        self.assertTrue(citation_report["all_citations_unparseable"])
        self.assertTrue((grounded.run_dir / "run.json").exists())

    def test_answer_grounded_accepts_normalized_heading_punctuation(self) -> None:
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
            retrieval = app.retrieve("alpha evidence")
            self.assertGreaterEqual(len(retrieval.candidates), 1)
            chunk = retrieval.candidates[0]
            # Trailing punctuation should normalize away so heading-only punctuation changes stay valid.
            answer_text = (
                "alpha evidence lives here. "
                f"[source: {chunk.rel_path}#{chunk.heading_path}!!! | {chunk.chunk_key}]"
            )

            with patch("agent.grounding.ensure_ollama_up"):
                with patch("agent.grounding.create_embedder", side_effect=dummy_embedder_factory):
                    with patch("agent.grounding.ollama_chat", return_value={"message": {"content": answer_text}}):
                        grounded = app.answer_grounded("Where is alpha evidence?")

        self.assertTrue(grounded.ok)
        self.assertTrue(grounded.record["citation_validation"]["valid"])
        self.assertEqual(grounded.record["citation_validation"]["counts"]["path_mismatches"], 0)
        self.assertIn("missing=0", grounded.text)
        self.assertIn("path_mismatches=0", grounded.text)
        self.assertIn("hash_mismatches=0", grounded.text)
        self.assertIn("not_in_snapshot=0", grounded.text)


if __name__ == "__main__":
    unittest.main()
