from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from tests.support import AppFixture, dummy_embedder_factory
from agent.embeddings import sync_embeddings
import agent.grounding as grounding_module


class RunLogShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note(
            "alpha.md",
            "---\nuuid: alpha-doc\ntitle: Alpha Doc\n---\n\n## Alpha\nalpha evidence lives here.\n",
        )

    def tearDown(self) -> None:
        self.fx.close()

    def test_run_log_includes_phase_1_fields(self) -> None:
        app = self.fx.build_app()
        self.assertEqual(app.ingest_corpus().errors, [])
        embed = sync_embeddings(
            app_config=app.config,
            security_root=app.roots.security_root,
            corpus_db_path=app.corpus_db_path(),
            embedder_factory=dummy_embedder_factory,
        )
        self.assertEqual(embed.errors, [])

        with patch("agent.app.create_embedder", side_effect=dummy_embedder_factory):
            seed = app.retrieve("alpha evidence")
            self.assertGreaterEqual(len(seed.candidates), 1)
            chunk = seed.candidates[0]
            answer_text = (
                f"alpha evidence lives here. "
                f"[source: {chunk.rel_path}#{chunk.heading_path} | {chunk.chunk_key}]"
            )
            with patch("agent.grounding.ensure_ollama_up"), \
                 patch("agent.grounding.create_embedder", side_effect=dummy_embedder_factory), \
                 patch("agent.grounding.ollama_chat", return_value={"message": {"content": answer_text}}):
                grounded = app.answer_grounded("Where is alpha evidence?")

        self.assertTrue(grounded.ok)
        persisted = json.loads((grounded.run_dir / "run.json").read_text(encoding="utf-8"))

        # Phase 1 run-log shape additions.
        self.assertIn("retrieval_rounds", persisted)
        self.assertIsInstance(persisted["retrieval_rounds"], list)
        self.assertEqual(len(persisted["retrieval_rounds"]), 1)
        # Back-compat: retrieval mirrors retrieval_rounds[-1].
        self.assertEqual(persisted["retrieval"], persisted["retrieval_rounds"][-1])

        self.assertIn("coverage", persisted)
        self.assertIsNone(persisted["coverage"])

        self.assertIn("memory_snapshot", persisted)
        self.assertIsNone(persisted["memory_snapshot"])

        self.assertIn("budget", persisted)
        budget = persisted["budget"]
        for key in (
            "wall_clock_s",
            "max_prompt_tokens",
            "elapsed_ms",
            "remaining_ms",
            "prompt_tokens_used",
            "degrade_during_generation",
        ):
            self.assertIn(key, budget)

        # Neighbor expansion fields surfaced even when disabled.
        round0 = persisted["retrieval_rounds"][0]
        self.assertEqual(round0["neighbor_expansion_applied"], False)
        self.assertEqual(round0["neighbor_chunks_added"], 0)
        self.assertEqual(round0["neighbor_warnings"], [])


if __name__ == "__main__":
    unittest.main()
