from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from tests.support import AppFixture, dummy_embedder_factory
from agent.embeddings import sync_embeddings


def _refinement_overrides(*, refine_enabled: bool, rewrite_enabled: bool, acronyms=None):
    return {
        "retrieval": {
            "refinement_round_enabled": refine_enabled,
            "coverage_predicate": {
                "lexical_threshold": 0.99,
                "vector_threshold": 0.99,
                "memory_weight": 0.0,
                "top_n": 10,
            },
            "rewrite": {
                "rule_based_enabled": rewrite_enabled,
                "acronyms_path": "configs/__nonexistent__.yaml",
                "acronyms": acronyms or {},
                "synonyms": {},
            },
        }
    }


class RefinementIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note(
            "alpha.md",
            "---\nuuid: alpha-doc\ntitle: Alpha\n---\n\n## Alpha\nalpha evidence lives here.\n",
        )
        self.fx.write_corpus_note(
            "beta.md",
            "---\nuuid: beta-doc\ntitle: Beta\n---\n\n## Beta\nbeta application programming interface evidence.\n",
        )

    def tearDown(self) -> None:
        self.fx.close()

    def _build_and_ingest(self, overrides):
        app = self.fx.build_app(config_override=overrides)
        self.assertEqual(app.ingest_corpus().errors, [])
        embed = sync_embeddings(
            app_config=app.config,
            security_root=app.roots.security_root,
            corpus_db_path=app.corpus_db_path(),
            embedder_factory=dummy_embedder_factory,
        )
        self.assertEqual(embed.errors, [])
        return app

    def _run_grounded(self, app, question: str, answer_chunk_key=None, answer_rel_path=None, answer_heading=None):
        from agent.app import LocalAgentApp  # noqa: F401
        with patch("agent.app.create_embedder", side_effect=dummy_embedder_factory):
            seed = app.retrieve("alpha")
            self.assertGreaterEqual(len(seed.candidates), 1)
            chunk = seed.candidates[0]
            ans_text = (
                f"answer body. "
                f"[source: {chunk.rel_path}#{chunk.heading_path} | {chunk.chunk_key}]"
            )
            with patch("agent.grounding.ensure_ollama_up"), \
                 patch("agent.grounding.create_embedder", side_effect=dummy_embedder_factory), \
                 patch("agent.grounding.ollama_chat", return_value={"message": {"content": ans_text}}):
                return app.answer_grounded(question)

    def test_round_2_skipped_when_coverage_passes(self) -> None:
        # Thresholds at 0.0 -> coverage always passes -> round 2 never fires.
        overrides = _refinement_overrides(refine_enabled=True, rewrite_enabled=True, acronyms={"alpha": "alpha-expanded"})
        overrides["retrieval"]["coverage_predicate"]["lexical_threshold"] = 0.0
        overrides["retrieval"]["coverage_predicate"]["vector_threshold"] = 0.0
        app = self._build_and_ingest(overrides)
        result = self._run_grounded(app, "alpha")
        self.assertTrue(result.ok)
        persisted = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(persisted["coverage"])
        self.assertFalse(persisted["coverage"]["should_refine"])
        self.assertEqual(len(persisted["retrieval_rounds"]), 1)
        self.assertFalse(persisted["refinement_applied"])
        self.assertEqual(persisted["rewritten_query"], "")

    def test_round_2_fires_when_coverage_fails(self) -> None:
        # Multi-token query where most tokens are unmatched -> lexical coverage
        # well below 0.99 threshold -> should_refine=True.
        overrides = _refinement_overrides(
            refine_enabled=True,
            rewrite_enabled=True,
            acronyms={"alpha": "beta application programming interface evidence"},
        )
        app = self._build_and_ingest(overrides)
        result = self._run_grounded(app, "alpha xyzzymystery quux foobarbaz")
        self.assertTrue(result.ok)
        persisted = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(persisted["coverage"])
        self.assertTrue(persisted["coverage"]["should_refine"])
        # If round 2 produced new candidates, refinement_applied is True and
        # there are 2 rounds. If round 2 produced no new candidates, we degrade
        # to round 1 and refinement_applied stays False.
        self.assertGreaterEqual(len(persisted["retrieval_rounds"]), 1)
        self.assertNotEqual(persisted["rewritten_query"], "")
        if persisted["refinement_applied"]:
            self.assertEqual(len(persisted["retrieval_rounds"]), 2)
            self.assertEqual(persisted["retrieval_rounds"][1]["round_index"], 2)
            self.assertIn("acronym_expansion", persisted["retrieval_rounds"][1]["rewrite"]["transforms_applied"])

    def test_round_2_disabled_flag_forces_skip(self) -> None:
        # Coverage would fail, but refinement_round_enabled is False -> skip.
        overrides = _refinement_overrides(
            refine_enabled=False,
            rewrite_enabled=True,
            acronyms={"alpha": "expanded alpha"},
        )
        app = self._build_and_ingest(overrides)
        result = self._run_grounded(app, "alpha")
        self.assertTrue(result.ok)
        persisted = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertIsNone(persisted["coverage"])
        self.assertEqual(len(persisted["retrieval_rounds"]), 1)
        self.assertFalse(persisted["refinement_applied"])
        # Back-compat: retrieval mirrors retrieval_rounds[-1] when refinement off.
        self.assertEqual(persisted["retrieval"], persisted["retrieval_rounds"][-1])

    def test_round_2_no_new_candidates_degrades_to_round_1(self) -> None:
        # rewrite enabled but acronyms map empty -> rewriter is identity ->
        # no round 2 even though coverage fails.
        overrides = _refinement_overrides(
            refine_enabled=True,
            rewrite_enabled=True,
            acronyms={},  # no map -> identity rewrite
        )
        app = self._build_and_ingest(overrides)
        result = self._run_grounded(app, "alpha xyzzymystery quux foobarbaz")
        self.assertTrue(result.ok)
        persisted = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(persisted["coverage"])
        self.assertTrue(persisted["coverage"]["should_refine"])
        self.assertEqual(len(persisted["retrieval_rounds"]), 1)
        self.assertFalse(persisted["refinement_applied"])
        self.assertEqual(persisted["rewritten_query"], "")


if __name__ == "__main__":
    unittest.main()
