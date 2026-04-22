"""Phase 4: end-to-end memory-rewrite isolation invariants."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.embeddings import sync_embeddings
from agent.session_memory import ChunkRef, FileSessionStore, SessionState
from tests.support import AppFixture, dummy_embedder_factory


def _seed_corpus(fx: AppFixture) -> None:
    fx.write_corpus_note(
        "alpha.md",
        "---\nuuid: alpha-doc\ntitle: Alpha Doc\n---\n\n## Alpha\nalpha evidence lives here.\n",
    )


class MemoryRewriteLeakTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        _seed_corpus(self.fx)

    def tearDown(self) -> None:
        self.fx.close()

    def _build_app(self, *, memory_rewrite_enabled: bool):
        # Enable refinement + rule-based rewrite so the memory channel actually
        # touches rewritten_query.
        return self.fx.build_app(
            config_override={
                "session": {
                    "enabled": True,
                    "require_daemon_for_cli": False,
                    "memory_rewrite_enabled": memory_rewrite_enabled,
                    "coverage_memory_weight": 0.0,
                },
                "daemon": {"enabled": False},
                "retrieval": {
                    "refinement_round_enabled": True,
                    "rewrite": {"rule_based_enabled": True},
                    "coverage_predicate": {
                        "lexical_threshold": 1.1,  # force refine
                        "vector_threshold": 0.0,
                    },
                },
            }
        )

    def _seed_session(self, session_id: str, topic: str, chunk_key: str = "") -> None:
        store = FileSessionStore(self.fx.workroot)
        refs = []
        if chunk_key:
            refs.append(
                ChunkRef(
                    chunk_key=chunk_key,
                    doc_key="alpha-doc",
                    rel_path="alpha.md",
                    heading_path="Alpha",
                )
            )
        store.save(
            SessionState(
                session_id=session_id,
                topic_summary=[topic],
                active_refs=refs,
                last_query="previous",
                turn_count=2,
                created_unix=1.0,
                updated_unix=2.0,
            )
        )

    def _ask(self, app, *, session_id: str):
        self.assertEqual(app.ingest_corpus().errors, [])
        embed = sync_embeddings(
            app_config=app.config,
            security_root=app.roots.security_root,
            corpus_db_path=app.corpus_db_path(),
            embedder_factory=dummy_embedder_factory,
        )
        self.assertEqual(embed.errors, [])
        with patch("agent.app.create_embedder", side_effect=dummy_embedder_factory):
            seed = app.retrieve("alpha")
            self.assertGreaterEqual(len(seed.candidates), 1)
            chunk = seed.candidates[0]
            answer = (
                f"alpha evidence lives here. "
                f"[source: {chunk.rel_path}#{chunk.heading_path} | {chunk.chunk_key}]"
            )
            with patch("agent.grounding.ensure_ollama_up"), \
                 patch("agent.grounding.create_embedder", side_effect=dummy_embedder_factory), \
                 patch("agent.grounding.ollama_chat", return_value={"message": {"content": answer}}):
                return app.answer_grounded("alpha?", session_id=session_id)

    def test_memory_text_never_in_evidence_even_when_rewrite_on(self) -> None:
        leak_kw = "INJECTED_MEMORY_TOKEN_ZQXP"
        app = self._build_app(memory_rewrite_enabled=True)
        self._seed_session("leaky-rw", leak_kw)
        grounded = self._ask(app, session_id="leaky-rw")
        run = json.loads((grounded.run_dir / "run.json").read_text(encoding="utf-8"))

        # Evidence surface must be clean: memory keyword never appears in
        # retrieval.results entries or in the citation audit footer inputs.
        for entry in run["retrieval"]["results"]:
            blob = json.dumps(entry, ensure_ascii=False)
            self.assertNotIn(leak_kw, blob)

        # Memory snapshot carries the keyword (expected) — this is how we know
        # the state was in fact wired in.
        self.assertIn(leak_kw, run["memory_snapshot"]["topic_summary"])

    def test_memory_rewrite_gate_off_leaves_rewritten_query_clean(self) -> None:
        leak_kw = "INJECTED_MEMORY_TOKEN_ZQXP"
        app = self._build_app(memory_rewrite_enabled=False)
        self._seed_session("gated-off", leak_kw)
        grounded = self._ask(app, session_id="gated-off")
        run = json.loads((grounded.run_dir / "run.json").read_text(encoding="utf-8"))
        # When the gate is off, the keyword must not appear anywhere in the
        # retrieval_rounds surface either.
        rounds_blob = json.dumps(run.get("retrieval_rounds") or [], ensure_ascii=False)
        self.assertNotIn(leak_kw, rounds_blob)


if __name__ == "__main__":
    unittest.main()
