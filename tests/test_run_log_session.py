"""Integration: --session wiring through grounding into the run log."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.embeddings import sync_embeddings
from agent.session_memory import FileSessionStore
from tests.support import AppFixture, dummy_embedder_factory


def _seed_corpus(fx: AppFixture) -> None:
    fx.write_corpus_note(
        "alpha.md",
        "---\nuuid: alpha-doc\ntitle: Alpha Doc\n---\n\n## Alpha\nalpha evidence lives here.\n",
    )


class SessionRunLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        _seed_corpus(self.fx)

    def tearDown(self) -> None:
        self.fx.close()

    def _build_app(self, *, session_enabled: bool):
        app = self.fx.build_app(
            config_override={
                "session": {
                    "enabled": session_enabled,
                    "topic_summary_top_k": 5,
                    "max_active_refs": 3,
                    "max_bundle_ids": 3,
                    "require_daemon_for_cli": False,
                },
                "daemon": {"enabled": False},
            }
        )
        self.assertEqual(app.ingest_corpus().errors, [])
        embed = sync_embeddings(
            app_config=app.config,
            security_root=app.roots.security_root,
            corpus_db_path=app.corpus_db_path(),
            embedder_factory=dummy_embedder_factory,
        )
        self.assertEqual(embed.errors, [])
        return app

    def _ask(self, app, *, session_id=None):
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
                return app.answer_grounded("Where is alpha evidence?", session_id=session_id)

    def _read_run(self, grounded) -> dict:
        return json.loads((grounded.run_dir / "run.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- tests

    def test_session_disabled_silently_ignores_session_id(self) -> None:
        app = self._build_app(session_enabled=False)
        grounded = self._ask(app, session_id="some-id")
        self.assertTrue(grounded.ok)
        run = self._read_run(grounded)
        self.assertIsNone(run["memory_snapshot"])
        self.assertNotIn("session_state_after", run)

    def test_first_query_in_session_equals_no_session_baseline(self) -> None:
        """Cold-start parity: enabling session memory must not affect retrieval shape."""
        app_off = self._build_app(session_enabled=False)
        baseline = self._read_run(self._ask(app_off, session_id=None))

        app_on = self._build_app(session_enabled=True)
        first = self._read_run(self._ask(app_on, session_id="cold-start-1"))

        # Memory snapshot exists on the first turn but its fields are all empty/zero.
        snap = first["memory_snapshot"]
        self.assertIsNotNone(snap)
        self.assertEqual(snap["topic_summary"], [])
        self.assertEqual(snap["active_refs"], [])
        self.assertEqual(snap["last_evidence_bundle_ids"], [])
        self.assertEqual(snap["turn_count"], 0)

        # Retrieval surface is byte-identical.
        self.assertEqual(first["retrieval"]["candidates_count"], baseline["retrieval"]["candidates_count"])
        self.assertEqual(first["retrieval"]["results"], baseline["retrieval"]["results"])

    def test_no_memory_text_in_evidence_snapshot(self) -> None:
        """The evidence list must never contain memory-side fields."""
        app = self._build_app(session_enabled=True)
        # Pre-seed a session state so memory_snapshot is non-trivial.
        store = FileSessionStore(self.fx.workroot)
        from agent.session_memory import SessionState

        store.save(
            SessionState(
                session_id="leaky",
                topic_summary=["INJECTED_KEYWORD_DO_NOT_LEAK"],
                last_query="prior question",
                turn_count=3,
                created_unix=1.0,
                updated_unix=2.0,
            )
        )
        grounded = self._ask(app, session_id="leaky")
        run = self._read_run(grounded)
        # Snapshot is captured...
        self.assertEqual(run["memory_snapshot"]["topic_summary"], ["INJECTED_KEYWORD_DO_NOT_LEAK"])
        # ...but memory text never enters retrieval.results entries.
        for entry in run["retrieval"]["results"]:
            blob = json.dumps(entry, ensure_ascii=False)
            self.assertNotIn("INJECTED_KEYWORD_DO_NOT_LEAK", blob)

    def test_state_persisted_after_answer(self) -> None:
        app = self._build_app(session_enabled=True)
        grounded = self._ask(app, session_id="persist-1")
        run = self._read_run(grounded)
        self.assertEqual(run["memory_snapshot"]["turn_count"], 0)
        # New state was written and reflected in record.
        after = run["session_state_after"]
        self.assertEqual(after["turn_count"], 1)
        self.assertEqual(after["session_id"], "persist-1")
        self.assertGreaterEqual(len(after["active_refs"]), 1)
        self.assertEqual(len(after["last_evidence_bundle_ids"]), 1)
        # On disk too.
        store = FileSessionStore(self.fx.workroot)
        loaded = store.get("persist-1")
        self.assertEqual(loaded.turn_count, 1)
        self.assertEqual(loaded.session_id, "persist-1")


if __name__ == "__main__":
    unittest.main()
