"""Phase 4: promotion CLI + memory_db sidecar provenance."""

from __future__ import annotations

import argparse
import json
import unittest

from agent.corpus_db import connect_db as connect_corpus_db, list_chunk_keys
from agent.memory_db import (
    add_promoted_memory,
    connect_db as connect_memory_db,
    get_promotion_provenance,
    init_db as init_memory_db,
    list_promotions_for_session,
)
from agent.session_memory import ChunkRef, FileSessionStore, SessionState
from agent.cli import _handle_memory_promote, _handle_memory_suggest
from tests.support import AppFixture


def _seed_corpus(fx: AppFixture) -> None:
    fx.write_corpus_note(
        "alpha.md",
        "---\nuuid: alpha-doc\ntitle: Alpha Doc\n---\n\n## Alpha\nalpha evidence lives here.\n",
    )


def _seed_session(app, chunk_key: str, doc_key: str) -> str:
    """Persist a SessionState with one active_ref; return session_id."""
    store = FileSessionStore(app.roots.workroot)
    state = SessionState(
        session_id="s1",
        topic_summary=["alpha"],
        active_refs=[
            ChunkRef(
                chunk_key=chunk_key,
                doc_key=doc_key,
                rel_path="alpha.md",
                heading_path="Alpha",
            )
        ],
        last_evidence_bundle_ids=["bundle:run1"],
        last_query="alpha?",
        turn_count=1,
    )
    store.save(state)
    return state.session_id


def _ns(**kwargs):
    ns = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


class PromotionDbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        _seed_corpus(self.fx)

    def tearDown(self) -> None:
        self.fx.close()

    def test_add_promoted_memory_writes_full_provenance(self) -> None:
        app = self.fx.build_app()
        self.assertEqual(app.ingest_corpus().errors, [])
        memory_db_path = app.memory_db_path()
        init_memory_db(memory_db_path)
        with connect_corpus_db(app.corpus_db_path()) as cc:
            chunk_keys = list_chunk_keys(cc)
        self.assertGreater(len(chunk_keys), 0)
        first_key = chunk_keys[0]

        with connect_memory_db(memory_db_path) as conn:
            memory_id = add_promoted_memory(
                conn,
                memory_type="user_fact",
                content="User prefers alpha.",
                chunk_keys=[first_key],
                session_id="s1",
                triggering_query_ids=["q1", "q2"],
                evidence_bundle_ids=["bundle:run1"],
                promoted_by="user",
                payload={"topic_summary": ["alpha"]},
                allowed_chunk_keys=chunk_keys,
            )
            conn.commit()

        with connect_memory_db(memory_db_path) as conn:
            prov = get_promotion_provenance(conn, memory_id)
        self.assertIsNotNone(prov)
        self.assertEqual(prov["session_id"], "s1")
        self.assertEqual(prov["promoted_by"], "user")
        self.assertEqual(prov["triggering_query_ids"], ["q1", "q2"])
        self.assertEqual(prov["evidence_bundle_ids"], ["bundle:run1"])
        self.assertEqual(prov["payload"]["topic_summary"], ["alpha"])
        self.assertGreater(prov["promoted_at"], 0.0)

    def test_add_promoted_memory_rejects_bad_promoted_by(self) -> None:
        app = self.fx.build_app()
        self.assertEqual(app.ingest_corpus().errors, [])
        memory_db_path = app.memory_db_path()
        init_memory_db(memory_db_path)
        with connect_corpus_db(app.corpus_db_path()) as cc:
            chunk_keys = list_chunk_keys(cc)
        with connect_memory_db(memory_db_path) as conn:
            with self.assertRaisesRegex(ValueError, "Unsupported promoted_by"):
                add_promoted_memory(
                    conn,
                    memory_type="user_fact",
                    content="x",
                    chunk_keys=[chunk_keys[0]],
                    session_id="s1",
                    triggering_query_ids=[],
                    evidence_bundle_ids=[],
                    promoted_by="rando",
                    allowed_chunk_keys=chunk_keys,
                )


class PromotionCLIHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        _seed_corpus(self.fx)

    def tearDown(self) -> None:
        self.fx.close()

    def _enabled_app(self, *, promotion_enabled=True, llm_suggest_enabled=False):
        return self.fx.build_app(
            config_override={
                "session": {
                    "enabled": True,
                    "require_daemon_for_cli": False,
                    "promotion": {
                        "enabled": promotion_enabled,
                        "llm_suggest_enabled": llm_suggest_enabled,
                    },
                },
                "daemon": {"enabled": False},
            }
        )

    def test_promotion_disabled_blocks_writes(self) -> None:
        app = self._enabled_app(promotion_enabled=False)
        self.assertEqual(app.ingest_corpus().errors, [])
        with connect_corpus_db(app.corpus_db_path()) as cc:
            keys = list_chunk_keys(cc)
        sid = _seed_session(app, chunk_key=keys[0], doc_key="alpha-doc")
        args = _ns(
            session_id=sid,
            refs=[keys[0]],
            type="user_fact",
            content="x",
            llm_suggest=False,
            yes=False,
            json=True,
        )
        rc = _handle_memory_promote(app, args)
        self.assertNotEqual(rc, 0)
        # memory_promotion sidecar must be empty.
        init_memory_db(app.memory_db_path())
        with connect_memory_db(app.memory_db_path()) as conn:
            self.assertEqual(list_promotions_for_session(conn, sid), [])

    def test_explicit_promotion_writes_row(self) -> None:
        app = self._enabled_app()
        self.assertEqual(app.ingest_corpus().errors, [])
        with connect_corpus_db(app.corpus_db_path()) as cc:
            keys = list_chunk_keys(cc)
        sid = _seed_session(app, chunk_key=keys[0], doc_key="alpha-doc")
        args = _ns(
            session_id=sid,
            refs=[keys[0]],
            type="user_fact",
            content="Prefer alpha.",
            llm_suggest=False,
            yes=False,
            json=True,
        )
        rc = _handle_memory_promote(app, args)
        self.assertEqual(rc, 0)
        with connect_memory_db(app.memory_db_path()) as conn:
            rows = list_promotions_for_session(conn, sid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["promoted_by"], "user")
        self.assertEqual(rows[0]["evidence_bundle_ids"], ["bundle:run1"])

    def test_llm_suggest_requires_confirmation(self) -> None:
        app = self._enabled_app(llm_suggest_enabled=True)
        self.assertEqual(app.ingest_corpus().errors, [])
        with connect_corpus_db(app.corpus_db_path()) as cc:
            keys = list_chunk_keys(cc)
        sid = _seed_session(app, chunk_key=keys[0], doc_key="alpha-doc")
        # Without --yes: proposal is printed, nothing is written.
        args = _ns(
            session_id=sid,
            refs=[keys[0]],
            type="user_fact",
            content=None,
            llm_suggest=True,
            yes=False,
            json=True,
        )
        rc = _handle_memory_promote(app, args)
        self.assertEqual(rc, 0)
        init_memory_db(app.memory_db_path())
        with connect_memory_db(app.memory_db_path()) as conn:
            rows_before = list_promotions_for_session(conn, sid)
        self.assertEqual(rows_before, [])

        # With --yes: write happens under the llm_suggested_user_confirmed tag.
        args.yes = True
        rc2 = _handle_memory_promote(app, args)
        self.assertEqual(rc2, 0)
        with connect_memory_db(app.memory_db_path()) as conn:
            rows_after = list_promotions_for_session(conn, sid)
        self.assertEqual(len(rows_after), 1)
        self.assertEqual(rows_after[0]["promoted_by"], "llm_suggested_user_confirmed")

    def test_llm_suggest_disabled_rejects(self) -> None:
        app = self._enabled_app(llm_suggest_enabled=False)
        self.assertEqual(app.ingest_corpus().errors, [])
        with connect_corpus_db(app.corpus_db_path()) as cc:
            keys = list_chunk_keys(cc)
        sid = _seed_session(app, chunk_key=keys[0], doc_key="alpha-doc")
        args = _ns(
            session_id=sid,
            refs=[keys[0]],
            type="user_fact",
            content=None,
            llm_suggest=True,
            yes=True,
            json=True,
        )
        rc = _handle_memory_promote(app, args)
        self.assertNotEqual(rc, 0)

    def test_reject_ref_not_in_session(self) -> None:
        app = self._enabled_app()
        self.assertEqual(app.ingest_corpus().errors, [])
        with connect_corpus_db(app.corpus_db_path()) as cc:
            keys = list_chunk_keys(cc)
        sid = _seed_session(app, chunk_key=keys[0], doc_key="alpha-doc")
        args = _ns(
            session_id=sid,
            refs=["some-other-chunk"],
            type="user_fact",
            content="x",
            llm_suggest=False,
            yes=False,
            json=True,
        )
        rc = _handle_memory_promote(app, args)
        self.assertNotEqual(rc, 0)

    def test_suggest_lists_active_refs(self) -> None:
        app = self._enabled_app()
        self.assertEqual(app.ingest_corpus().errors, [])
        with connect_corpus_db(app.corpus_db_path()) as cc:
            keys = list_chunk_keys(cc)
        sid = _seed_session(app, chunk_key=keys[0], doc_key="alpha-doc")
        args = _ns(session_id=sid, json=True)
        rc = _handle_memory_suggest(app, args)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
