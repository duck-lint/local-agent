from __future__ import annotations

import threading
import unittest

from agent.daemon.client import DaemonClient, DaemonUnreachableError
from agent.daemon.server import DaemonServer
from agent.session_memory import (
    ChunkRef,
    FileSessionStore,
    SessionState,
)
from tests.support import AppFixture


class DaemonRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.store = FileSessionStore(self.fx.workroot)
        # bind to ephemeral port to avoid collisions across runs
        self.server = DaemonServer(host="127.0.0.1", port=0, store=self.store, idle_timeout_s=0)
        self.server.serve_forever_in_thread()
        host, port = self.server.address
        self.client = DaemonClient(host=host, port=port, timeout_s=2.0)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.fx.close()

    def test_ping(self) -> None:
        self.assertTrue(self.client.ping())

    def test_session_update_then_get_round_trip(self) -> None:
        state = SessionState(
            session_id="r1",
            topic_summary=["foo", "bar"],
            active_refs=[ChunkRef(chunk_key="k1", doc_key="d1", rel_path="a.md", heading_path="H")],
            last_evidence_bundle_ids=["bundle:run-9"],
            last_query="q",
            turn_count=2,
            created_unix=1.0,
            updated_unix=2.0,
        )
        self.client.session_update(state)
        loaded = self.client.session_get("r1")
        self.assertEqual(loaded.session_id, "r1")
        self.assertEqual(loaded.topic_summary, ["foo", "bar"])
        self.assertEqual(loaded.active_refs[0].chunk_key, "k1")
        self.assertEqual(loaded.turn_count, 2)

    def test_session_list_and_clear(self) -> None:
        self.client.session_update(SessionState(session_id="alpha"))
        self.client.session_update(SessionState(session_id="beta"))
        ids = self.client.session_list()
        self.assertIn("alpha", ids)
        self.assertIn("beta", ids)
        self.assertTrue(self.client.session_clear("alpha"))
        self.assertNotIn("alpha", self.client.session_list())

    def test_bad_session_id_rejected(self) -> None:
        with self.assertRaises(Exception):
            self.client.session_get("../etc/passwd")

    def test_shutdown_unblocks_wait_until_stopped(self) -> None:
        done = threading.Event()

        def _wait_for_shutdown() -> None:
            if self.server.wait_until_stopped(2.0):
                done.set()

        waiter = threading.Thread(target=_wait_for_shutdown)
        waiter.start()
        self.server.shutdown()
        waiter.join(timeout=2.0)

        self.assertTrue(done.is_set())

    def test_idle_timeout_unblocks_wait_until_stopped(self) -> None:
        fx = AppFixture()
        server = None
        try:
            store = FileSessionStore(fx.workroot)
            server = DaemonServer(host="127.0.0.1", port=0, store=store, idle_timeout_s=1)
            server.serve_forever_in_thread()

            self.assertTrue(server.wait_until_stopped(2.5))
        finally:
            if server is not None:
                server.shutdown()
            fx.close()


class DaemonFailFastTests(unittest.TestCase):
    def test_unreachable_raises_typed_error(self) -> None:
        # Port 1 is privileged & should refuse connections in test envs.
        client = DaemonClient(host="127.0.0.1", port=1, timeout_s=0.5)
        with self.assertRaises(DaemonUnreachableError) as ctx:
            client.ping()
        self.assertEqual(ctx.exception.code, "DAEMON_UNREACHABLE")


if __name__ == "__main__":
    unittest.main()
