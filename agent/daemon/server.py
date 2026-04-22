"""Loopback TCP daemon hosting the session-memory store.

Threaded: each connection handled in its own thread. The handler decodes a
single JSON line, dispatches by ``method``, and writes a single JSON line
response. The store is mediated by an internal lock for safety even though
``FileSessionStore`` is itself thread-safe.

Idle-timeout shutdown: a watcher thread observes ``last_activity`` and shuts
the server down if no request has arrived within ``idle_timeout_s`` seconds
(set ``idle_timeout_s <= 0`` to disable).
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Optional

from agent.daemon.protocol import (
    ALL_METHODS,
    METHOD_PING,
    METHOD_SESSION_CLEAR,
    METHOD_SESSION_GET,
    METHOD_SESSION_LIST,
    METHOD_SESSION_SHOW,
    METHOD_SESSION_UPDATE,
    METHOD_SHUTDOWN,
    PROTOCOL_VERSION,
    make_error,
    make_ok,
)
from agent.session_memory import (
    ChunkRef,
    FileSessionStore,
    SessionState,
    SessionStoreError,
    validate_session_id,
)


class _Handler(socketserver.StreamRequestHandler):
    server: "_ThreadedTCPServer"  # type: ignore[assignment]

    def handle(self) -> None:  # noqa: D401
        line = self.rfile.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8").strip())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._write(make_error("?", "PROTOCOL_DECODE_ERROR", str(exc)))
            return
        if not isinstance(request, dict):
            self._write(make_error("?", "PROTOCOL_BAD_REQUEST", "request must be a JSON object"))
            return
        request_id = str(request.get("id", "?"))
        method = str(request.get("method", ""))
        params = request.get("params") or {}
        if not isinstance(params, dict):
            self._write(make_error(request_id, "PROTOCOL_BAD_PARAMS", "params must be an object"))
            return
        if method not in ALL_METHODS:
            self._write(make_error(request_id, "UNKNOWN_METHOD", f"unknown method {method!r}"))
            return
        try:
            response = self.server.dispatch(request_id, method, params)
        except Exception as exc:  # noqa: BLE001 — surface as protocol error
            response = make_error(request_id, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
        self._write(response)

    def _write(self, payload: dict) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        store: FileSessionStore,
        *,
        idle_timeout_s: int,
    ) -> None:
        super().__init__(address, _Handler)
        self.store = store
        self.idle_timeout_s = int(idle_timeout_s)
        self._lock = threading.Lock()
        self.last_activity = time.monotonic()

    def dispatch(self, request_id: str, method: str, params: dict) -> dict:
        with self._lock:
            self.last_activity = time.monotonic()

        if method == METHOD_PING:
            return make_ok(request_id, {"pong": True, "v": PROTOCOL_VERSION})

        if method == METHOD_SHUTDOWN:
            # Trigger asynchronous shutdown so we can still write the response.
            threading.Thread(target=self.shutdown, daemon=True).start()
            return make_ok(request_id, {"shutting_down": True})

        if method == METHOD_SESSION_GET or method == METHOD_SESSION_SHOW:
            session_id = params.get("session_id")
            if not isinstance(session_id, str):
                return make_error(request_id, "BAD_PARAM", "session_id must be a string")
            try:
                validate_session_id(session_id)
            except ValueError as exc:
                return make_error(request_id, "BAD_PARAM", str(exc))
            try:
                state = self.store.get(session_id)
            except SessionStoreError as exc:
                return make_error(request_id, "STORE_ERROR", str(exc))
            return make_ok(request_id, {"state": state.to_dict()})

        if method == METHOD_SESSION_UPDATE:
            state_raw = params.get("state")
            if not isinstance(state_raw, dict):
                return make_error(request_id, "BAD_PARAM", "state must be an object")
            try:
                state = SessionState.from_dict(state_raw)
                validate_session_id(state.session_id)
            except (ValueError, TypeError) as exc:
                return make_error(request_id, "BAD_PARAM", str(exc))
            try:
                self.store.save(state)
            except SessionStoreError as exc:
                return make_error(request_id, "STORE_ERROR", str(exc))
            return make_ok(request_id, {"saved": True, "session_id": state.session_id})

        if method == METHOD_SESSION_LIST:
            return make_ok(request_id, {"session_ids": self.store.list()})

        if method == METHOD_SESSION_CLEAR:
            session_id = params.get("session_id")
            if not isinstance(session_id, str):
                return make_error(request_id, "BAD_PARAM", "session_id must be a string")
            try:
                deleted = self.store.clear(session_id)
            except (SessionStoreError, ValueError) as exc:
                return make_error(request_id, "STORE_ERROR", str(exc))
            return make_ok(request_id, {"deleted": deleted})

        return make_error(request_id, "UNKNOWN_METHOD", method)


class DaemonServer:
    """Wrapper around the threaded TCP server.

    Lifecycle: ``serve_forever_in_thread()`` returns once the listening socket
    is bound. ``shutdown()`` performs orderly shutdown. ``wait_until_stopped()``
    blocks a foreground wrapper until the server loop actually exits.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        store: FileSessionStore,
        idle_timeout_s: int = 0,
    ) -> None:
        self._server = _ThreadedTCPServer((host, port), store, idle_timeout_s=idle_timeout_s)
        self._serve_thread: Optional[threading.Thread] = None
        self._idle_thread: Optional[threading.Thread] = None
        self._idle_stop_event = threading.Event()
        self._stopped_event = threading.Event()

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address  # type: ignore[return-value]

    def serve_forever_in_thread(self) -> None:
        if self._serve_thread is not None:
            return
        self._stopped_event.clear()

        def _serve() -> None:
            try:
                self._server.serve_forever()
            finally:
                self._stopped_event.set()

        self._serve_thread = threading.Thread(
            target=_serve,
            name="local-agent-daemon-server",
            daemon=True,
        )
        self._serve_thread.start()
        if self._server.idle_timeout_s > 0:
            self._idle_thread = threading.Thread(
                target=self._idle_watcher,
                name="local-agent-daemon-idle",
                daemon=True,
            )
            self._idle_thread.start()

    def _idle_watcher(self) -> None:
        timeout = self._server.idle_timeout_s
        while not self._idle_stop_event.is_set():
            self._idle_stop_event.wait(timeout=min(timeout, 5))
            if self._idle_stop_event.is_set():
                return
            since = time.monotonic() - self._server.last_activity
            if since >= timeout:
                self.shutdown()
                return

    def shutdown(self) -> None:
        self._idle_stop_event.set()
        try:
            self._server.shutdown()
        except Exception:  # pragma: no cover - already stopping
            pass
        try:
            self._server.server_close()
        except Exception:  # pragma: no cover
            pass

    def wait_until_stopped(self, timeout_s: Optional[float] = None) -> bool:
        if self._serve_thread is None:
            return True
        timeout = None if timeout_s is None else float(timeout_s)
        return self._stopped_event.wait(timeout=timeout)
