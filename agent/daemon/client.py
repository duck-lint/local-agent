"""Fail-fast client for the local-loopback session-memory daemon."""

from __future__ import annotations

import json
import socket
from typing import Optional

from agent.daemon.protocol import (
    METHOD_PING,
    METHOD_SESSION_CLEAR,
    METHOD_SESSION_GET,
    METHOD_SESSION_LIST,
    METHOD_SESSION_SHOW,
    METHOD_SESSION_UPDATE,
    METHOD_SHUTDOWN,
    make_request,
)
from agent.session_memory import SessionState


class DaemonError(Exception):
    """Base class for daemon client errors."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class DaemonUnreachableError(DaemonError):
    """Raised when the loopback socket cannot be reached."""

    def __init__(self, message: str) -> None:
        super().__init__("DAEMON_UNREACHABLE", message)


class DaemonProtocolError(DaemonError):
    """Raised when the response is malformed or signals a server-side error."""


class DaemonClient:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int,
        timeout_s: float = 5.0,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)

    # ------------------------------------------------------------------ raw
    def _round_trip(self, method: str, params: Optional[dict] = None) -> dict:
        request = make_request(method, params or {}, request_id="1")
        payload = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self._host, self._port), timeout=self._timeout_s) as sock:
                sock.settimeout(self._timeout_s)
                sock.sendall(payload)
                buf = bytearray()
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if b"\n" in chunk:
                        break
        except (ConnectionRefusedError, OSError, socket.timeout) as exc:
            raise DaemonUnreachableError(
                f"could not reach daemon at {self._host}:{self._port}: {exc}"
            ) from exc
        line, _, _ = buf.partition(b"\n")
        if not line:
            raise DaemonProtocolError("EMPTY_RESPONSE", "daemon returned no data")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DaemonProtocolError("DECODE_ERROR", str(exc)) from exc
        if not isinstance(response, dict):
            raise DaemonProtocolError("BAD_RESPONSE", "response was not a JSON object")
        if not response.get("ok"):
            err = response.get("error") or {}
            raise DaemonProtocolError(
                str(err.get("code", "UNKNOWN")),
                str(err.get("message", "unknown daemon error")),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise DaemonProtocolError("BAD_RESULT", "result was not a JSON object")
        return result

    # ------------------------------------------------------------------ api
    def ping(self) -> bool:
        result = self._round_trip(METHOD_PING)
        return bool(result.get("pong"))

    def shutdown(self) -> bool:
        try:
            result = self._round_trip(METHOD_SHUTDOWN)
        except DaemonUnreachableError:
            # Already down — treat as success for idempotency.
            return True
        return bool(result.get("shutting_down"))

    def session_get(self, session_id: str) -> SessionState:
        result = self._round_trip(METHOD_SESSION_GET, {"session_id": session_id})
        state_raw = result.get("state") or {}
        if not isinstance(state_raw, dict):
            raise DaemonProtocolError("BAD_RESULT", "state was not a JSON object")
        return SessionState.from_dict(state_raw)

    def session_show(self, session_id: str) -> SessionState:
        result = self._round_trip(METHOD_SESSION_SHOW, {"session_id": session_id})
        return SessionState.from_dict(result.get("state") or {})

    def session_update(self, state: SessionState) -> str:
        result = self._round_trip(METHOD_SESSION_UPDATE, {"state": state.to_dict()})
        return str(result.get("session_id", state.session_id))

    def session_list(self) -> list[str]:
        result = self._round_trip(METHOD_SESSION_LIST)
        ids = result.get("session_ids") or []
        return [str(x) for x in ids]

    def session_clear(self, session_id: str) -> bool:
        result = self._round_trip(METHOD_SESSION_CLEAR, {"session_id": session_id})
        return bool(result.get("deleted"))
