"""Wire format for the local-loopback session-memory daemon.

Frame: a single line of JSON terminated by ``\n``. Both directions.

Request shape:
    {"v": 1, "method": "<method>", "params": {...}, "id": "<opaque>"}

Response shape on success:
    {"v": 1, "id": "<opaque>", "ok": true, "result": {...}}

Response shape on failure:
    {"v": 1, "id": "<opaque>", "ok": false, "error": {"code": "...", "message": "..."}}
"""

from __future__ import annotations

PROTOCOL_VERSION = 1

METHOD_PING = "ping"
METHOD_SHUTDOWN = "shutdown"
METHOD_SESSION_GET = "session.get"
METHOD_SESSION_UPDATE = "session.update"
METHOD_SESSION_LIST = "session.list"
METHOD_SESSION_CLEAR = "session.clear"
METHOD_SESSION_SHOW = "session.show"

ALL_METHODS = frozenset({
    METHOD_PING,
    METHOD_SHUTDOWN,
    METHOD_SESSION_GET,
    METHOD_SESSION_UPDATE,
    METHOD_SESSION_LIST,
    METHOD_SESSION_CLEAR,
    METHOD_SESSION_SHOW,
})


def make_request(method: str, params: dict | None = None, request_id: str = "1") -> dict:
    return {
        "v": PROTOCOL_VERSION,
        "method": method,
        "params": params or {},
        "id": request_id,
    }


def make_ok(request_id: str, result: dict) -> dict:
    return {"v": PROTOCOL_VERSION, "id": request_id, "ok": True, "result": result}


def make_error(request_id: str, code: str, message: str) -> dict:
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "ok": False,
        "error": {"code": code, "message": message},
    }
