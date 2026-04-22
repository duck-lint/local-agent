"""Phase 3 daemon package.

Modules:
  * ``protocol``  — request/response shapes and JSON line framing.
  * ``server``    — threaded TCP loopback server wrapping ``FileSessionStore``.
  * ``client``    — fail-fast client for CLI use.
"""

from __future__ import annotations

from agent.daemon.client import (
    DaemonClient,
    DaemonError,
    DaemonProtocolError,
    DaemonUnreachableError,
)
from agent.daemon.protocol import (
    METHOD_PING,
    METHOD_SESSION_CLEAR,
    METHOD_SESSION_GET,
    METHOD_SESSION_LIST,
    METHOD_SESSION_SHOW,
    METHOD_SESSION_UPDATE,
    METHOD_SHUTDOWN,
    PROTOCOL_VERSION,
)
from agent.daemon.server import DaemonServer

__all__ = [
    "DaemonClient",
    "DaemonError",
    "DaemonProtocolError",
    "DaemonServer",
    "DaemonUnreachableError",
    "METHOD_PING",
    "METHOD_SESSION_CLEAR",
    "METHOD_SESSION_GET",
    "METHOD_SESSION_LIST",
    "METHOD_SESSION_SHOW",
    "METHOD_SESSION_UPDATE",
    "METHOD_SHUTDOWN",
    "PROTOCOL_VERSION",
]
