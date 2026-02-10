from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


def try_parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Attempts to parse a tool call from the assistant text.
    Requires the entire response to be a single JSON object with type=tool_call.
    """
    candidate = text.strip()

    # quick reject
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None
    if obj.get("type") != "tool_call":
        return None
    name = obj.get("name")
    args = obj.get("args")

    if not isinstance(name, str) or not isinstance(args, dict):
        return None

    return ToolCall(name=name, args=args)
