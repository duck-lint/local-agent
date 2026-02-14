from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


@dataclass
class ToolCallParse:
    tool_call: ToolCall
    trailing_text: str


def _to_tool_call(obj: Any) -> Optional[ToolCall]:
    if not isinstance(obj, dict):
        return None
    if obj.get("type") != "tool_call":
        return None

    name = obj.get("name")
    args = obj.get("args")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(args, dict):
        return None
    return ToolCall(name=name, args=args)


def _find_first_json_object_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Returns (start, end) indices for the first complete JSON object encountered
    while scanning left-to-right.
    """
    start = -1
    n = len(text)

    depth = 0
    in_string = False
    escape = False

    for j in range(n):
        ch = text[j]
        if start < 0:
            if ch == "{":
                start = j
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (start, j)
            if depth < 0:
                return None
    return None


def try_parse_tool_call(text: str) -> Optional[ToolCallParse]:
    """
    Parse tool call JSON from assistant text.

    1) Strict mode: the entire response is one JSON object.
    2) Prefix mode: the response starts with a complete JSON object, followed by
       optional trailing text.
    """
    candidate = text.strip()

    # Strict parse: entire response must be one JSON object.
    try:
        strict_obj = json.loads(candidate)
    except json.JSONDecodeError:
        strict_obj = None

    strict_tool_call = _to_tool_call(strict_obj)
    if strict_tool_call is not None:
        return ToolCallParse(tool_call=strict_tool_call, trailing_text="")

    # Prefix parse: first object at start of response may be followed by text.
    span = _find_first_json_object_span(text)
    if span is None:
        return None

    start_idx, end_idx = span
    if text[:start_idx].strip():
        return None
    prefix = text[start_idx : end_idx + 1]
    try:
        prefix_obj = json.loads(prefix)
    except json.JSONDecodeError:
        return None

    prefix_tool_call = _to_tool_call(prefix_obj)
    if prefix_tool_call is None:
        return None

    trailing = text[end_idx + 1 :].strip()
    return ToolCallParse(tool_call=prefix_tool_call, trailing_text=trailing)
