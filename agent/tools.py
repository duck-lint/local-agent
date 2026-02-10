from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict


class ToolError(Exception):
    """Raised when a tool is called with invalid args or cannot complete safely."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: Dict[str, Any]  # simple JSON-schema-ish dict
    func: Callable[[Dict[str, Any]], Dict[str, Any]]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _read_text_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read a UTF-8-ish text file from disk.

    Args:
      path (str): required
      max_chars (int): optional, default 12000. Range: [200, 200000]

    Returns:
      path, sha256, chars_full, chars_returned, truncated, text
    """
    path = args.get("path")
    if not path or not isinstance(path, str):
        raise ToolError("read_text_file requires args.path as a non-empty string")

    max_chars = args.get("max_chars", 12000)
    if not isinstance(max_chars, int) or max_chars < 200 or max_chars > 200_000:
        raise ToolError("read_text_file args.max_chars must be an int in [200, 200000]")

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ToolError(f"File does not exist: {p}")
    if p.is_dir():
        raise ToolError(f"Path is a directory, expected a file: {p}")

    text_full = p.read_text(encoding="utf-8", errors="replace")
    digest = _sha256_text(text_full)

    truncated = False
    text = text_full
    if len(text_full) > max_chars:
        truncated = True
        text = text_full[:max_chars]

    return {
        "path": str(p),
        "sha256": digest,
        "chars_full": len(text_full),
        "chars_returned": len(text),
        "truncated": truncated,
        "text": text,
    }


TOOLS: Dict[str, ToolSpec] = {
    "read_text_file": ToolSpec(
        name="read_text_file",
        description="Read a text file from disk and return its contents (optionally truncated).",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        func=_read_text_file,
    ),
}
