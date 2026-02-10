from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: Dict[str, Any]  # simple JSON-schema-ish dict
    func: Callable[[Dict[str, Any]], Dict[str, Any]]


def _read_text_file(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path")
    if not path or not isinstance(path, str):
        raise ToolError("read_text_file requires args.path as a non-empty string")

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ToolError(f"File does not exist: {p}")
    if p.is_dir():
        raise ToolError(f"Path is a directory, expected a file: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(p),
        "chars": len(text),
        "text": text,
    }


TOOLS: Dict[str, ToolSpec] = {
    "read_text_file": ToolSpec(
        name="read_text_file",
        description="Read a UTF-8 text file from disk and return its contents.",
        args_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        func=_read_text_file,
    )
}
