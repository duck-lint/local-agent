from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ToolError(Exception):
    """Raised when a tool is called with invalid args or cannot complete safely."""

    def __init__(self, code: str = "TOOL_ERROR", message: Optional[str] = None):
        # Backward-compatible: ToolError("message") keeps default code.
        if message is None:
            message = code
            code = "TOOL_ERROR"
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: Dict[str, Any]  # simple JSON-schema-ish dict
    func: Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class ReadTextFilePolicy:
    allowed_roots: list[Path]
    allowed_exts: set[str]
    deny_absolute_paths: bool
    deny_hidden_paths: bool
    allow_any_path: bool


DEFAULT_ALLOWED_EXTS = {".md", ".txt", ".json"}
_READ_TEXT_FILE_POLICY = ReadTextFilePolicy(
    allowed_roots=[],
    allowed_exts=set(DEFAULT_ALLOWED_EXTS),
    deny_absolute_paths=True,
    deny_hidden_paths=True,
    allow_any_path=False,
)
_WORKSPACE_ROOT = Path.cwd().resolve()


def _normalize_ext(ext: str) -> str:
    out = ext.strip().lower()
    if not out:
        return out
    return out if out.startswith(".") else f".{out}"


def _parse_allowed_exts(raw: Any) -> set[str]:
    if not isinstance(raw, list):
        return set(DEFAULT_ALLOWED_EXTS)
    exts = {_normalize_ext(str(x)) for x in raw if str(x).strip()}
    exts = {e for e in exts if e}
    return exts or set(DEFAULT_ALLOWED_EXTS)


def _has_windows_anchor(path_text: str) -> bool:
    s = path_text.strip()
    # Drive letter or UNC path.
    return bool(re.match(r"^[a-zA-Z]:", s)) or s.startswith("\\\\") or s.startswith("//")


def _is_within(candidate: Path, base: Path) -> bool:
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


def _validate_existing_file(path: Path) -> Path:
    if not path.exists():
        raise ToolError("FILE_NOT_FOUND", f"File does not exist: {path}")
    if path.is_dir():
        raise ToolError("PATH_IS_DIRECTORY", f"Path is a directory, expected a file: {path}")
    return path


def configure_tool_security(policy_dict: dict, workspace_root: Path) -> None:
    global _READ_TEXT_FILE_POLICY, _WORKSPACE_ROOT

    raw = policy_dict if isinstance(policy_dict, dict) else {}
    workspace = workspace_root.resolve()
    _WORKSPACE_ROOT = workspace
    auto_create_allowed_roots = bool(raw.get("auto_create_allowed_roots", False))
    roots_must_be_within_workspace = bool(raw.get("roots_must_be_within_workspace", False))

    roots_raw = raw.get("allowed_roots", [])
    if not isinstance(roots_raw, list):
        roots_raw = []

    roots: list[Path] = []
    for item in roots_raw:
        txt = str(item).strip()
        if not txt:
            continue
        p = Path(txt).expanduser()
        if not p.is_absolute():
            p = workspace / p

        p_abs = Path(os.path.abspath(str(p)))
        if roots_must_be_within_workspace and not _is_within(p_abs, workspace):
            continue

        if auto_create_allowed_roots:
            try:
                p_abs.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue

        try:
            r = p_abs.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if r.exists() and r.is_dir():
            roots.append(r)

    if not roots:
        raise ValueError("No valid allowed_roots. Create corpus/ runs/ scratch/ or update configs/default.yaml.")

    _READ_TEXT_FILE_POLICY = ReadTextFilePolicy(
        allowed_roots=roots,
        allowed_exts=_parse_allowed_exts(raw.get("allowed_exts")),
        deny_absolute_paths=bool(raw.get("deny_absolute_paths", True)),
        deny_hidden_paths=bool(raw.get("deny_hidden_paths", True)),
        allow_any_path=bool(raw.get("allow_any_path", False)),
    )


def resolve_and_validate_path(requested: str, policy: ReadTextFilePolicy) -> Path:
    requested_path = Path(requested).expanduser()
    requested_text = str(requested_path)
    has_path_separator = ("/" in requested) or ("\\" in requested)

    if policy.allow_any_path:
        try:
            candidate = Path(os.path.abspath(str(requested_path)))
        except OSError as exc:
            raise ToolError("PATH_DENIED", f"Invalid path: {requested}") from exc
        return _validate_existing_file(candidate)

    if policy.deny_absolute_paths and (requested_path.is_absolute() or _has_windows_anchor(requested_text)):
        raise ToolError("PATH_DENIED", f"Absolute or anchored paths are denied by policy: {requested}")

    resolved_roots: list[Path] = []
    for base in policy.allowed_roots:
        try:
            resolved_roots.append(base.resolve(strict=True))
        except FileNotFoundError:
            continue

    if has_path_separator:
        # Explicit subpath mode: interpret request as workspace-relative path and
        # validate it is contained in at least one allowlisted root.
        try:
            if requested_path.is_absolute() or _has_windows_anchor(requested_text):
                candidate_abs = Path(os.path.abspath(str(requested_path)))
            else:
                candidate_abs = Path(os.path.abspath(str(_WORKSPACE_ROOT / requested_path)))
        except OSError as exc:
            raise ToolError("PATH_DENIED", f"Invalid path: {requested}") from exc

        file_missing_inside_root = False
        for base_resolved in resolved_roots:
            if not _is_within(candidate_abs, base_resolved):
                continue

            suffix = candidate_abs.suffix.lower()
            if policy.allowed_exts and suffix not in policy.allowed_exts:
                shown = suffix if suffix else "<none>"
                raise ToolError("PATH_DENIED", f"Path extension '{shown}' is denied by policy.")

            if policy.deny_hidden_paths:
                rel = candidate_abs.relative_to(base_resolved)
                if any(part.startswith(".") for part in rel.parts):
                    raise ToolError("PATH_DENIED", "Hidden paths are denied by policy.")

            if not candidate_abs.exists():
                file_missing_inside_root = True
                continue

            try:
                candidate = candidate_abs.resolve(strict=True)
            except (FileNotFoundError, OSError):
                file_missing_inside_root = True
                continue
            if not _is_within(candidate, base_resolved):
                continue

            return _validate_existing_file(candidate)

        if file_missing_inside_root:
            raise ToolError("FILE_NOT_FOUND", f"File does not exist under allowed roots: {requested}")
        raise ToolError("PATH_DENIED", f"Path escapes allowed roots or is denied by policy: {requested}")

    # Bare filename mode: search each allowed root and detect ambiguous matches.
    file_missing_inside_root = False
    lexical_containment_seen = False
    matches: list[Path] = []
    for base_resolved in resolved_roots:
        tentative = base_resolved / requested_path

        try:
            tentative_abs = Path(os.path.abspath(str(tentative)))
        except OSError as exc:
            raise ToolError("PATH_DENIED", f"Invalid path: {requested}") from exc

        if not _is_within(tentative_abs, base_resolved):
            continue
        lexical_containment_seen = True

        suffix = tentative_abs.suffix.lower()
        if policy.allowed_exts and suffix not in policy.allowed_exts:
            shown = suffix if suffix else "<none>"
            raise ToolError("PATH_DENIED", f"Path extension '{shown}' is denied by policy.")

        if policy.deny_hidden_paths:
            rel = tentative_abs.relative_to(base_resolved)
            if any(part.startswith(".") for part in rel.parts):
                raise ToolError("PATH_DENIED", "Hidden paths are denied by policy.")

        if not tentative_abs.exists():
            file_missing_inside_root = True
            continue

        try:
            candidate = tentative_abs.resolve(strict=True)
        except (FileNotFoundError, OSError):
            file_missing_inside_root = True
            continue
        if not _is_within(candidate, base_resolved):
            continue

        matches.append(_validate_existing_file(candidate))

    unique_matches = sorted({str(p): p for p in matches}.values(), key=lambda p: str(p))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        listed = ", ".join(str(p) for p in unique_matches)
        raise ToolError("AMBIGUOUS_PATH", f"Multiple matches for '{requested}': {listed}")
    if file_missing_inside_root or lexical_containment_seen:
        raise ToolError("FILE_NOT_FOUND", f"File does not exist under allowed roots: {requested}")
    raise ToolError("PATH_DENIED", f"Path escapes allowed roots or is denied by policy: {requested}")


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
        raise ToolError("INVALID_ARGS", "read_text_file requires args.path as a non-empty string")

    max_chars = args.get("max_chars", 12000)
    if not isinstance(max_chars, int) or max_chars < 200 or max_chars > 200_000:
        raise ToolError("INVALID_ARGS", "read_text_file args.max_chars must be an int in [200, 200000]")

    p = resolve_and_validate_path(path, _READ_TEXT_FILE_POLICY)

    try:
        text_full = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError("FILE_READ_ERROR", f"Failed to read file: {p}") from exc

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
