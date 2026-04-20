from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


class StageDumper:
    """Opt-in per-document stage dumps for debugging the corpus pipeline.

    When `root` is None, all methods are no-ops. When set, dumps land under
    `<root>/<run_id>/stage_1_input/<rel>.input.txt` and
    `<root>/<run_id>/stage_2_chunks/<rel>.chunks.jsonl`.
    """

    def __init__(self, root: Optional[Path], run_id: str) -> None:
        self._enabled = root is not None
        self._base = (root / run_id) if root is not None else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _safe_rel(self, source_name: str, rel_path: str) -> Path:
        # Slash-only, strip leading slashes; nest by source for cross-source clarity.
        clean = rel_path.replace("\\", "/").lstrip("/")
        return Path(source_name) / clean

    def dump_stage1_input(self, *, source_name: str, rel_path: str, text: str) -> None:
        if not self._enabled or self._base is None:
            return
        target = self._base / "stage_1_input" / self._safe_rel(source_name, rel_path)
        target = target.with_suffix(target.suffix + ".input.txt") if target.suffix else target.with_name(target.name + ".input.txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def dump_stage2_chunks(self, *, source_name: str, rel_path: str, chunks: Iterable[Any]) -> None:
        if not self._enabled or self._base is None:
            return
        target = self._base / "stage_2_chunks" / self._safe_rel(source_name, rel_path)
        target = target.with_suffix(target.suffix + ".chunks.jsonl") if target.suffix else target.with_name(target.name + ".chunks.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                row = asdict(chunk) if is_dataclass(chunk) else dict(chunk)
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
