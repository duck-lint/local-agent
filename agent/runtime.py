from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from agent.app_types import AppConfig


def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir(security_root: Path) -> Path:
    base = security_root.resolve() / "runs"
    base.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    run_dir = base / run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    for index in range(1, 1000):
        candidate = base / f"{run_id}_{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError("Unable to allocate a unique run directory")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def now_unix() -> float:
    return time.time()


def print_output(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()


def strip_thinking(resp: dict[str, Any] | None) -> dict[str, Any] | None:
    if resp is None:
        return None
    out = dict(resp)
    message = out.get("message")
    if isinstance(message, dict) and "thinking" in message:
        cleaned = dict(message)
        cleaned.pop("thinking", None)
        out["message"] = cleaned
    return out


def select_models(
    app_config: AppConfig,
    question: str,
    *,
    force_big_second: bool = False,
    force_fast: bool = False,
) -> tuple[str, str]:
    base_model = app_config.model
    model_fast = app_config.model_fast or base_model
    model_big = app_config.model_big or base_model
    q_lower = question.lower()
    wants_big = any(trigger in q_lower for trigger in app_config.big_triggers)

    if app_config.prefer_fast:
        first_model = model_fast
        second_model = model_big if wants_big else model_fast
    else:
        first_model = model_big
        second_model = model_big

    if force_fast:
        return model_fast, model_fast
    if force_big_second:
        return first_model, model_big
    return first_model, second_model


def render_query_results(rows: list[dict[str, Any]], query_text: str) -> str:
    if not rows:
        return f"No chunks matched query: {query_text}"
    rendered: list[str] = []
    for index, row in enumerate(rows, start=1):
        rendered.append(f"[{index}] {row.get('rel_path', '')}")
        heading_path = str(row.get("heading_path") or "").strip()
        if heading_path:
            rendered.append(f"heading_path: {heading_path}")
        chunk_title = str(row.get("chunk_title") or "").strip()
        if chunk_title:
            rendered.append(f"chunk_title: {chunk_title}")
        rendered.append(str(row.get("chunk_text") or row.get("text") or ""))
        rendered.append("")
    return "\n".join(rendered).rstrip()


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[source:\s+[^\]|]+\|\s*[0-9a-f]{32}\]", answer))
