from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent.stage_dump import StageDumper


@dataclass(frozen=True)
class _FakeChunk:
    chunk_key: str
    text: str


def test_stage_dumper_disabled_is_noop(tmp_path: Path):
    dumper = StageDumper(None, "20260101_000000")
    assert dumper.enabled is False
    dumper.dump_stage1_input(source_name="vault", rel_path="a.md", text="hi")
    dumper.dump_stage2_chunks(source_name="vault", rel_path="a.md", chunks=[])
    assert list(tmp_path.iterdir()) == []


def test_stage_dumper_writes_input_and_chunks(tmp_path: Path):
    dumper = StageDumper(tmp_path, "20260101_000000")
    assert dumper.enabled is True
    dumper.dump_stage1_input(source_name="vault", rel_path="folder/note.md", text="raw body")
    dumper.dump_stage2_chunks(
        source_name="vault",
        rel_path="folder/note.md",
        chunks=[_FakeChunk(chunk_key="k1", text="t1"), _FakeChunk(chunk_key="k2", text="t2")],
    )
    base = tmp_path / "20260101_000000"
    input_path = base / "stage_1_input" / "vault" / "folder" / "note.md.input.txt"
    chunks_path = base / "stage_2_chunks" / "vault" / "folder" / "note.md.chunks.jsonl"
    assert input_path.read_text(encoding="utf-8") == "raw body"
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == [{"chunk_key": "k1", "text": "t1"}, {"chunk_key": "k2", "text": "t2"}]
