from __future__ import annotations

import json
from pathlib import Path

from agent.manifests import (
    MANIFEST_INDEX_FILENAME,
    append_manifest_index,
    git_info,
    stable_settings_hash,
    system_info,
    utc_iso,
    utc_now,
    write_run_manifest,
)


def test_stable_settings_hash_is_deterministic_and_order_independent():
    a = {"x": 1, "y": [1, 2, 3], "z": "abc"}
    b = {"z": "abc", "y": [1, 2, 3], "x": 1}
    short_a, full_a = stable_settings_hash(a)
    short_b, full_b = stable_settings_hash(b)
    assert short_a == short_b
    assert full_a == full_b
    assert len(short_a) == 16
    assert len(full_a) == 64


def test_stable_settings_hash_differs_on_value_change():
    short_a, _ = stable_settings_hash({"k": 1})
    short_b, _ = stable_settings_hash({"k": 2})
    assert short_a != short_b


def test_utc_iso_uses_z_suffix():
    now = utc_now()
    iso = utc_iso(now)
    assert iso.endswith("Z")
    assert "+00:00" not in iso


def test_system_info_keys():
    info = system_info()
    assert set(info.keys()) == {"hostname", "platform", "python_version"}
    assert all(isinstance(v, str) and v for v in info.values())


def test_git_info_safe_outside_repo(tmp_path: Path):
    commit, dirty = git_info(tmp_path)
    # Outside any git repo OR git missing → both None.
    assert commit is None or isinstance(commit, str)
    assert dirty is None or isinstance(dirty, bool)


def test_write_run_manifest_filename_and_content(tmp_path: Path):
    finished = utc_now()
    payload = {"run_id": "test", "kind": "index", "outcomes": {"docs": 3}}
    manifest_path = write_run_manifest(
        manifest_dir=tmp_path / "manifests",
        kind="index",
        settings_hash_short="abcdef0123456789",
        finished_at=finished,
        payload=payload,
    )
    assert manifest_path.exists()
    assert manifest_path.name.startswith("run_manifest_index_")
    assert manifest_path.name.endswith("_abcdef0123456789.json")
    # No leftover .tmp.
    leftovers = list((tmp_path / "manifests").glob("*.json.tmp"))
    assert leftovers == []
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == payload


def test_append_manifest_index_is_jsonl(tmp_path: Path):
    md = tmp_path / "manifests"
    line_a = {"manifest_filename": "a.json", "run_id": "A"}
    line_b = {"manifest_filename": "b.json", "run_id": "B"}
    append_manifest_index(md, line_a)
    append_manifest_index(md, line_b)
    index_path = md / MANIFEST_INDEX_FILENAME
    text = index_path.read_text(encoding="utf-8")
    rows = [json.loads(row) for row in text.splitlines() if row.strip()]
    assert rows == [line_a, line_b]
