from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


MANIFEST_INDEX_FILENAME = "run_manifests_index.jsonl"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def stable_settings_hash(payload: dict[str, Any]) -> tuple[str, str]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()
    return digest[:16], digest


def git_info(start_dir: Path) -> tuple[Optional[str], Optional[bool]]:
    start = Path(start_dir).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if commit.returncode != 0:
            return None, None
        commit_sha = commit.stdout.strip() or None
        dirty = subprocess.run(
            ["git", "-C", str(start), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if dirty.returncode != 0:
            return commit_sha, None
        return commit_sha, bool(dirty.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, None


def system_info() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
    }


def write_run_manifest(
    *,
    manifest_dir: Path,
    kind: str,
    settings_hash_short: str,
    finished_at: datetime,
    payload: dict[str, Any],
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = finished_at.strftime("%Y%m%d_%H%M%S")
    filename = f"run_manifest_{kind}_{ts}_{settings_hash_short}.json"
    target = manifest_dir / filename
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def append_manifest_index(manifest_dir: Path, index_line: dict[str, Any]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    index_path = manifest_dir / MANIFEST_INDEX_FILENAME
    with index_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(index_line, ensure_ascii=False, separators=(",", ":")) + "\n")
    return index_path
