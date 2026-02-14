from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_targets(root: Path) -> tuple[list[Path], list[Path]]:
    dirs: list[Path] = []
    files: list[Path] = []
    skip_roots = {".git", ".venv"}
    for p in root.rglob("*"):
        if any(part in skip_roots for part in p.parts):
            continue
        if p.is_dir() and p.name in {"__pycache__", ".pytest_cache"}:
            dirs.append(p)
        elif p.is_file() and p.suffix.lower() == ".pyc":
            files.append(p)
    dirs = sorted(set(dirs), key=lambda x: str(x))
    files = sorted(set(files), key=lambda x: str(x))
    return dirs, files


def clean_artifacts(root: Path, dry_run: bool) -> int:
    dirs, files = _find_targets(root)
    if dry_run:
        for p in dirs:
            print(f"DIR  {p}")
        for p in files:
            print(f"FILE {p}")
        print(f"\n[dry-run] dirs={len(dirs)} files={len(files)}")
        return 0

    removed_dirs = 0
    removed_files = 0
    for p in dirs:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            removed_dirs += 1
    for p in files:
        if p.exists():
            try:
                p.unlink()
                removed_files += 1
            except OSError:
                pass
    print(f"[ok] removed dirs={removed_dirs} files={removed_files}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove local Python build/test artifacts from this repo."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print targets without deleting.",
    )
    args = parser.parse_args()
    return clean_artifacts(root=_repo_root(), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
