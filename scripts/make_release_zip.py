from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import zipfile


def _repo_roots() -> tuple[Path, Path]:
    local_agent_root = Path(__file__).resolve().parents[1]
    workspace_root = local_agent_root.parent
    return local_agent_root, workspace_root


def _is_excluded(rel: PurePosixPath) -> bool:
    parts = set(rel.parts)
    if ".git" in parts:
        return True
    if ".venv" in parts:
        return True
    if "__pycache__" in parts:
        return True
    if ".pytest_cache" in parts:
        return True
    if any(p.endswith(".egg-info") for p in rel.parts):
        return True

    path = rel.as_posix()
    if path.startswith("dist/") or path.startswith("build/"):
        return True
    if path.startswith("local-agent-workroot/runs/"):
        return True
    if path.endswith(".pyc"):
        return True
    return False


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            out.append(p)
    return out


def _collect_release_files(include_workroot: bool) -> list[tuple[Path, PurePosixPath]]:
    local_agent_root, workspace_root = _repo_roots()
    selected: dict[str, tuple[Path, PurePosixPath]] = {}

    def add_file(path: Path) -> None:
        try:
            rel = PurePosixPath(path.resolve().relative_to(workspace_root.resolve()).as_posix())
        except ValueError:
            return
        if _is_excluded(rel):
            return
        selected[rel.as_posix()] = (path, rel)

    def add_tree(path: Path, suffix_filter: str | None = None) -> None:
        for p in _iter_files(path):
            if suffix_filter is not None and p.suffix.lower() != suffix_filter:
                continue
            add_file(p)

    add_tree(local_agent_root / "agent", suffix_filter=".py")
    add_tree(local_agent_root / "configs")
    for rel_file in [
        "README.md",
        "SECURITY.md",
        "OPERATOR_QUICKREF.md",
        "pyproject.toml",
        "repo_marker.py",
        ".gitignore",
    ]:
        add_file(local_agent_root / rel_file)

    if include_workroot:
        workroot = workspace_root / "local-agent-workroot"
        for p in _iter_files(workroot):
            rel = PurePosixPath(p.resolve().relative_to(workspace_root.resolve()).as_posix())
            if len(rel.parts) != 2:
                continue
            if p.suffix.lower() in {".ps1", ".sh", ".md", ".txt", ".json", ".yaml", ".yml"}:
                add_file(p)
        add_file(workroot / "allowed" / ".gitkeep")
        add_tree(workroot / "allowed" / "sample")
        add_tree(workroot / "configs")

    return [selected[k] for k in sorted(selected)]


def make_release_zip(out_path: Path, include_workroot: bool, dry_run: bool) -> int:
    entries = _collect_release_files(include_workroot=include_workroot)
    if dry_run:
        for _, rel in entries:
            print(rel.as_posix())
        print(f"\n[dry-run] files={len(entries)}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_path, rel in entries:
            zf.write(abs_path, arcname=rel.as_posix())
    print(f"[ok] wrote {out_path} (files={len(entries)})")
    return 0


def main() -> int:
    _, workspace_root = _repo_roots()
    parser = argparse.ArgumentParser(
        description="Create a clean release zip with curated local-agent payload."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(workspace_root / "dist" / "local-agent-release.zip"),
        help="Output zip path.",
    )
    parser.add_argument(
        "--include-workroot",
        action="store_true",
        help="Include curated local-agent-workroot payload (excluding runs/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print included paths without writing a zip.",
    )
    args = parser.parse_args()
    return make_release_zip(
        out_path=Path(args.out).expanduser().resolve(),
        include_workroot=bool(args.include_workroot),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
