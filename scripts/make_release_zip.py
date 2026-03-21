from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import zipfile

WORKROOT_ENV_VAR = "LOCAL_AGENT_WORKROOT"


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


def _resolve_workroot(
    workspace_root: Path,
    workroot: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Path:
    raw = workroot
    if raw is None:
        scope = os.environ if env is None else env
        raw = scope.get(WORKROOT_ENV_VAR)
    if raw is None:
        return workspace_root / "local-agent-workroot"
    return Path(raw).expanduser().resolve()


def _collect_release_files(
    include_workroot: bool,
    *,
    workroot: Path | None = None,
) -> list[tuple[Path, PurePosixPath]]:
    local_agent_root, workspace_root = _repo_roots()
    selected: dict[str, tuple[Path, PurePosixPath]] = {}

    def add_selected_file(path: Path, rel: PurePosixPath) -> None:
        if not path.exists() or not path.is_file():
            return
        if _is_excluded(rel):
            return
        selected[rel.as_posix()] = (path, rel)

    def add_repo_file(path: Path) -> None:
        try:
            rel = PurePosixPath(path.resolve().relative_to(workspace_root.resolve()).as_posix())
        except ValueError:
            return
        add_selected_file(path, rel)

    def add_workroot_file(path: Path, workroot_root: Path) -> None:
        try:
            rel = PurePosixPath(path.resolve().relative_to(workroot_root.resolve()).as_posix())
        except ValueError:
            return
        add_selected_file(path, PurePosixPath("local-agent-workroot") / rel)

    def add_tree(path: Path, suffix_filter: str | None = None, *, add_file_fn=add_repo_file) -> None:
        for p in _iter_files(path):
            if suffix_filter is not None and p.suffix.lower() != suffix_filter:
                continue
            add_file_fn(p)

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
        add_repo_file(local_agent_root / rel_file)

    if include_workroot:
        workroot_root = _resolve_workroot(workspace_root, workroot)
        for p in _iter_files(workroot_root):
            rel = PurePosixPath(p.resolve().relative_to(workroot_root.resolve()).as_posix())
            if len(rel.parts) != 1:
                continue
            if p.suffix.lower() in {".ps1", ".sh", ".md", ".txt", ".json", ".yaml", ".yml"}:
                add_workroot_file(p, workroot_root)
        add_workroot_file(workroot_root / "allowed" / ".gitkeep", workroot_root)
        add_tree(
            workroot_root / "allowed" / "sample",
            add_file_fn=lambda path: add_workroot_file(path, workroot_root),
        )
        add_tree(
            workroot_root / "configs",
            add_file_fn=lambda path: add_workroot_file(path, workroot_root),
        )

    return [selected[k] for k in sorted(selected)]


def make_release_zip(
    out_path: Path,
    include_workroot: bool,
    dry_run: bool,
    *,
    workroot: Path | None = None,
) -> int:
    entries = _collect_release_files(include_workroot=include_workroot, workroot=workroot)
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
    default_workroot = _resolve_workroot(workspace_root)
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
        "--workroot",
        type=str,
        default=str(default_workroot),
        help=(
            "Workroot to package when --include-workroot is set. "
            f"Defaults to {WORKROOT_ENV_VAR} or ../local-agent-workroot."
        ),
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
        workroot=Path(args.workroot).expanduser().resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
