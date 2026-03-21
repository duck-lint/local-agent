from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_make_release_zip_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "make_release_zip.py"
    spec = importlib.util.spec_from_file_location("test_make_release_zip_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


make_release_zip_script = _load_make_release_zip_module()


class ToolingScriptsTests(unittest.TestCase):
    def test_resolve_workroot_prefers_argument_then_env_then_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            workspace_root.mkdir(parents=True, exist_ok=True)

            default_path = make_release_zip_script._resolve_workroot(workspace_root, env={})
            self.assertEqual(default_path, workspace_root / "local-agent-workroot")

            env_path = make_release_zip_script._resolve_workroot(
                workspace_root,
                env={make_release_zip_script.WORKROOT_ENV_VAR: str(Path(tmpdir) / "env-workroot")},
            )
            self.assertEqual(env_path, (Path(tmpdir) / "env-workroot").resolve())

            explicit_path = make_release_zip_script._resolve_workroot(
                workspace_root,
                workroot=Path(tmpdir) / "cli-workroot",
                env={make_release_zip_script.WORKROOT_ENV_VAR: str(Path(tmpdir) / "env-workroot")},
            )
            self.assertEqual(explicit_path, (Path(tmpdir) / "cli-workroot").resolve())

    def test_collect_release_files_uses_explicit_workroot_prefix_and_excludes_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            repo_root = workspace_root / "local-agent"
            external_workroot = Path(tmpdir) / "external-workroot"

            (repo_root / "agent").mkdir(parents=True, exist_ok=True)
            (repo_root / "configs").mkdir(parents=True, exist_ok=True)
            (repo_root / "agent" / "__init__.py").write_text("", encoding="utf-8")
            (repo_root / "configs" / "default.yaml").write_text("model: test\n", encoding="utf-8")
            (repo_root / "README.md").write_text("readme\n", encoding="utf-8")
            (repo_root / "SECURITY.md").write_text("security\n", encoding="utf-8")
            (repo_root / "OPERATOR_QUICKREF.md").write_text("quickref\n", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            (repo_root / "repo_marker.py").write_text("", encoding="utf-8")
            (repo_root / ".gitignore").write_text("dist/\n", encoding="utf-8")

            (external_workroot / "allowed" / "sample").mkdir(parents=True, exist_ok=True)
            (external_workroot / "configs").mkdir(parents=True, exist_ok=True)
            (external_workroot / "runs").mkdir(parents=True, exist_ok=True)
            (external_workroot / "allowed" / ".gitkeep").write_text("", encoding="utf-8")
            (external_workroot / "allowed" / "sample" / "note.md").write_text(
                "sample\n",
                encoding="utf-8",
            )
            (external_workroot / "configs" / "local.yaml").write_text("name: test\n", encoding="utf-8")
            (external_workroot / "notes.txt").write_text("top-level\n", encoding="utf-8")
            (external_workroot / "runs" / "run.json").write_text("{}", encoding="utf-8")

            with patch.object(
                make_release_zip_script,
                "_repo_roots",
                return_value=(repo_root, workspace_root),
            ):
                entries = make_release_zip_script._collect_release_files(
                    include_workroot=True,
                    workroot=external_workroot,
                )

            rel_paths = {rel.as_posix() for _, rel in entries}
            self.assertIn("local-agent/README.md", rel_paths)
            self.assertIn("local-agent/agent/__init__.py", rel_paths)
            self.assertIn("local-agent-workroot/notes.txt", rel_paths)
            self.assertIn("local-agent-workroot/allowed/.gitkeep", rel_paths)
            self.assertIn("local-agent-workroot/allowed/sample/note.md", rel_paths)
            self.assertIn("local-agent-workroot/configs/local.yaml", rel_paths)
            self.assertNotIn("local-agent-workroot/runs/run.json", rel_paths)


if __name__ == "__main__":
    unittest.main()
