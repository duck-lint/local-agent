from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agent.__main__ import load_config_with_path, resolve_runtime_roots, root_log_fields
from agent.runtime_config import resolve_ollama_base_url
from agent.tools import TOOLS, configure_tool_security


def _read_text_file(args: dict) -> dict:
    return TOOLS["read_text_file"].func(args)


class RootResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

        self.repo_root = self.tmp_path / "repo"
        self.config_dir = self.repo_root / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / "default.yaml"
        self.config_path.write_text("model: test-model\n", encoding="utf-8")

        self.workroot = self.tmp_path / "workroot"
        (self.workroot / "allowed" / "corpus").mkdir(parents=True, exist_ok=True)
        (self.workroot / "allowed" / "runs").mkdir(parents=True, exist_ok=True)
        (self.workroot / "allowed" / "scratch").mkdir(parents=True, exist_ok=True)
        (self.workroot / "runs").mkdir(parents=True, exist_ok=True)
        self.target_file = self.workroot / "allowed" / "corpus" / "a.md"
        self.target_file.write_text("hello from workroot", encoding="utf-8")

        self.elsewhere = self.tmp_path / "elsewhere"
        self.elsewhere.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def test_resolve_roots_with_sibling_workroot_and_root_log_fields(self) -> None:
        roots = resolve_runtime_roots(
            resolved_config_path=self.config_path,
            cfg={},
            cli_workroot="../workroot",
            cwd=self.elsewhere,
            package_root=self.repo_root,
        )

        self.assertEqual(roots["config_root"], self.repo_root.resolve())
        self.assertEqual(roots["package_root"], self.repo_root.resolve())
        self.assertEqual(roots["workroot"], self.workroot.resolve())
        self.assertEqual(roots["security_root"], self.workroot.resolve())

        fields = root_log_fields(roots)
        self.assertIn("config_root", fields)
        self.assertIn("package_root", fields)
        self.assertIn("workroot", fields)
        self.assertIn("security_root", fields)
        self.assertNotIn("workspace_root", fields)

    def test_security_anchor_uses_workroot_even_when_cwd_differs(self) -> None:
        roots = resolve_runtime_roots(
            resolved_config_path=self.config_path,
            cfg={},
            cli_workroot=str(self.workroot),
            cwd=self.elsewhere,
            package_root=self.repo_root,
        )

        os.chdir(self.elsewhere)
        configure_tool_security(
            {
                "allowed_roots": ["allowed/corpus/", "allowed/runs/", "allowed/scratch/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": True,
            },
            workspace_root=roots["security_root"],
        )

        result = _read_text_file({"path": "allowed/corpus/a.md"})
        self.assertEqual(result["path"], str(self.target_file.resolve()))
        self.assertEqual(result["text"], "hello from workroot")

    def test_security_root_falls_back_to_config_root_without_workroot(self) -> None:
        roots = resolve_runtime_roots(
            resolved_config_path=self.config_path,
            cfg={},
            cli_workroot=None,
            cwd=self.elsewhere,
            package_root=self.repo_root,
        )
        self.assertEqual(roots["security_root"], self.repo_root.resolve())

    def test_security_root_falls_back_to_cwd_when_no_config_or_workroot(self) -> None:
        roots = resolve_runtime_roots(
            resolved_config_path=None,
            cfg={},
            cli_workroot=None,
            cwd=self.elsewhere,
            package_root=self.repo_root,
        )
        self.assertEqual(roots["security_root"], self.elsewhere.resolve())

    def test_workroot_precedence_cli_over_env_and_config(self) -> None:
        roots = resolve_runtime_roots(
            resolved_config_path=self.config_path,
            cfg={"workroot": "cfg-root"},
            cli_workroot="cli-root",
            cwd=self.elsewhere,
            env_workroot="env-root",
            package_root=self.repo_root,
        )
        self.assertEqual(roots["workroot"], (self.repo_root / "cli-root").resolve())

    def test_shipped_config_allowed_roots_follow_overridden_workroot(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        cfg, cfg_path = load_config_with_path(start_dir=repo_root, repo_root=repo_root)
        self.assertIsNotNone(cfg_path)

        portable_workroot = self.tmp_path / "portable-external-root"
        (portable_workroot / "allowed" / "corpus").mkdir(parents=True, exist_ok=True)
        (portable_workroot / "allowed" / "scratch").mkdir(parents=True, exist_ok=True)
        (portable_workroot / "runs").mkdir(parents=True, exist_ok=True)
        portable_file = portable_workroot / "allowed" / "corpus" / "portable.md"
        portable_file.write_text("portable config root", encoding="utf-8")

        roots = resolve_runtime_roots(
            resolved_config_path=cfg_path,
            cfg=cfg,
            cli_workroot=str(portable_workroot),
            cwd=self.elsewhere,
            package_root=repo_root,
        )

        os.chdir(self.elsewhere)
        configure_tool_security(
            cfg.get("security", {}),
            workspace_root=roots["security_root"],
            resolved_config_path=cfg_path,
        )

        result = _read_text_file({"path": "allowed/corpus/portable.md"})
        self.assertEqual(result["path"], str(portable_file.resolve()))
        self.assertEqual(result["text"], "portable config root")

    def test_ollama_base_url_precedence_cli_over_env_and_config(self) -> None:
        resolved = resolve_ollama_base_url(
            {"ollama_base_url": "http://config-host:11434"},
            cli_base_url="https://cli-host:22434/",
            env_base_url="http://env-host:11434",
        )
        self.assertEqual(resolved, "https://cli-host:22434")

    def test_ollama_base_url_precedence_env_over_config(self) -> None:
        resolved = resolve_ollama_base_url(
            {"ollama_base_url": "http://config-host:11434"},
            env_base_url="http://env-host:11434/",
        )
        self.assertEqual(resolved, "http://env-host:11434")

    def test_ollama_base_url_rejects_paths_and_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not include credentials"):
            resolve_ollama_base_url({}, cli_base_url="http://user:pass@example.test:11434")
        with self.assertRaisesRegex(ValueError, "must not include a path"):
            resolve_ollama_base_url({}, cli_base_url="http://example.test:11434/api")


if __name__ == "__main__":
    unittest.main()
