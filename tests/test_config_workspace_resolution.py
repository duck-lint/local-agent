from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agent.__main__ import (
    discover_config_path,
    load_config_with_path,
    select_reread_path,
    workspace_root_from_config_path,
)
from agent.tools import TOOLS, configure_tool_security


def _read_text_file(args: dict) -> dict:
    return TOOLS["read_text_file"].func(args)


class ConfigWorkspaceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

        self.repo_root = self.tmp_path / "repo_root"
        self.workroot = self.tmp_path / "workroot"
        self.repo_config_dir = self.repo_root / "configs"
        self.allowed = self.workroot / "allowed"

        self.repo_config_dir.mkdir(parents=True, exist_ok=True)
        (self.allowed / "corpus").mkdir(parents=True, exist_ok=True)
        (self.allowed / "runs").mkdir(parents=True, exist_ok=True)
        (self.allowed / "scratch").mkdir(parents=True, exist_ok=True)

        self.file_path = self.allowed / "corpus" / "a.md"
        self.file_path.write_text("hello from pinned repo config", encoding="utf-8")

        repo_config = """
model: gpt-oss:120b
model_fast: repo-pinned-model
model_big: gpt-oss:120b
security:
  allowed_roots:
    - "../workroot/allowed/corpus/"
    - "../workroot/allowed/runs/"
    - "../workroot/allowed/scratch/"
  allowed_exts:
    - ".md"
    - ".txt"
    - ".json"
  deny_absolute_paths: true
  deny_hidden_paths: true
  allow_any_path: false
  auto_create_allowed_roots: true
  roots_must_be_within_security_root: false
""".strip()
        (self.repo_config_dir / "default.yaml").write_text(repo_config + "\n", encoding="utf-8")

        # Competing config near CWD should be ignored in repo-pinned mode.
        near_config_dir = self.workroot / "configs"
        near_config_dir.mkdir(parents=True, exist_ok=True)
        near_config = "model_fast: should-not-be-selected\n"
        (near_config_dir / "default.yaml").write_text(near_config, encoding="utf-8")

        os.chdir(self.allowed)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def test_repo_config_is_selected_even_with_nearer_cwd_config(self) -> None:
        discovered = discover_config_path(start_dir=Path.cwd(), repo_root=self.repo_root)
        expected_cfg = (self.repo_root / "configs" / "default.yaml").resolve()
        self.assertIsNotNone(discovered)
        self.assertEqual(discovered.resolve(), expected_cfg)

        cfg, cfg_path = load_config_with_path(start_dir=Path.cwd(), repo_root=self.repo_root)
        self.assertEqual(cfg_path.resolve(), expected_cfg)
        self.assertEqual(cfg.get("model_fast"), "repo-pinned-model")

        workspace_root = workspace_root_from_config_path(cfg_path, fallback=Path.cwd())
        self.assertEqual(workspace_root, self.repo_root.resolve())

        configure_tool_security(cfg.get("security", {}), workspace_root=workspace_root)
        result = _read_text_file({"path": "a.md"})
        self.assertEqual(result["text"], "hello from pinned repo config")
        self.assertEqual(result["path"], str(self.file_path.resolve()))

    def test_missing_repo_config_returns_empty(self) -> None:
        empty_repo = self.tmp_path / "empty_repo"
        empty_repo.mkdir(parents=True, exist_ok=True)

        cfg, cfg_path = load_config_with_path(start_dir=Path.cwd(), repo_root=empty_repo)
        self.assertEqual(cfg, {})
        self.assertIsNone(cfg_path)


class RereadPathSelectionTests(unittest.TestCase):
    def test_prefers_original_requested_path(self) -> None:
        original_tool_args = {"path": "allowed/corpus/a.md", "max_chars": 12000}
        evidence_obj = {"path": "C:/tmp/workroot/allowed/corpus/a.md"}

        chosen = select_reread_path(original_tool_args, evidence_obj)
        self.assertEqual(chosen, "allowed/corpus/a.md")

    def test_falls_back_to_evidence_path_when_original_missing(self) -> None:
        chosen = select_reread_path({}, {"path": "C:/tmp/workroot/allowed/corpus/a.md"})
        self.assertEqual(chosen, "C:/tmp/workroot/allowed/corpus/a.md")


if __name__ == "__main__":
    unittest.main()
