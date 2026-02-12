from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agent.tools import TOOLS, ToolError, configure_tool_security


def _read_text_file(args: dict) -> dict:
    return TOOLS["read_text_file"].func(args)


class ReadTextFileSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.allowed_root = self.tmp_path / "allowed_root"
        self.allowed_root.mkdir(parents=True, exist_ok=True)
        os.chdir(self.allowed_root)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _configure(self) -> None:
        configure_tool_security(
            {
                "allowed_roots": ["corpus/", "runs/", "scratch/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
            },
            workspace_root=Path.cwd(),
        )

    def test_allowed_read(self) -> None:
        self._configure()
        p = self.allowed_root / "safe.md"
        p.write_text("hello", encoding="utf-8")

        result = _read_text_file({"path": "safe.md"})
        self.assertEqual(result["path"], str(p.resolve()))
        self.assertEqual(result["chars_full"], 5)
        self.assertEqual(result["chars_returned"], 5)
        self.assertEqual(result["truncated"], False)
        self.assertEqual(result["text"], "hello")
        self.assertIn("sha256", result)

    def test_traversal_blocked(self) -> None:
        self._configure()
        secret = self.tmp_path / "secret.md"
        secret.write_text("secret", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "../secret.md"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

    def test_absolute_path_blocked(self) -> None:
        self._configure()
        secret = self.tmp_path / "secret.md"
        secret.write_text("secret", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": str(secret.resolve())})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

    @unittest.skipUnless(os.name == "posix", "Symlink escape test requires POSIX semantics")
    def test_symlink_escape_blocked(self) -> None:
        self._configure()
        secret = self.tmp_path / "secret.md"
        secret.write_text("secret", encoding="utf-8")
        link = self.allowed_root / "link.md"
        os.symlink(secret, link)

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "link.md"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

    def test_extension_blocked(self) -> None:
        self._configure()
        p = self.allowed_root / "safe.yaml"
        p.write_text("x: 1\n", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "safe.yaml"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

    def test_hidden_path_blocked(self) -> None:
        self._configure()
        p = self.allowed_root / ".secret.md"
        p.write_text("hidden", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": ".secret.md"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")


if __name__ == "__main__":
    unittest.main()
