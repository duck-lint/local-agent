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
        (self.allowed_root / "corpus").mkdir(parents=True, exist_ok=True)
        (self.allowed_root / "runs").mkdir(parents=True, exist_ok=True)
        (self.allowed_root / "scratch").mkdir(parents=True, exist_ok=True)
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
                "auto_create_allowed_roots": False,
                "roots_must_be_within_workspace": True,
            },
            workspace_root=Path.cwd(),
        )

    def test_allowed_read(self) -> None:
        self._configure()
        p = self.allowed_root / "corpus" / "safe.md"
        p.write_text("hello", encoding="utf-8")

        result = _read_text_file({"path": "safe.md"})
        self.assertEqual(result["path"], str(p.resolve()))
        self.assertEqual(result["chars_full"], 5)
        self.assertEqual(result["chars_returned"], 5)
        self.assertEqual(result["truncated"], False)
        self.assertEqual(result["text"], "hello")
        self.assertIn("sha256", result)

    def test_explicit_subpath_allowed(self) -> None:
        self._configure()
        p = self.allowed_root / "corpus" / "nested" / "ok.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("nested ok", encoding="utf-8")

        result = _read_text_file({"path": "corpus/nested/ok.md"})
        self.assertEqual(result["path"], str(p.resolve()))
        self.assertEqual(result["text"], "nested ok")

    def test_ambiguous_filename_denied(self) -> None:
        self._configure()
        p1 = self.allowed_root / "corpus" / "dupe.md"
        p2 = self.allowed_root / "scratch" / "dupe.md"
        p1.write_text("one", encoding="utf-8")
        p2.write_text("two", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "dupe.md"})
        self.assertEqual(cm.exception.code, "AMBIGUOUS_PATH")
        self.assertIn(str(p1.resolve()), str(cm.exception))
        self.assertIn(str(p2.resolve()), str(cm.exception))

    def test_denies_dotenv_no_ext(self) -> None:
        self._configure()
        p = self.allowed_root / "corpus" / ".env"
        p.write_text("SECRET=1\n", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "corpus/.env"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

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
        link = self.allowed_root / "corpus" / "link.md"
        os.symlink(secret, link)

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "link.md"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

    def test_extension_blocked(self) -> None:
        self._configure()
        p = self.allowed_root / "corpus" / "safe.yaml"
        p.write_text("x: 1\n", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "safe.yaml"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

    def test_hidden_path_blocked(self) -> None:
        self._configure()
        p = self.allowed_root / "corpus" / ".secret.md"
        p.write_text("hidden", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": ".secret.md"})
        self.assertEqual(cm.exception.code, "PATH_DENIED")

    def test_workspace_root_file_denied_when_not_allowlisted(self) -> None:
        self._configure()
        p = self.allowed_root / "safe.md"
        p.write_text("root", encoding="utf-8")

        with self.assertRaises(ToolError) as cm:
            _read_text_file({"path": "safe.md"})
        self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")

    def test_misconfigured_roots_fail_closed(self) -> None:
        with self.assertRaises(ValueError) as cm:
            configure_tool_security(
                {
                    "allowed_roots": ["does_not_exist/"],
                    "allowed_exts": [".md", ".txt", ".json"],
                    "deny_absolute_paths": True,
                    "deny_hidden_paths": True,
                    "allow_any_path": False,
                    "auto_create_allowed_roots": False,
                    "roots_must_be_within_workspace": True,
                },
                workspace_root=Path.cwd(),
            )
        self.assertIn("No valid allowed_roots", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
