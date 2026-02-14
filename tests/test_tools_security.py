from __future__ import annotations

import os
import subprocess
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
                "roots_must_be_within_security_root": True,
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

    def test_explicit_subpath_is_workspace_root_relative_not_cwd(self) -> None:
        self._configure()
        p = self.allowed_root / "corpus" / "a" / "b.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("workspace anchored", encoding="utf-8")
        outside = self.tmp_path / "outside"
        outside.mkdir(parents=True, exist_ok=True)
        old_cwd = os.getcwd()

        os.chdir(outside)
        try:
            result = _read_text_file({"path": "corpus/a/b.md"})
        finally:
            os.chdir(old_cwd)

        self.assertEqual(result["path"], str(p.resolve()))
        self.assertEqual(result["text"], "workspace anchored")

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
        self.assertIn("Resolved path escapes allowed roots", str(cm.exception))

    @unittest.skipUnless(os.name == "posix", "Symlink root poisoning test requires POSIX semantics")
    def test_symlink_root_poisoning_rejected_when_containment_enabled(self) -> None:
        outside = self.tmp_path / "outside_root"
        outside.mkdir(parents=True, exist_ok=True)
        poisoned = self.allowed_root / "poisoned_root"
        os.symlink(outside, poisoned, target_is_directory=True)

        with self.assertRaises(ValueError) as cm:
            configure_tool_security(
                {
                    "allowed_roots": ["poisoned_root/"],
                    "allowed_exts": [".md", ".txt", ".json"],
                    "deny_absolute_paths": True,
                    "deny_hidden_paths": True,
                    "allow_any_path": False,
                    "auto_create_allowed_roots": False,
                    "roots_must_be_within_security_root": True,
                },
                workspace_root=self.allowed_root,
            )
        msg = str(cm.exception)
        self.assertIn("security_root", msg)
        self.assertIn("offending_root_lexical", msg)
        self.assertIn("offending_root_resolved", msg)
        self.assertIn(str(self.allowed_root.resolve()), msg)
        self.assertIn(str(outside.resolve()), msg)

    @unittest.skipUnless(os.name == "nt", "Windows junction poisoning test requires Windows")
    def test_junction_root_poisoning_rejected_when_containment_enabled(self) -> None:
        outside = self.tmp_path / "outside_root"
        outside.mkdir(parents=True, exist_ok=True)
        poisoned = self.allowed_root / "poisoned_junction"
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(poisoned), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self.skipTest(f"Unable to create junction: {proc.stderr.strip() or proc.stdout.strip()}")

        try:
            with self.assertRaises(ValueError) as cm:
                configure_tool_security(
                    {
                        "allowed_roots": ["poisoned_junction/"],
                        "allowed_exts": [".md", ".txt", ".json"],
                        "deny_absolute_paths": True,
                        "deny_hidden_paths": True,
                        "allow_any_path": False,
                        "auto_create_allowed_roots": False,
                        "roots_must_be_within_security_root": True,
                    },
                    workspace_root=self.allowed_root,
                )
            msg = str(cm.exception)
            self.assertIn("security_root", msg)
            self.assertIn("offending_root_lexical", msg)
            self.assertIn("offending_root_resolved", msg)
            self.assertIn(str(self.allowed_root.resolve()), msg)
            self.assertIn(str(outside.resolve()), msg)
        finally:
            subprocess.run(["cmd", "/c", "rmdir", str(poisoned)], capture_output=True, text=True, check=False)

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
                    "roots_must_be_within_security_root": True,
                },
                workspace_root=Path.cwd(),
            )
        self.assertIn("No valid allowed_roots", str(cm.exception))

    def test_outside_root_fails_loudly_when_containment_enabled(self) -> None:
        outside = self.tmp_path / "outside_root"
        outside.mkdir(parents=True, exist_ok=True)
        cfg_path = self.tmp_path / "repo" / "configs" / "default.yaml"

        with self.assertRaises(ValueError) as cm:
            configure_tool_security(
                {
                    "allowed_roots": [str(outside)],
                    "allowed_exts": [".md", ".txt", ".json"],
                    "deny_absolute_paths": True,
                    "deny_hidden_paths": True,
                    "allow_any_path": False,
                    "auto_create_allowed_roots": False,
                    "roots_must_be_within_security_root": True,
                },
                workspace_root=self.allowed_root,
                resolved_config_path=cfg_path,
            )

        msg = str(cm.exception)
        self.assertIn("security_root", msg)
        self.assertIn(str(self.allowed_root.resolve()), msg)
        self.assertIn(str(outside.resolve()), msg)
        self.assertIn(str(cfg_path.resolve()), msg)

    def test_outside_root_allowed_when_containment_disabled(self) -> None:
        outside = self.tmp_path / "outside_root"
        outside.mkdir(parents=True, exist_ok=True)
        target = outside / "outside.md"
        target.write_text("outside allowed", encoding="utf-8")

        configure_tool_security(
            {
                "allowed_roots": [str(outside)],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": False,
            },
            workspace_root=self.allowed_root,
        )

        result = _read_text_file({"path": "outside.md"})
        self.assertEqual(result["path"], str(target.resolve()))
        self.assertEqual(result["text"], "outside allowed")


if __name__ == "__main__":
    unittest.main()
