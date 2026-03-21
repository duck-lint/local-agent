from __future__ import annotations

import unittest

from agent.tools import TOOLS, ToolError, get_read_text_file_policy, resolve_and_validate_path
from tests.support import AppFixture


class ToolSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = AppFixture()
        self.fx.write_corpus_note("note.md", "alpha evidence\n")
        allowed_dupe = self.fx.workroot / "allowed" / "dupe.md"
        allowed_dupe.write_text("allowed copy\n", encoding="utf-8")
        runs_dupe = self.fx.workroot / "runs" / "dupe.md"
        runs_dupe.write_text("runs copy\n", encoding="utf-8")
        hidden = self.fx.workroot / "allowed" / "corpus" / ".env"
        hidden.write_text("token=secret\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.fx.close()

    def test_read_text_file_allows_allowlisted_corpus_path(self) -> None:
        result = TOOLS["read_text_file"].func({"path": "allowed/corpus/note.md", "max_chars": 200})
        self.assertEqual(result["text"], "alpha evidence\n")
        self.assertEqual(result["chars_full"], len("alpha evidence\n"))
        self.assertEqual(result["truncated"], False)

    def test_hidden_path_is_denied(self) -> None:
        with self.assertRaises(ToolError) as ctx:
            resolve_and_validate_path("allowed/corpus/.env", get_read_text_file_policy())
        self.assertEqual(ctx.exception.code, "PATH_DENIED")

    def test_traversal_path_is_denied(self) -> None:
        with self.assertRaises(ToolError) as ctx:
            resolve_and_validate_path("../../secret.md", get_read_text_file_policy())
        self.assertEqual(ctx.exception.code, "PATH_DENIED")

    def test_bare_filename_is_rejected_when_ambiguous(self) -> None:
        with self.assertRaises(ToolError) as ctx:
            resolve_and_validate_path("dupe.md", get_read_text_file_policy())
        self.assertEqual(ctx.exception.code, "AMBIGUOUS_PATH")


if __name__ == "__main__":
    unittest.main()
