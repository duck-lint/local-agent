from __future__ import annotations

import unittest

from agent.__main__ import ensure_canonical_scope_footer_tail, has_exact_scope_footer


class ScopeFooterNormalizationTests(unittest.TestCase):
    def test_replaces_full_sha_tail_footer_with_canonical(self) -> None:
        canonical = (
            "Scope: full evidence from read_text_file (10/10), "
            "sha256=ec41a89da94f9246783b018f7f327f4d5751bd9fcc9cefb009f190b7434a079c"
        )
        full_sha_tail = (
            "Scope: full evidence from read_text_file (10/10), "
            "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        text = f"Answer body\n\n{full_sha_tail}"

        out, changed = ensure_canonical_scope_footer_tail(text, canonical)

        self.assertTrue(changed)
        self.assertTrue(has_exact_scope_footer(out, canonical))
        scope_tail_lines = [line for line in out.splitlines() if line.lstrip().startswith("Scope:")]
        self.assertEqual(scope_tail_lines[-1], canonical)
        self.assertEqual(scope_tail_lines.count(canonical), 1)

    def test_already_canonical_footer_unchanged(self) -> None:
        canonical = (
            "Scope: full evidence from read_text_file (10/10), "
            "sha256=ec41a89da94f9246783b018f7f327f4d5751bd9fcc9cefb009f190b7434a079c"
        )
        text = f"Answer body\n{canonical}\n"

        out, changed = ensure_canonical_scope_footer_tail(text, canonical)

        self.assertFalse(changed)
        self.assertEqual(out, text)
        self.assertTrue(has_exact_scope_footer(out, canonical))

    def test_collapses_multiple_trailing_scope_lines(self) -> None:
        canonical = (
            "Scope: full evidence from read_text_file (10/10), "
            "sha256=ec41a89da94f9246783b018f7f327f4d5751bd9fcc9cefb009f190b7434a079c"
        )
        text = (
            "Answer body\n"
            "Scope: partial evidence from read_text_file (8/10), "
            "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            "Scope: full evidence from read_text_file (10/10), "
            "sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
        )

        out, changed = ensure_canonical_scope_footer_tail(text, canonical)

        self.assertTrue(changed)
        self.assertTrue(has_exact_scope_footer(out, canonical))
        trailing_scope_lines = [line for line in out.splitlines() if line.lstrip().startswith("Scope:")]
        self.assertEqual(trailing_scope_lines[-1], canonical)
        self.assertEqual(trailing_scope_lines.count(canonical), 1)

    def test_preserves_middle_scope_line_and_normalizes_tail(self) -> None:
        canonical = (
            "Scope: full evidence from read_text_file (10/10), "
            "sha256=ec41a89da94f9246783b018f7f327f4d5751bd9fcc9cefb009f190b7434a079c"
        )
        middle_scope = "Scope: this is explanatory text in the body"
        text = (
            f"Intro\n{middle_scope}\n"
            "More body text\n"
            "Scope: partial evidence from read_text_file (9/10), "
            "sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n"
        )

        out, changed = ensure_canonical_scope_footer_tail(text, canonical)

        self.assertTrue(changed)
        self.assertIn(middle_scope, out)
        self.assertTrue(has_exact_scope_footer(out, canonical))
        self.assertEqual(out.splitlines().count(middle_scope), 1)


if __name__ == "__main__":
    unittest.main()
