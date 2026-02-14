from __future__ import annotations

import unittest

from agent.chunking import (
    chunk_markdown,
    chunk_markdown_obsidian_v1,
)


class Phase2ChunkingTests(unittest.TestCase):
    def test_h2_split_prefers_h2_sections(self) -> None:
        body = (
            "# Top\n"
            "intro paragraph before sections.\n\n"
            "## Alpha\n"
            "alpha paragraph one.\n\n"
            "### Alpha Child\n"
            "alpha child paragraph.\n\n"
            "## Beta\n"
            "beta paragraph one.\n"
        )
        chunks = chunk_markdown_obsidian_v1(body_text=body, max_chars=400, overlap=40)
        self.assertGreaterEqual(len(chunks), 3)

        first_alpha = None
        first_beta = None
        for i, chunk in enumerate(chunks):
            self.assertNotIn("## Alpha", chunk.text)
            self.assertNotIn("## Beta", chunk.text)
            if chunk.heading_path and "H2: Alpha" in chunk.heading_path and first_alpha is None:
                first_alpha = i
            if chunk.heading_path and "H2: Beta" in chunk.heading_path and first_beta is None:
                first_beta = i

        self.assertIsNotNone(first_alpha)
        self.assertIsNotNone(first_beta)
        self.assertLess(first_alpha, first_beta)

    def test_smallest_heading_when_no_h2(self) -> None:
        body = (
            "### Gamma\n"
            "gamma paragraph.\n\n"
            "### Delta\n"
            "delta paragraph.\n"
        )
        chunks = chunk_markdown_obsidian_v1(body_text=body, max_chars=200, overlap=30)
        self.assertGreaterEqual(len(chunks), 2)
        paths = [chunk.heading_path for chunk in chunks if chunk.heading_path]
        self.assertTrue(any(path == "H3: Gamma" for path in paths))
        self.assertTrue(any(path == "H3: Delta" for path in paths))

    def test_large_paragraph_split_is_deterministic(self) -> None:
        sentence = "This is a deterministic sentence for chunk splitting."
        large_para = " ".join([sentence for _ in range(30)])
        body = f"## Long\n{large_para}\n"

        run1 = chunk_markdown_obsidian_v1(body_text=body, max_chars=180, overlap=25)
        run2 = chunk_markdown_obsidian_v1(body_text=body, max_chars=180, overlap=25)
        self.assertGreaterEqual(len(run1), 2)
        self.assertEqual(
            [
                (
                    c.chunk_index,
                    c.start_char,
                    c.end_char,
                    c.section_ordinal,
                    c.chunk_ordinal,
                    c.heading_path,
                    c.text,
                )
                for c in run1
            ],
            [
                (
                    c.chunk_index,
                    c.start_char,
                    c.end_char,
                    c.section_ordinal,
                    c.chunk_ordinal,
                    c.heading_path,
                    c.text,
                )
                for c in run2
            ],
        )
        self.assertTrue(all(len(c.text) <= 180 for c in run1))

    def test_body_only_invariant_for_legacy_alias(self) -> None:
        text = (
            "---\n"
            "uuid: test-uuid\n"
            "---\n"
            "\n"
            "Body content remains.\n"
        )
        chunks = chunk_markdown(text=text, max_chars=200, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Body content remains.", chunks[0].text)
        self.assertNotIn("uuid:", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
