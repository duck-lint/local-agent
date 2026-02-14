from __future__ import annotations

import unittest

from agent.embedding_fingerprint import (
    compute_chunk_preprocess_sig,
    compute_embed_sig,
    compute_query_preprocess_sig,
    pack_vector_f32_le,
    preprocess_chunk_text,
    preprocess_query_text,
    unpack_vector_f32_le,
)


class EmbeddingFingerprintTests(unittest.TestCase):
    def test_chunk_preprocess_sig_deterministic(self) -> None:
        sig1 = compute_chunk_preprocess_sig("obsidian_v1")
        sig2 = compute_chunk_preprocess_sig("obsidian_v1")
        self.assertEqual(sig1, sig2)
        self.assertEqual(len(sig1), 32)

    def test_query_preprocess_sig_deterministic(self) -> None:
        sig1 = compute_query_preprocess_sig("obsidian_v1")
        sig2 = compute_query_preprocess_sig("obsidian_v1")
        self.assertEqual(sig1, sig2)
        self.assertEqual(len(sig1), 32)

    def test_embed_sig_uses_chunk_preprocess_sig(self) -> None:
        sig1 = compute_embed_sig(
            chunk_key="abc123",
            chunk_sha="def456",
            model_id="nomic-embed-text-v1.5",
            dim=768,
            chunk_preprocess_sig="feedbeef",
        )
        sig2 = compute_embed_sig(
            chunk_key="abc123",
            chunk_sha="def456",
            model_id="nomic-embed-text-v1.5",
            dim=768,
            chunk_preprocess_sig="feedbeef",
        )
        sig3 = compute_embed_sig(
            chunk_key="abc123",
            chunk_sha="def456",
            model_id="nomic-embed-text-v1.5",
            dim=768,
            chunk_preprocess_sig="beadfeed",
        )
        self.assertEqual(sig1, sig2)
        self.assertNotEqual(sig1, sig3)

    def test_query_preprocess_has_no_fake_prefix(self) -> None:
        rendered = preprocess_query_text(query="hello\r\nworld  ", preprocess_name="obsidian_v1")
        self.assertEqual(rendered, "hello\nworld")
        self.assertNotIn("rel_path=", rendered)
        self.assertNotIn("__query__", rendered)

    def test_chunk_preprocess_prefix_is_stable(self) -> None:
        text = "line one  \r\nline two\t\r\n"
        rendered = preprocess_chunk_text(
            preprocess_name="obsidian_v1",
            rel_path="notes/a.md",
            heading_path="H2: Alpha",
            chunk_text=text,
        )
        self.assertTrue(rendered.startswith("rel_path=notes/a.md\nheading_path=H2: Alpha\n\n"))
        self.assertIn("line one\nline two", rendered)

    def test_vector_blob_roundtrip(self) -> None:
        original = [0.25, -1.0, 3.5, 2.0]
        blob = pack_vector_f32_le(original)
        roundtrip = unpack_vector_f32_le(blob)
        self.assertEqual(len(roundtrip), len(original))
        for got, expected in zip(roundtrip, original):
            self.assertAlmostEqual(float(got), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
