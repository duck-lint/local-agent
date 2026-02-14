from __future__ import annotations

import unittest

from agent.embedding_fingerprint import (
    build_chunk_embedding_input,
    compute_embed_preprocess_sig,
    compute_embed_sig,
    pack_vector_f32_le,
    unpack_vector_f32_le,
)


class EmbeddingFingerprintTests(unittest.TestCase):
    def test_preprocess_sig_is_deterministic(self) -> None:
        sig1 = compute_embed_preprocess_sig("obsidian_v1")
        sig2 = compute_embed_preprocess_sig("obsidian_v1")
        self.assertEqual(sig1, sig2)
        self.assertEqual(len(sig1), 32)

    def test_embed_sig_is_deterministic(self) -> None:
        sig1 = compute_embed_sig(
            chunk_key="abc123",
            chunk_sha="def456",
            model_id="nomic-embed-text-v1.5",
            dim=768,
            preprocess_sig="feedbeef",
        )
        sig2 = compute_embed_sig(
            chunk_key="abc123",
            chunk_sha="def456",
            model_id="nomic-embed-text-v1.5",
            dim=768,
            preprocess_sig="feedbeef",
        )
        sig3 = compute_embed_sig(
            chunk_key="abc123",
            chunk_sha="changed",
            model_id="nomic-embed-text-v1.5",
            dim=768,
            preprocess_sig="feedbeef",
        )
        self.assertEqual(sig1, sig2)
        self.assertNotEqual(sig1, sig3)

    def test_vector_blob_roundtrip(self) -> None:
        original = [0.25, -1.0, 3.5, 2.0]
        blob = pack_vector_f32_le(original)
        roundtrip = unpack_vector_f32_le(blob)
        self.assertEqual(len(roundtrip), len(original))
        for got, expected in zip(roundtrip, original):
            self.assertAlmostEqual(float(got), float(expected), places=6)

    def test_preprocess_prefix_is_stable(self) -> None:
        text = "line one  \r\nline two\t\r\n"
        rendered = build_chunk_embedding_input(
            preprocess_name="obsidian_v1",
            rel_path="notes/a.md",
            heading_path="H2: Alpha",
            text=text,
        )
        self.assertTrue(rendered.startswith("rel_path=notes/a.md\nheading_path=H2: Alpha\n\n"))
        self.assertIn("line one\nline two", rendered)


if __name__ == "__main__":
    unittest.main()
