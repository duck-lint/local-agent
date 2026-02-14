from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.embed_runtime_fingerprint import (
    build_ollama_runtime_fingerprint,
    build_torch_runtime_fingerprint,
    compute_model_files_fingerprint,
)


class EmbedRuntimeFingerprintTests(unittest.TestCase):
    def test_ollama_runtime_fingerprint_is_deterministic(self) -> None:
        fp1 = build_ollama_runtime_fingerprint(base_url="http://127.0.0.1:11434/", model_id="m")
        fp2 = build_ollama_runtime_fingerprint(base_url="http://127.0.0.1:11434", model_id="m")
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 32)

    def test_torch_runtime_fingerprint_is_deterministic(self) -> None:
        fp1 = build_torch_runtime_fingerprint(
            model_id="m",
            local_model_path="C:/x",
            model_files_fingerprint="abc",
            torch_version="1",
            transformers_version="2",
            sentence_transformers_version="3",
            pooling="mean",
            max_length=128,
            dtype="float32",
            normalize=True,
        )
        fp2 = build_torch_runtime_fingerprint(
            model_id="m",
            local_model_path="C:/x",
            model_files_fingerprint="abc",
            torch_version="1",
            transformers_version="2",
            sentence_transformers_version="3",
            pooling="mean",
            max_length=128,
            dtype="float32",
            normalize=True,
        )
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 32)

    def test_model_files_fingerprint_changes_on_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"x":1}', encoding="utf-8")
            fp1 = compute_model_files_fingerprint(root)
            (root / "config.json").write_text('{"x":2}', encoding="utf-8")
            fp2 = compute_model_files_fingerprint(root)
        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
