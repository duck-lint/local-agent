from __future__ import annotations

from array import array
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.embedders.torch_embedder import TorchEmbedder


class _FakeSentenceTransformer:
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.max_seq_length = 0
        self.half_called = False
        self.float_called = False

    def half(self) -> None:
        self.half_called = True

    def float(self) -> None:
        self.float_called = True

    def encode(self, texts, **kwargs):
        _ = kwargs
        return [[0.6, 0.8] for _ in texts]


class TorchEmbedderTests(unittest.TestCase):
    def test_missing_local_model_path_errors_clearly(self) -> None:
        missing = str(Path(tempfile.gettempdir()) / "missing-local-model-does-not-exist")
        with self.assertRaises(RuntimeError) as ctx:
            TorchEmbedder(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                local_model_path=missing,
                cache_dir="",
                device="cpu",
                dtype="float32",
                batch_size=8,
                max_length=64,
                pooling="mean",
                normalize=True,
                trust_remote_code=False,
                offline_only=True,
            )
        self.assertIn("local_model_path", str(ctx.exception))

    def test_embed_texts_returns_f32_arrays_and_dim(self) -> None:
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_st_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            with patch.dict("sys.modules", {"torch": fake_torch, "sentence_transformers": fake_st_module}):
                with patch.dict(os.environ, {}, clear=False):
                    emb = TorchEmbedder(
                        model_id="sentence-transformers/all-MiniLM-L6-v2",
                        local_model_path=str(model_dir),
                        cache_dir="",
                        device="auto",
                        dtype="float16",
                        batch_size=8,
                        max_length=64,
                        pooling="mean",
                        normalize=True,
                        trust_remote_code=False,
                        offline_only=True,
                    )
                    vecs = emb.embed_texts(["a", "b"])
                    self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
                    self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")

        self.assertEqual(emb.embed_dim, 2)
        self.assertEqual(len(vecs), 2)
        self.assertIsInstance(vecs[0], array)
        self.assertAlmostEqual(float(vecs[0][0]), 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
