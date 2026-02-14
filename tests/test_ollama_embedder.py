from __future__ import annotations

from array import array
import unittest
from unittest.mock import patch

import requests

from agent.embedders.ollama import OllamaEmbedder


class _FakeResponse:
    def __init__(self, status_code: int, payload=None) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class OllamaEmbedderTests(unittest.TestCase):
    def test_embed_uses_primary_embed_endpoint_when_available(self) -> None:
        calls: list[str] = []

        def _post(url, json, timeout):  # type: ignore[no-untyped-def]
            _ = timeout
            calls.append(url)
            self.assertEqual(json["model"], "m")
            self.assertIn("input", json)
            return _FakeResponse(200, {"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

        emb = OllamaEmbedder(base_url="http://127.0.0.1:11434", model_id="m", timeout_s=2)
        with patch("agent.embedders.ollama.requests.post", side_effect=_post):
            out = emb.embed_texts(["a", "b"])

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/api/embed"))
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], array)
        self.assertEqual(emb.embed_dim, 2)

    def test_embed_falls_back_to_embeddings_on_404(self) -> None:
        calls: list[str] = []

        def _post(url, json, timeout):  # type: ignore[no-untyped-def]
            _ = timeout
            calls.append(url)
            if url.endswith("/api/embed"):
                return _FakeResponse(404, {"error": "not found"})
            if url.endswith("/api/embeddings"):
                prompt = str(json.get("prompt", ""))
                if prompt == "a":
                    return _FakeResponse(200, {"embedding": [1.0, 0.0]})
                return _FakeResponse(200, {"embedding": [0.0, 1.0]})
            raise AssertionError(url)

        emb = OllamaEmbedder(base_url="http://127.0.0.1:11434", model_id="m", timeout_s=2)
        with patch("agent.embedders.ollama.requests.post", side_effect=_post):
            out = emb.embed_texts(["a", "b"])

        self.assertEqual(len(out), 2)
        self.assertGreaterEqual(sum(1 for c in calls if c.endswith("/api/embeddings")), 2)

    def test_embed_reports_both_endpoints_when_both_fail(self) -> None:
        def _post(url, json, timeout):  # type: ignore[no-untyped-def]
            _ = json, timeout
            if url.endswith("/api/embed"):
                return _FakeResponse(404, {"error": "not found"})
            return _FakeResponse(404, {"error": "not found"})

        emb = OllamaEmbedder(base_url="http://127.0.0.1:11434", model_id="m", timeout_s=2)
        with patch("agent.embedders.ollama.requests.post", side_effect=_post):
            with self.assertRaises(RuntimeError) as ctx:
                emb.embed_texts(["a"])
        text = str(ctx.exception)
        self.assertIn("/api/embed", text)
        self.assertIn("/api/embeddings", text)


if __name__ == "__main__":
    unittest.main()
