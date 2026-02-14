from __future__ import annotations

from array import array
from typing import Any

import requests

from agent.embedder import Embedder


class OllamaEmbedder(Embedder):
    def __init__(self, *, base_url: str, model_id: str, timeout_s: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = int(timeout_s)
        self._embed_dim = 0

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def runtime_fingerprint(self) -> str:
        return f"provider=ollama;base_url={self.base_url};model_id={self.model_id}"

    def embed_texts(self, texts: list[str]) -> list[array]:
        if not texts:
            return []

        bulk = self._try_embed_api_embed_bulk(texts)
        if bulk is not None:
            out = [array("f", vec) for vec in bulk]
            if out and self._embed_dim <= 0:
                self._embed_dim = len(out[0])
            return out

        out: list[array] = []
        for text in texts:
            out.append(array("f", self._embed_single(text)))
        if out and self._embed_dim <= 0:
            self._embed_dim = len(out[0])
        return out

    def _embed_single(self, text: str) -> list[float]:
        last_error: Exception | None = None

        payloads = [
            ("/api/embeddings", {"model": self.model_id, "prompt": text}),
            ("/api/embed", {"model": self.model_id, "input": text}),
        ]
        for endpoint, payload in payloads:
            try:
                data = self._post_json(endpoint, payload)
                vec = _extract_single_embedding(data)
                return vec
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise RuntimeError(f"Ollama embedding request failed: {last_error}")

    def _try_embed_api_embed_bulk(self, texts: list[str]) -> list[list[float]] | None:
        payload = {"model": self.model_id, "input": texts}
        try:
            data = self._post_json("/api/embed", payload)
            vectors = _extract_many_embeddings(data, expected=len(texts))
            return vectors
        except Exception:
            return None

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"{endpoint} request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"{endpoint} returned non-JSON response") from exc


def _coerce_vector(raw: Any) -> list[float]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Embedding vector is missing or empty")
    out: list[float] = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError("Embedding vector contains bool")
        if not isinstance(value, (int, float)):
            raise ValueError("Embedding vector contains non-numeric value")
        out.append(float(value))
    return out


def _extract_single_embedding(payload: Any) -> list[float]:
    if not isinstance(payload, dict):
        raise ValueError("Embedding payload is not a JSON object")

    if "embedding" in payload:
        return _coerce_vector(payload.get("embedding"))

    if "embeddings" in payload:
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return _coerce_vector(embeddings[0])
    raise ValueError("Embedding payload missing 'embedding'/'embeddings'")


def _extract_many_embeddings(payload: Any, *, expected: int) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise ValueError("Embedding payload is not a JSON object")

    raw_embeddings = payload.get("embeddings")
    if not isinstance(raw_embeddings, list):
        raise ValueError("Embedding payload missing 'embeddings' list")

    vectors = [_coerce_vector(item) for item in raw_embeddings]
    if len(vectors) != expected:
        raise ValueError(f"Embedding count mismatch: expected={expected} got={len(vectors)}")

    if not vectors:
        return vectors
    expected_dim = len(vectors[0])
    if expected_dim <= 0:
        raise ValueError("Embedding dimension must be positive")
    if any(len(v) != expected_dim for v in vectors):
        raise ValueError("Embedding vectors have inconsistent dimensions")
    return vectors
