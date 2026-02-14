from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Any

import requests

from agent.embed_runtime_fingerprint import build_ollama_runtime_fingerprint
from agent.embedder import Embedder


@dataclass
class _EndpointError(RuntimeError):
    endpoint: str
    status_code: int | None
    detail: str

    def __str__(self) -> str:
        status = self.status_code if self.status_code is not None else "unknown"
        return f"{self.endpoint} failed (status={status}): {self.detail}"


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
        return build_ollama_runtime_fingerprint(base_url=self.base_url, model_id=self.model_id)

    def embed_texts(self, texts: list[str]) -> list[array]:
        if not texts:
            return []

        first_error: _EndpointError | None = None
        try:
            payload = {"model": self.model_id, "input": texts}
            data = self._post_json("/api/embed", payload)
            vectors = _extract_many_embeddings(data, expected=len(texts))
            return self._to_arrays(vectors)
        except _EndpointError as exc:
            first_error = exc
            if exc.status_code != 404:
                raise RuntimeError(f"Ollama embedding request failed via /api/embed: {exc}") from exc

        try:
            vectors = [self._embed_single_via_embeddings_endpoint(text) for text in texts]
            return self._to_arrays(vectors)
        except _EndpointError as exc:
            detail = (
                "Ollama embedding failed after trying /api/embed then /api/embeddings: "
                f"{first_error}; {exc}"
            )
            raise RuntimeError(detail) from exc

    def _to_arrays(self, vectors: list[list[float]]) -> list[array]:
        out = [array("f", vec) for vec in vectors]
        if out and self._embed_dim <= 0:
            self._embed_dim = len(out[0])
        return out

    def _embed_single_via_embeddings_endpoint(self, text: str) -> list[float]:
        payload = {"model": self.model_id, "prompt": text}
        data = self._post_json("/api/embeddings", payload)
        return _extract_single_embedding(data)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise _EndpointError(endpoint=endpoint, status_code=None, detail=str(exc)) from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise _EndpointError(endpoint=endpoint, status_code=response.status_code, detail=str(exc)) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise _EndpointError(endpoint=endpoint, status_code=response.status_code, detail="non-JSON response") from exc


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
    if raw_embeddings is None and "embedding" in payload:
        raw_embeddings = [payload.get("embedding")]
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
