from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
