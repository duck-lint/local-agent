from __future__ import annotations

from array import array
from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[array]:
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        return 0

    def runtime_fingerprint(self) -> str:
        return ""
