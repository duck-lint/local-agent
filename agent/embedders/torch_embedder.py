from __future__ import annotations

from array import array
import os
from pathlib import Path
from typing import Optional

from agent.embedder import Embedder


class TorchEmbedder(Embedder):
    def __init__(
        self,
        *,
        model_id: str,
        local_model_path: str,
        cache_dir: str,
        device: str,
        dtype: str,
        batch_size: int,
        max_length: int,
        pooling: str,
        normalize: bool,
        trust_remote_code: bool,
        offline_only: bool,
    ) -> None:
        self.model_id = str(model_id).strip()
        if not self.model_id:
            raise ValueError("phase3.embed.model_id must be non-empty")

        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self.pooling = str(pooling).strip().lower() or "mean"
        self.normalize = bool(normalize)
        self.trust_remote_code = bool(trust_remote_code)
        self.offline_only = bool(offline_only)
        self._embed_dim = 0
        self._runtime_fp = ""

        if self.pooling != "mean":
            raise ValueError(f"phase3.embed.torch.pooling must be 'mean', got={self.pooling}")
        if not self.normalize:
            raise ValueError("phase3.embed.torch.normalize must be true for phase3 invariants")

        model_source = self._resolve_model_source(local_model_path)
        cache_dir_value = str(cache_dir).strip()
        cache_folder = str(Path(cache_dir_value).expanduser().resolve()) if cache_dir_value else None

        if self.offline_only:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - import errors are environment specific
            raise RuntimeError("Torch provider requires optional dependency 'torch'") from exc
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - import errors are environment specific
            raise RuntimeError("Torch provider requires optional dependency 'sentence-transformers'") from exc

        resolved_device = self._resolve_device(torch, device)
        target_dtype = self._resolve_dtype(resolved_device, dtype)

        local_files_only = True if self.offline_only else False
        model: Optional[object] = None
        try:
            model = SentenceTransformer(
                model_name_or_path=model_source,
                device=resolved_device,
                cache_folder=cache_folder,
                trust_remote_code=self.trust_remote_code,
                local_files_only=local_files_only,
            )
        except TypeError:
            model = SentenceTransformer(
                model_name_or_path=model_source,
                device=resolved_device,
                cache_folder=cache_folder,
                trust_remote_code=self.trust_remote_code,
                model_kwargs={"local_files_only": local_files_only},
                tokenizer_kwargs={"local_files_only": local_files_only},
            )
        except Exception as exc:
            if self.offline_only:
                raise RuntimeError(
                    "Torch model not found locally. Provide phase3.embed.torch.local_model_path "
                    "or pre-download into cache_dir."
                ) from exc
            raise

        if model is None:
            raise RuntimeError("Failed to initialize sentence-transformers model")

        model.max_seq_length = self.max_length  # type: ignore[attr-defined]
        if target_dtype == "float16":
            model.half()  # type: ignore[attr-defined]
        else:
            model.float()  # type: ignore[attr-defined]
        self._model = model
        self._device = resolved_device
        self._dtype = target_dtype
        self._runtime_fp = (
            "provider=torch"
            f";model_id={self.model_id}"
            f";source={model_source}"
            f";device={self._device}"
            f";dtype={self._dtype}"
            f";max_length={self.max_length}"
            f";pooling={self.pooling}"
            f";normalize={int(self.normalize)}"
            f";offline_only={int(self.offline_only)}"
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def runtime_fingerprint(self) -> str:
        return self._runtime_fp

    def embed_texts(self, texts: list[str]) -> list[array]:
        if not texts:
            return []
        vectors = self._model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        out: list[array] = []
        for row in vectors:
            arr = array("f", [float(value) for value in row])
            out.append(arr)
        if out and self._embed_dim <= 0:
            self._embed_dim = len(out[0])
        return out

    def _resolve_model_source(self, local_model_path: str) -> str:
        path_text = str(local_model_path).strip()
        if not path_text:
            return self.model_id
        p = Path(path_text).expanduser()
        if not p.exists():
            raise RuntimeError(
                f"Torch local model path does not exist: {p}. "
                "Provide phase3.embed.torch.local_model_path or pre-download into cache_dir."
            )
        return str(p.resolve())

    def _resolve_device(self, torch_module: object, configured: str) -> str:
        want = str(configured).strip().lower() or "auto"
        cuda_ok = bool(getattr(torch_module.cuda, "is_available")())  # type: ignore[attr-defined]
        if want == "auto":
            return "cuda" if cuda_ok else "cpu"
        if want == "cuda" and not cuda_ok:
            raise RuntimeError("phase3.embed.torch.device='cuda' but CUDA is not available")
        if want not in {"cpu", "cuda"}:
            raise ValueError(f"phase3.embed.torch.device must be one of auto|cpu|cuda, got={configured}")
        return want

    def _resolve_dtype(self, device: str, configured: str) -> str:
        want = str(configured).strip().lower() or "float32"
        if want not in {"float16", "float32"}:
            raise ValueError(f"phase3.embed.torch.dtype must be float16|float32, got={configured}")
        if device == "cpu":
            return "float32"
        return want
