from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional


_KEY_FILENAMES = {
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "modules.json",
    "sentence_bert_config.json",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:32]


def _sha256_bytes(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()[:32]


def _hash_file(path: Path) -> str:
    size = path.stat().st_size
    if size <= 32 * 1024 * 1024:
        with path.open("rb") as fh:
            return _sha256_bytes(iter(lambda: fh.read(1024 * 1024), b""))

    with path.open("rb") as fh:
        head = fh.read(1024 * 1024)
        fh.seek(max(0, size - (1024 * 1024)))
        tail = fh.read(1024 * 1024)
    return _sha256_bytes([head, tail, str(size).encode("utf-8")])


def compute_model_files_fingerprint(model_root: Optional[Path]) -> str:
    if model_root is None:
        return _sha256_text("model_root=unresolved")
    root = model_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return _sha256_text(f"model_root=missing:{root}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    meta_lines: list[str] = []
    hashed_lines: list[str] = []
    weight_candidates: list[Path] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        meta_lines.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}")
        lower = rel.lower()
        name = path.name.lower()
        if name in _KEY_FILENAMES or lower.endswith("/1_pooling/config.json"):
            hashed_lines.append(f"{rel}|{_hash_file(path)}")
        if (
            name.startswith("pytorch_model")
            or name.endswith(".safetensors")
            or name.endswith(".bin")
            or name.endswith(".onnx")
        ):
            weight_candidates.append(path)

    if weight_candidates:
        first_weight = sorted(weight_candidates)[0]
        rel = first_weight.relative_to(root).as_posix()
        hashed_lines.append(f"{rel}|{_hash_file(first_weight)}")

    payload = "\n".join(
        [
            "scheme=model_files_v1",
            f"root={root.as_posix()}",
            "meta:",
            *meta_lines,
            "hashes:",
            *sorted(hashed_lines),
        ]
    )
    return _sha256_text(payload)


def _canonical_hash(payload: dict[str, object]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_text(text)


def build_torch_runtime_fingerprint(
    *,
    model_id: str,
    local_model_path: str,
    model_files_fingerprint: str,
    torch_version: str,
    transformers_version: str,
    sentence_transformers_version: str,
    pooling: str,
    max_length: int,
    dtype: str,
    normalize: bool,
) -> str:
    return _canonical_hash(
        {
            "provider": "torch",
            "model_id": model_id,
            "local_model_path": local_model_path,
            "model_files_fingerprint": model_files_fingerprint,
            "torch_version": torch_version,
            "transformers_version": transformers_version,
            "sentence_transformers_version": sentence_transformers_version,
            "pooling": pooling,
            "max_length": int(max_length),
            "dtype": dtype,
            "normalize": bool(normalize),
        }
    )


def build_ollama_runtime_fingerprint(*, base_url: str, model_id: str) -> str:
    return _canonical_hash(
        {
            "provider": "ollama",
            "base_url": base_url.rstrip("/"),
            "model_id": model_id,
        }
    )
