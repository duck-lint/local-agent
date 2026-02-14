from __future__ import annotations

from array import array
import hashlib
import math
import sys
from typing import Iterable

PREPROCESS_NAME_OBSIDIAN_V1 = "obsidian_v1"

_COMMON_STEPS = (
    "normalize_newlines=\\n;"
    "strip_trailing_whitespace_per_line=true;"
    "utf8_replace=true"
)
_CHUNK_FRAMING_STEPS = "prefix=rel_path+heading_path"
_QUERY_FRAMING_STEPS = "prefix=none"


def normalize_embedding_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines)


def preprocess_chunk_text(
    *,
    rel_path: str,
    heading_path: str,
    chunk_text: str,
    preprocess_name: str,
) -> str:
    if preprocess_name != PREPROCESS_NAME_OBSIDIAN_V1:
        raise ValueError(f"Unsupported chunk preprocess: {preprocess_name}")
    normalized = normalize_embedding_text(chunk_text)
    rel = normalize_embedding_text(rel_path)
    heading = normalize_embedding_text(heading_path)
    return f"rel_path={rel}\nheading_path={heading}\n\n{normalized}"


def preprocess_query_text(*, query: str, preprocess_name: str) -> str:
    if preprocess_name != PREPROCESS_NAME_OBSIDIAN_V1:
        raise ValueError(f"Unsupported query preprocess: {preprocess_name}")
    return normalize_embedding_text(query)


def compute_chunk_preprocess_sig(preprocess_name: str) -> str:
    if preprocess_name != PREPROCESS_NAME_OBSIDIAN_V1:
        raise ValueError(f"Unsupported chunk preprocess: {preprocess_name}")
    payload = f"name={preprocess_name}\nsteps={_COMMON_STEPS};{_CHUNK_FRAMING_STEPS}\n"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:32]


def compute_query_preprocess_sig(preprocess_name: str) -> str:
    if preprocess_name != PREPROCESS_NAME_OBSIDIAN_V1:
        raise ValueError(f"Unsupported query preprocess: {preprocess_name}")
    payload = f"name={preprocess_name}\nsteps={_COMMON_STEPS};{_QUERY_FRAMING_STEPS}\n"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:32]


def compute_embed_sig(
    *,
    chunk_key: str,
    chunk_sha: str,
    model_id: str,
    dim: int,
    chunk_preprocess_sig: str,
) -> str:
    payload = f"{chunk_key}|{chunk_sha}|{model_id}|{int(dim)}|{chunk_preprocess_sig}"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:32]


def normalize_vector(values: Iterable[float]) -> list[float]:
    vec = [float(v) for v in values]
    if not vec:
        return vec
    norm_sq = 0.0
    for value in vec:
        norm_sq += value * value
    if norm_sq <= 0.0:
        return [0.0 for _ in vec]
    inv = 1.0 / math.sqrt(norm_sq)
    return [value * inv for value in vec]


def pack_vector_f32_le(values: Iterable[float]) -> bytes:
    arr = array("f", [float(v) for v in values])
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def unpack_vector_f32_le(blob: bytes) -> array:
    if len(blob) % 4 != 0:
        raise ValueError("Vector blob byte length must be divisible by 4")
    arr = array("f")
    arr.frombytes(blob)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr
