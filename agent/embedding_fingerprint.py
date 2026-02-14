from __future__ import annotations

from array import array
import hashlib
import sys
from typing import Iterable

PREPROCESS_NAME_OBSIDIAN_V1 = "obsidian_v1"

_CANONICAL_STEPS = {
    PREPROCESS_NAME_OBSIDIAN_V1: (
        "normalize_newlines=\\n;"
        "strip_trailing_whitespace_per_line=true;"
        "utf8_replace=true;"
        "prefix=rel_path+heading_path"
    )
}


def normalize_embedding_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines)


def build_chunk_embedding_input(
    *,
    preprocess_name: str,
    rel_path: str,
    heading_path: str,
    text: str,
) -> str:
    if preprocess_name != PREPROCESS_NAME_OBSIDIAN_V1:
        raise ValueError(f"Unsupported embedding preprocess: {preprocess_name}")
    normalized = normalize_embedding_text(text)
    rel = normalize_embedding_text(rel_path)
    heading = normalize_embedding_text(heading_path)
    return f"rel_path={rel}\nheading_path={heading}\n\n{normalized}"


def build_query_embedding_input(*, preprocess_name: str, query: str) -> str:
    return build_chunk_embedding_input(
        preprocess_name=preprocess_name,
        rel_path="__query__",
        heading_path="",
        text=query,
    )


def compute_embed_preprocess_sig(preprocess_name: str) -> str:
    steps = _CANONICAL_STEPS.get(preprocess_name)
    if steps is None:
        raise ValueError(f"Unsupported embedding preprocess: {preprocess_name}")
    payload = f"name={preprocess_name}\nsteps={steps}\n"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:32]


def compute_embed_sig(
    *,
    chunk_key: str,
    chunk_sha: str,
    model_id: str,
    dim: int,
    preprocess_sig: str,
) -> str:
    payload = f"{chunk_key}|{chunk_sha}|{model_id}|{int(dim)}|{preprocess_sig}"
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:32]


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
