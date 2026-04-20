from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class SourceConfig:
    name: str
    root: str
    kind: str


@dataclass(frozen=True)
class SecurityConfig:
    allowed_roots: list[str]
    allowed_exts: list[str]
    deny_absolute_paths: bool
    deny_hidden_paths: bool
    allow_any_path: bool
    auto_create_allowed_roots: bool
    roots_must_be_within_security_root: bool


@dataclass(frozen=True)
class CorpusConfig:
    db_path: str
    sources: list[SourceConfig]
    max_chars: int


@dataclass(frozen=True)
class EmbeddingsTorchConfig:
    local_model_path: str
    cache_dir: str
    device: str
    dtype: str
    batch_size: int
    max_length: int
    pooling: str
    normalize: bool
    trust_remote_code: bool
    offline_only: bool


@dataclass(frozen=True)
class EmbeddingsConfig:
    db_path: str
    provider: str
    model_id: str
    preprocess: str
    chunk_preprocess_sig: str
    query_preprocess_sig: str
    batch_size: int
    torch: EmbeddingsTorchConfig


@dataclass(frozen=True)
class RetrievalConfig:
    lexical_k: int
    vector_k: int
    vector_fetch_k: int
    rel_path_prefix: str
    fusion: str


@dataclass(frozen=True)
class CitationValidationConfig:
    enabled: bool
    strict: bool
    require_in_snapshot: bool
    heading_match: str
    normalize_heading: bool


@dataclass(frozen=True)
class GroundingConfig:
    evidence_top_n: int
    citation_validation: CitationValidationConfig


@dataclass(frozen=True)
class RunsConfig:
    log_evidence_excerpts: bool
    max_total_evidence_chars: int
    max_excerpt_chars: int


@dataclass(frozen=True)
class MemoryConfig:
    db_path: str
    enabled: bool


@dataclass(frozen=True)
class AppConfig:
    model: str
    model_fast: str
    model_big: str
    prefer_fast: bool
    big_triggers: list[str]
    ollama_base_url: str
    max_tokens: int
    max_tokens_big_second: int
    temperature: float
    timeout_s: int
    timeout_s_big_second: int
    max_chars_full_read: int
    workroot: str
    security: SecurityConfig
    corpus: CorpusConfig
    embeddings: EmbeddingsConfig
    retrieval: RetrievalConfig
    grounding: GroundingConfig
    runs: RunsConfig
    memory: MemoryConfig


@dataclass(frozen=True)
class AppRoots:
    config_root: Optional[Path]
    package_root: Path
    workroot: Optional[Path]
    security_root: Path


@dataclass(frozen=True)
class DocumentRecord:
    doc_key: str
    source_name: str
    rel_path: str
    source_uri: str
    source_hash: str
    abs_path: str
    title: str
    folder: str
    doc_type: str
    sensitivity: str
    entry_date: Optional[str]
    source_date: Optional[str]
    frontmatter: dict[str, Any]
    yaml_present: int
    yaml_parse_ok: Optional[int]
    yaml_error: Optional[str]
    mtime: float
    size: int


@dataclass(frozen=True)
class ChunkRecord:
    chunk_key: str
    doc_key: str
    chunk_kind: str
    chunk_index: int
    section_index: int
    heading_path: str
    chunk_anchor: str
    chunk_title: str
    text: str
    chunk_hash: str
    start_char: int
    end_char: int
    out_links: list[dict[str, str]]
    section_ordinal: Optional[int] = None


@dataclass(frozen=True)
class ChatResult:
    ok: bool
    text: str
    model_used: str
    run_dir: Path
    record: dict[str, Any]
    raw_response: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class GroundedAnswerResult:
    ok: bool
    text: str
    model_used: str
    run_dir: Path
    record: dict[str, Any]
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class CorpusSyncResult:
    sources_total: int
    docs_scanned: int
    docs_changed: int
    docs_unchanged: int
    docs_pruned: int
    chunks_written: int
    total_docs: int
    total_chunks: int
    errors: list[str]
    corpus_db_path: str
    corpus_contract_sig: str


@dataclass(frozen=True)
class EmbeddingSyncResult:
    total_chunks: int
    existing_embeddings: int
    embeddings_total_before: int
    embeddings_total_after: int
    orphan_embeddings_before: int
    orphan_embeddings_pruned: int
    missing: int
    outdated: int
    embedded_written: int
    skipped_ok: int
    errors: list[str]
    dim: Optional[int]
    provider: str
    model_id: str
    embed_runtime_fingerprint: str
    chunk_preprocess_sig: str
    query_preprocess_sig: str
    vectors_normalized: bool
    embeddings_db_path: str


@dataclass(frozen=True)
class DoctorCheck:
    ok: bool
    code: str
    message: str
    suggested_fix: Optional[str] = None


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    checks: list[DoctorCheck]
    summary: dict[str, Any] = field(default_factory=dict)
