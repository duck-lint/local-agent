from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from agent.app_types import (
    AppConfig,
    AppRoots,
    CitationValidationConfig,
    CorpusConfig,
    EmbeddingsConfig,
    EmbeddingsTorchConfig,
    GroundingConfig,
    MemoryConfig,
    RetrievalConfig,
    RunsConfig,
    SecurityConfig,
    SourceConfig,
)
from agent.runtime_config import (
    DEFAULT_OLLAMA_BASE_URL,
    LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR,
    OLLAMA_BASE_URL_FALLBACK_ENV,
    OLLAMA_BASE_URL_ENV,
    resolve_ollama_base_url,
)


WORKROOT_ENV_VAR = "LOCAL_AGENT_WORKROOT"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "qwen2.5:14b-instruct",
    "model_fast": "qwen2.5:14b-instruct",
    "model_big": "qwen2.5:14b-instruct",
    "prefer_fast": True,
    "big_triggers": [
        "deep",
        "long",
        "essay",
        "synthesize",
        "thorough",
        "in depth",
        "detailed",
        "analysis",
    ],
    "ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
    "max_tokens": 800,
    "max_tokens_big_second": 4500,
    "temperature": 0.2,
    "timeout_s": 300,
    "timeout_s_big_second": 600,
    "max_chars_full_read": 200000,
    "workroot": "../local-agent-workroot/",
    "security": {
        "allowed_roots": ["runs/", "allowed/"],
        "allowed_exts": [".md", ".txt", ".json"],
        "deny_absolute_paths": True,
        "deny_hidden_paths": True,
        "allow_any_path": False,
        "auto_create_allowed_roots": True,
        "roots_must_be_within_security_root": True,
    },
    "corpus": {
        "db_path": "index/index.sqlite",
        "sources": [
            {"name": "corpus", "root": "allowed/corpus/", "kind": "corpus"},
            {"name": "scratch", "root": "allowed/scratch/", "kind": "scratch"},
        ],
        "max_chars": 1200,
    },
    "embeddings": {
        "db_path": "embeddings/db/embeddings.sqlite",
        "provider": "torch",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "preprocess": "obsidian_v1",
        "chunk_preprocess_sig": "",
        "query_preprocess_sig": "",
        "batch_size": 64,
        "torch": {
            "local_model_path": "",
            "cache_dir": "",
            "device": "auto",
            "dtype": "float16",
            "batch_size": 64,
            "max_length": 512,
            "pooling": "mean",
            "normalize": True,
            "trust_remote_code": False,
            "offline_only": True,
        },
    },
    "retrieval": {
        "lexical_k": 20,
        "vector_k": 20,
        "vector_fetch_k": 0,
        "rel_path_prefix": "",
        "fusion": "simple_union",
    },
    "grounding": {
        "evidence_top_n": 8,
        "citation_validation": {
            "enabled": True,
            "strict": False,
            "require_in_snapshot": True,
            "heading_match": "prefix",
            "normalize_heading": True,
        },
    },
    "runs": {
        "log_evidence_excerpts": True,
        "max_total_evidence_chars": 200000,
        "max_excerpt_chars": 1200,
    },
    "memory": {
        "db_path": "memory/durable.sqlite",
        "enabled": True,
    },
}


def _obsolete_keys() -> set[str]:
    pieces = (("ph", "ase2"), ("ph", "ase3"), ("st", "age"), ("st", "ages"), ("le", "gacy"))
    return {"".join(item) for item in pieces}


OBSOLETE_KEYS = _obsolete_keys()


def _string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            out[key] = deep_merge_config(value, {})
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_config(out[key], value)  # type: ignore[arg-type]
        elif isinstance(value, dict):
            out[key] = deep_merge_config({}, value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _reject_obsolete_keys(obj: Any, *, path: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).strip().lower() in OBSOLETE_KEYS:
                location = path or "<root>"
                raise ValueError(
                    f"Obsolete config key '{key}' found at {location}. "
                    "Use corpus/embeddings/retrieval/grounding/runs/memory vocabulary only."
                )
            next_path = f"{path}.{key}" if path else str(key)
            _reject_obsolete_keys(value, path=next_path)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _reject_obsolete_keys(item, path=f"{path}[{idx}]")


def discover_config_path(
    start_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    _ = start_dir
    root = (repo_root or Path(__file__).resolve().parent.parent).resolve()
    candidate = root / "configs" / "default.yaml"
    if candidate.exists():
        return candidate
    return None


def apply_env_config_overrides(
    cfg: dict[str, Any],
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    ollama_base_url = _string(env.get(LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR))
    if not ollama_base_url:
        return cfg
    return deep_merge_config(cfg, {"ollama_base_url": ollama_base_url})


def load_config_with_path(
    start_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    cfg_path = discover_config_path(start_dir=start_dir, repo_root=repo_root)
    if cfg_path is None:
        merged = apply_env_config_overrides(deep_merge_config(DEFAULT_CONFIG, {}))
        return merged, None
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path}: configs/default.yaml must contain a mapping/object")
    _reject_obsolete_keys(data)
    merged = deep_merge_config(DEFAULT_CONFIG, data)
    merged = apply_env_config_overrides(merged)
    return merged, cfg_path


def load_config() -> dict[str, Any]:
    return load_config_with_path()[0]


def config_root_from_config_path(config_path: Optional[Path]) -> Optional[Path]:
    if config_path is None:
        return None
    cfg_parent = config_path.resolve().parent
    if cfg_parent.name.lower() == "configs":
        return cfg_parent.parent
    return cfg_parent


def _resolve_candidate_root(raw_value: Optional[str], base_dir: Path) -> Optional[Path]:
    if not raw_value:
        return None
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def resolve_runtime_roots(
    resolved_config_path: Optional[Path],
    cfg: Mapping[str, Any],
    cli_workroot: Optional[str],
    cwd: Optional[Path] = None,
    env_workroot: Optional[str] = None,
    package_root: Optional[Path] = None,
) -> AppRoots:
    cwd_resolved = (cwd or Path.cwd()).resolve()
    package_root_resolved = (package_root or Path(__file__).resolve().parent.parent).resolve()
    config_root = config_root_from_config_path(resolved_config_path)

    cli_value = _string(cli_workroot)
    env_value = _string(env_workroot if env_workroot is not None else os.environ.get(WORKROOT_ENV_VAR))
    cfg_value = _string(cfg.get("workroot"))

    selected_workroot = cli_value or env_value or cfg_value
    relative_base = config_root or cwd_resolved
    workroot = _resolve_candidate_root(selected_workroot, relative_base)
    security_root = workroot or config_root or cwd_resolved

    return AppRoots(
        config_root=config_root,
        package_root=package_root_resolved,
        workroot=workroot,
        security_root=security_root,
    )


def root_log_fields(roots: AppRoots) -> dict[str, Optional[str]]:
    return {
        "config_root": str(roots.config_root) if roots.config_root else None,
        "package_root": str(roots.package_root),
        "workroot": str(roots.workroot) if roots.workroot else None,
        "security_root": str(roots.security_root),
    }


def _parse_source_configs(raw: Any) -> list[SourceConfig]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("corpus.sources must be a non-empty list")
    seen_names: set[str] = set()
    out: list[SourceConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("corpus.sources entries must be objects")
        name = _string(item.get("name"))
        root = _string(item.get("root"))
        kind = _string(item.get("kind"))
        if not name or not root or not kind:
            raise ValueError("corpus.sources entries require name, root, and kind")
        if name in seen_names:
            raise ValueError(f"Duplicate corpus source name: {name}")
        seen_names.add(name)
        out.append(SourceConfig(name=name, root=root, kind=kind))
    return out


def build_app_config(raw_cfg: Mapping[str, Any]) -> AppConfig:
    cfg = deep_merge_config(DEFAULT_CONFIG, dict(raw_cfg))
    _reject_obsolete_keys(cfg)

    security_raw = cfg.get("security") if isinstance(cfg.get("security"), dict) else {}
    security = SecurityConfig(
        allowed_roots=[str(item) for item in security_raw.get("allowed_roots", [])],
        allowed_exts=[str(item) for item in security_raw.get("allowed_exts", [])],
        deny_absolute_paths=_as_bool(security_raw.get("deny_absolute_paths"), True),
        deny_hidden_paths=_as_bool(security_raw.get("deny_hidden_paths"), True),
        allow_any_path=_as_bool(security_raw.get("allow_any_path"), False),
        auto_create_allowed_roots=_as_bool(security_raw.get("auto_create_allowed_roots"), True),
        roots_must_be_within_security_root=_as_bool(
            security_raw.get("roots_must_be_within_security_root"),
            True,
        ),
    )

    corpus_raw = cfg.get("corpus") if isinstance(cfg.get("corpus"), dict) else {}
    corpus = CorpusConfig(
        db_path=_string(corpus_raw.get("db_path"), DEFAULT_CONFIG["corpus"]["db_path"]),
        sources=_parse_source_configs(corpus_raw.get("sources", DEFAULT_CONFIG["corpus"]["sources"])),
        max_chars=_as_int(corpus_raw.get("max_chars"), DEFAULT_CONFIG["corpus"]["max_chars"]),
    )
    if corpus.max_chars <= 0:
        raise ValueError("corpus.max_chars must be > 0")

    embeddings_raw = cfg.get("embeddings") if isinstance(cfg.get("embeddings"), dict) else {}
    torch_raw = embeddings_raw.get("torch") if isinstance(embeddings_raw.get("torch"), dict) else {}
    torch_cfg = EmbeddingsTorchConfig(
        local_model_path=_string(torch_raw.get("local_model_path")),
        cache_dir=_string(torch_raw.get("cache_dir")),
        device=_string(torch_raw.get("device"), "auto").lower(),
        dtype=_string(torch_raw.get("dtype"), "float16").lower(),
        batch_size=_as_int(torch_raw.get("batch_size"), 64),
        max_length=_as_int(torch_raw.get("max_length"), 512),
        pooling=_string(torch_raw.get("pooling"), "mean").lower(),
        normalize=_as_bool(torch_raw.get("normalize"), True),
        trust_remote_code=_as_bool(torch_raw.get("trust_remote_code"), False),
        offline_only=_as_bool(torch_raw.get("offline_only"), True),
    )
    embeddings = EmbeddingsConfig(
        db_path=_string(embeddings_raw.get("db_path"), DEFAULT_CONFIG["embeddings"]["db_path"]),
        provider=_string(embeddings_raw.get("provider"), "torch").lower(),
        model_id=_string(embeddings_raw.get("model_id"), DEFAULT_CONFIG["embeddings"]["model_id"]),
        preprocess=_string(embeddings_raw.get("preprocess"), "obsidian_v1"),
        chunk_preprocess_sig=_string(embeddings_raw.get("chunk_preprocess_sig")),
        query_preprocess_sig=_string(embeddings_raw.get("query_preprocess_sig")),
        batch_size=_as_int(embeddings_raw.get("batch_size"), 64),
        torch=torch_cfg,
    )

    retrieval_raw = cfg.get("retrieval") if isinstance(cfg.get("retrieval"), dict) else {}
    retrieval = RetrievalConfig(
        lexical_k=_as_int(retrieval_raw.get("lexical_k"), 20),
        vector_k=_as_int(retrieval_raw.get("vector_k"), 20),
        vector_fetch_k=_as_int(retrieval_raw.get("vector_fetch_k"), 0),
        rel_path_prefix=_string(retrieval_raw.get("rel_path_prefix")),
        fusion=_string(retrieval_raw.get("fusion"), "simple_union"),
    )

    grounding_raw = cfg.get("grounding") if isinstance(cfg.get("grounding"), dict) else {}
    citation_raw = (
        grounding_raw.get("citation_validation")
        if isinstance(grounding_raw.get("citation_validation"), dict)
        else {}
    )
    grounding = GroundingConfig(
        evidence_top_n=_as_int(grounding_raw.get("evidence_top_n"), 8),
        citation_validation=CitationValidationConfig(
            enabled=_as_bool(citation_raw.get("enabled"), True),
            strict=_as_bool(citation_raw.get("strict"), False),
            require_in_snapshot=_as_bool(citation_raw.get("require_in_snapshot"), True),
            heading_match=_string(citation_raw.get("heading_match"), "prefix").lower(),
            normalize_heading=_as_bool(citation_raw.get("normalize_heading"), True),
        ),
    )
    if grounding.citation_validation.heading_match not in {"exact", "prefix", "ignore"}:
        raise ValueError("grounding.citation_validation.heading_match must be exact|prefix|ignore")

    runs_raw = cfg.get("runs") if isinstance(cfg.get("runs"), dict) else {}
    runs = RunsConfig(
        log_evidence_excerpts=_as_bool(runs_raw.get("log_evidence_excerpts"), True),
        max_total_evidence_chars=_as_int(runs_raw.get("max_total_evidence_chars"), 200000),
        max_excerpt_chars=_as_int(runs_raw.get("max_excerpt_chars"), 1200),
    )

    memory_raw = cfg.get("memory") if isinstance(cfg.get("memory"), dict) else {}
    memory = MemoryConfig(
        db_path=_string(memory_raw.get("db_path"), DEFAULT_CONFIG["memory"]["db_path"]),
        enabled=_as_bool(memory_raw.get("enabled"), True),
    )

    return AppConfig(
        model=_string(cfg.get("model"), DEFAULT_CONFIG["model"]),
        model_fast=_string(cfg.get("model_fast"), _string(cfg.get("model"), DEFAULT_CONFIG["model"])),
        model_big=_string(cfg.get("model_big"), _string(cfg.get("model"), DEFAULT_CONFIG["model"])),
        prefer_fast=_as_bool(cfg.get("prefer_fast"), True),
        big_triggers=[str(item).strip().lower() for item in cfg.get("big_triggers", []) if str(item).strip()],
        ollama_base_url=resolve_ollama_base_url(cfg),
        max_tokens=_as_int(cfg.get("max_tokens"), DEFAULT_CONFIG["max_tokens"]),
        max_tokens_big_second=_as_int(
            cfg.get("max_tokens_big_second"),
            DEFAULT_CONFIG["max_tokens_big_second"],
        ),
        temperature=_as_float(cfg.get("temperature"), DEFAULT_CONFIG["temperature"]),
        timeout_s=_as_int(cfg.get("timeout_s"), DEFAULT_CONFIG["timeout_s"]),
        timeout_s_big_second=_as_int(
            cfg.get("timeout_s_big_second"),
            DEFAULT_CONFIG["timeout_s_big_second"],
        ),
        max_chars_full_read=_as_int(cfg.get("max_chars_full_read"), DEFAULT_CONFIG["max_chars_full_read"]),
        workroot=_string(cfg.get("workroot"), DEFAULT_CONFIG["workroot"]),
        security=security,
        corpus=corpus,
        embeddings=embeddings,
        retrieval=retrieval,
        grounding=grounding,
        runs=runs,
        memory=memory,
    )


def config_summary_for_cli(app_config: AppConfig) -> dict[str, Any]:
    return {
        "ollama_base_url_precedence": [
            "--ollama-base-url",
            OLLAMA_BASE_URL_ENV,
            OLLAMA_BASE_URL_FALLBACK_ENV,
            "ollama_base_url",
        ],
        "workroot_precedence": ["--workroot", WORKROOT_ENV_VAR, "workroot"],
        "models": {
            "model": app_config.model,
            "model_fast": app_config.model_fast,
            "model_big": app_config.model_big,
        },
    }
