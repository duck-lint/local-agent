from __future__ import annotations

import os
from typing import Any


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_BASE_URL_ENV_VAR = "LOCAL_AGENT_OLLAMA_BASE_URL"


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_base_url(base_url: str) -> str:
    """
    Normalize the Ollama base URL so callers can safely append paths like "/api/...".

    Currently this just strips trailing slashes to avoid "//api/..." when concatenating,
    while preserving the scheme, host, and port as provided.
    """
    return base_url.rstrip("/")


def resolve_ollama_base_url(cfg: dict[str, Any] | None, *, env_value: str | None = None) -> str:
    env_base_url = _string_value(
        env_value if env_value is not None else os.environ.get(OLLAMA_BASE_URL_ENV_VAR)
    )
    if env_base_url is not None:
        return _normalize_base_url(env_base_url)

    if isinstance(cfg, dict):
        configured_base_url = _string_value(cfg.get("ollama_base_url"))
        if configured_base_url is not None:
            return _normalize_base_url(configured_base_url)

    return DEFAULT_OLLAMA_BASE_URL
