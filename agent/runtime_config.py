from __future__ import annotations

import os
from typing import Any, Mapping


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR = "LOCAL_AGENT_OLLAMA_BASE_URL"
COMPAT_OLLAMA_BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
OLLAMA_BASE_URL_ENV_VARS = (
    LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR,
    COMPAT_OLLAMA_BASE_URL_ENV_VAR,
)


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_ollama_base_url(value: Any, *, default: str = DEFAULT_OLLAMA_BASE_URL) -> str:
    text = _string(value) or default
    if "://" not in text:
        text = f"http://{text}"
    return text.rstrip("/")


def resolve_ollama_base_url(
    cfg: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    env_map = os.environ if env is None else env
    for env_var in OLLAMA_BASE_URL_ENV_VARS:
        candidate = _string(env_map.get(env_var))
        if candidate:
            return normalize_ollama_base_url(candidate)
    if cfg is None:
        return normalize_ollama_base_url(None)
    return normalize_ollama_base_url(cfg.get("ollama_base_url"))
