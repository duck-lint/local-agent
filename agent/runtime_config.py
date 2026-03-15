from __future__ import annotations

import os
from typing import Any, Mapping

from agent.embedders.ollama import normalize_ollama_base_url as _normalize_origin


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR = "LOCAL_AGENT_OLLAMA_BASE_URL"
COMPAT_OLLAMA_BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
OLLAMA_BASE_URL_ENV = LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR
OLLAMA_BASE_URL_FALLBACK_ENV = COMPAT_OLLAMA_BASE_URL_ENV_VAR
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
    return _normalize_origin(text)


def resolve_ollama_base_url(
    cfg: Mapping[str, Any] | None = None,
    *,
    cli_base_url: Any = None,
    env_base_url: Any = None,
    env: Mapping[str, str] | None = None,
    cli_override: Any = None,
    config_value: Any = None,
    default: Any = DEFAULT_OLLAMA_BASE_URL,
) -> str:
    """Resolve the effective Ollama base URL.

    Canonical callers should use cli_override/env/config_value/default. The
    cli_base_url/env_base_url aliases remain for older callers and tests.
    """
    env_map = os.environ if env is None else env
    explicit_config = _string(config_value)
    cfg_value = _string(cfg.get("ollama_base_url")) if cfg is not None else ""
    cli_candidate = cli_override if cli_override is not None else cli_base_url
    # config_value is the canonical explicit keyword used by newer callers;
    # cfg["ollama_base_url"] keeps older call sites working when they pass cfg only.
    candidates = [
        cli_candidate,
        env_base_url,
        env_map.get(LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR),
        env_map.get(COMPAT_OLLAMA_BASE_URL_ENV_VAR),
        explicit_config if explicit_config else cfg_value,
        default,
    ]
    for candidate in candidates:
        text = _string(candidate)
        if text:
            return normalize_ollama_base_url(text)
    raise ValueError(
        "Ollama base URL not configured. Set --ollama-base-url, "
        f"{LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR}, {COMPAT_OLLAMA_BASE_URL_ENV_VAR}, "
        "or configure ollama_base_url."
    )
