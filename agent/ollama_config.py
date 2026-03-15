from __future__ import annotations

from typing import Mapping, Optional

from agent.runtime_config import (
    DEFAULT_OLLAMA_BASE_URL,
    LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR,
    OLLAMA_BASE_URL_FALLBACK_ENV,
    OLLAMA_BASE_URL_ENV,
    normalize_ollama_base_url,
    resolve_ollama_base_url as _resolve_runtime_ollama_base_url,
)

OLLAMA_BASE_URL_ENV_VAR = LOCAL_AGENT_OLLAMA_BASE_URL_ENV_VAR


def resolve_ollama_base_url(
    *,
    cli_override: Optional[str],
    env: Optional[Mapping[str, str]],
    config_value: Optional[str],
    default: Optional[str] = DEFAULT_OLLAMA_BASE_URL,
) -> str:
    return _resolve_runtime_ollama_base_url(
        env=env,
        cli_override=cli_override,
        config_value=config_value,
        default=default,
    )
