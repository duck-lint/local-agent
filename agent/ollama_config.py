from __future__ import annotations

import os
from typing import Mapping, Optional
from urllib.parse import urlparse

OLLAMA_BASE_URL_ENV = "LOCAL_AGENT_OLLAMA_BASE_URL"
OLLAMA_BASE_URL_FALLBACK_ENV = "OLLAMA_BASE_URL"


def _normalize_ollama_base_url(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError("Ollama base URL is empty")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Ollama base URL must include http or https scheme")
    if not parsed.netloc:
        raise ValueError("Ollama base URL must include host or host:port")
    return text.rstrip("/")


def resolve_ollama_base_url(
    *,
    cli_override: Optional[str],
    env: Optional[Mapping[str, str]],
    config_value: Optional[str],
    default: Optional[str] = None,
) -> str:
    environment = env or os.environ
    candidates = [
        cli_override,
        environment.get(OLLAMA_BASE_URL_ENV),
        environment.get(OLLAMA_BASE_URL_FALLBACK_ENV),
        config_value,
        default,
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        return _normalize_ollama_base_url(candidate)

    raise ValueError(
        "Ollama base URL not configured. Set --ollama-base-url, "
        f"{OLLAMA_BASE_URL_ENV}, {OLLAMA_BASE_URL_FALLBACK_ENV}, or configure ollama_base_url."
    )
