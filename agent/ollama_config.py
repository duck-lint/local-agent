from __future__ import annotations

import os
from typing import Mapping, Optional
from urllib.parse import urlparse, urlunparse

OLLAMA_BASE_URL_ENV = "LOCAL_AGENT_OLLAMA_BASE_URL"
OLLAMA_BASE_URL_FALLBACK_ENV = "OLLAMA_BASE_URL"


def _normalize_ollama_base_url(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError("Ollama base URL is empty")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Ollama base URL must include http or https scheme")
    if parsed.username or parsed.password:
        raise ValueError("Ollama base URL must not include credentials")
    if not parsed.hostname:
        raise ValueError("Ollama base URL must include host or host:port")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama base URL port is invalid") from exc
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("Ollama base URL must be a bare origin without path, query, or fragment")
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def resolve_ollama_base_url(
    *,
    cli_override: Optional[str],
    env: Optional[Mapping[str, str]],
    config_value: Optional[str],
    default: Optional[str] = None,
) -> str:
    environment = os.environ if env is None else env
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
