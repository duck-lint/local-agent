from __future__ import annotations

from typing import Any

import requests

from agent.embedders.ollama import normalize_ollama_base_url, redact_ollama_error_detail


def _ollama_request_error(action: str, base_url: str, exc: Exception) -> RuntimeError:
    detail = redact_ollama_error_detail(str(exc), normalize_ollama_base_url(base_url))
    return RuntimeError(f"Ollama {action} failed: {detail}")


def ensure_ollama_up(base_url: str, timeout_s: int) -> None:
    normalized_base_url = normalize_ollama_base_url(base_url)
    try:
        response = requests.get(f"{normalized_base_url}/api/tags", timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise _ollama_request_error("reachability check", normalized_base_url, exc) from exc


def ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    normalized_base_url = normalize_ollama_base_url(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    try:
        response = requests.post(
            f"{normalized_base_url}/api/chat",
            json=payload,
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise _ollama_request_error("chat request", normalized_base_url, exc) from exc
    if not isinstance(data, dict):
        raise ValueError("Unexpected Ollama response type; expected JSON object")
    return data


def get_assistant_text(resp: dict[str, Any]) -> str:
    message = resp.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    return content.strip()
