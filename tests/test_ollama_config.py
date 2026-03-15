from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.ollama_config import (
    OLLAMA_BASE_URL_ENV,
    OLLAMA_BASE_URL_FALLBACK_ENV,
    resolve_ollama_base_url,
)


class OllamaConfigTests(unittest.TestCase):
    def test_cli_override_wins_and_normalizes(self) -> None:
        base_url = resolve_ollama_base_url(
            cli_override="https://cli.example:11434/",
            env={OLLAMA_BASE_URL_ENV: "http://env.example:11434"},
            config_value="http://config.example:11434",
        )
        self.assertEqual(base_url, "https://cli.example:11434")

    def test_prefers_local_agent_env_over_generic(self) -> None:
        base_url = resolve_ollama_base_url(
            cli_override=None,
            env={
                OLLAMA_BASE_URL_ENV: "http://lan-host:11434",
                OLLAMA_BASE_URL_FALLBACK_ENV: "http://generic:11434",
            },
            config_value="http://config.example:11434",
        )
        self.assertEqual(base_url, "http://lan-host:11434")

    def test_uses_generic_env_when_primary_missing(self) -> None:
        base_url = resolve_ollama_base_url(
            cli_override=None,
            env={OLLAMA_BASE_URL_FALLBACK_ENV: "http://generic:11434/"},
            config_value="http://config.example:11434",
        )
        self.assertEqual(base_url, "http://generic:11434")

    def test_env_none_reads_process_environment(self) -> None:
        with patch.dict(os.environ, {OLLAMA_BASE_URL_ENV: "http://env.example:11434"}, clear=True):
            base_url = resolve_ollama_base_url(
                cli_override=None,
                env=None,
                config_value="http://config.example:11434",
            )
        self.assertEqual(base_url, "http://env.example:11434")

    def test_empty_env_does_not_fall_back_to_process_environment(self) -> None:
        with patch.dict(os.environ, {OLLAMA_BASE_URL_ENV: "http://env.example:11434"}, clear=True):
            base_url = resolve_ollama_base_url(
                cli_override=None,
                env={},
                config_value="http://config.example:11434/",
            )
        self.assertEqual(base_url, "http://config.example:11434")

    def test_config_used_when_no_overrides(self) -> None:
        base_url = resolve_ollama_base_url(
            cli_override=None,
            env={},
            config_value="http://config.example:11434/",
        )
        self.assertEqual(base_url, "http://config.example:11434")

    def test_default_used_when_config_missing(self) -> None:
        base_url = resolve_ollama_base_url(
            cli_override=None,
            env={},
            config_value=None,
            default="http://default:11434",
        )
        self.assertEqual(base_url, "http://default:11434")

    def test_invalid_urls_are_rejected(self) -> None:
        invalid_values = [
            "localhost:11434",
            "http://host:11434/ollama",
            "http://host:11434?x=1",
            "http://host:11434/#frag",
            "http://host:11434/;params",
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_ollama_base_url(
                        cli_override=value,
                        env={},
                        config_value=None,
                        default=None,
                    )


if __name__ == "__main__":
    unittest.main()
