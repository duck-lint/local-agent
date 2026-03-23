from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from agent.app import LocalAgentApp
from agent.app_types import AppRoots
from agent.config import build_app_config, deep_merge_config, DEFAULT_CONFIG
from agent.tools import configure_tool_security


class DummyEmbedder:
    def __init__(self, runtime_fp: str = "dummy-runtime-v1") -> None:
        self._runtime_fp = runtime_fp
        self._embed_dim = 4

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def runtime_fingerprint(self) -> str:
        return self._runtime_fp

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            base = float(len(text) % 10)
            out.append([base + 0.1, base + 0.2, base + 0.3, base + 0.4])
        return out


def dummy_embedder_factory(embeddings_cfg, base_url: str, timeout_s: int) -> DummyEmbedder:
    _ = embeddings_cfg, base_url, timeout_s
    return DummyEmbedder()


class AppFixture:
    def __init__(self) -> None:
        temp_root = Path(__file__).resolve().parent.parent / ".tmp" / "test-runtime"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.tmp_path = temp_root / f"case-{uuid.uuid4().hex}"
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.workroot = self.tmp_path / "workroot"
        self.workroot.mkdir(parents=True, exist_ok=True)
        (self.workroot / "allowed" / "corpus").mkdir(parents=True, exist_ok=True)
        (self.workroot / "allowed" / "scratch").mkdir(parents=True, exist_ok=True)
        (self.workroot / "runs").mkdir(parents=True, exist_ok=True)
        self.config_path = self.tmp_path / "repo" / "configs" / "default.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text("model: test-model\n", encoding="utf-8")
        self.raw_config = deep_merge_config(
            DEFAULT_CONFIG,
            {
                "model": "test-model",
                "model_fast": "test-model-fast",
                "model_big": "test-model-big",
                "timeout_s": 1,
                "timeout_s_big_second": 1,
                "max_tokens_big_second": 128,
                "temperature": 0.0,
                "workroot": str(self.workroot),
                "corpus": {
                    "db_path": "index/index.sqlite",
                    "max_chars": 120,
                    "overlap": 20,
                },
                "embeddings": {
                    "provider": "torch",
                    "model_id": "dummy-embed-model",
                    "preprocess": "obsidian_v1",
                    "batch_size": 8,
                    "torch": {
                        "local_model_path": "",
                        "cache_dir": "",
                        "device": "cpu",
                        "dtype": "float32",
                        "batch_size": 8,
                        "max_length": 128,
                        "pooling": "mean",
                        "normalize": True,
                        "trust_remote_code": False,
                        "offline_only": True,
                    },
                },
            },
        )
        self.app_config = build_app_config(self.raw_config)
        self.roots = AppRoots(
            config_root=self.config_path.parent.parent,
            package_root=self.config_path.parent.parent,
            workroot=self.workroot,
            security_root=self.workroot,
        )
        configure_tool_security(
            {
                "allowed_roots": self.app_config.security.allowed_roots,
                "allowed_exts": self.app_config.security.allowed_exts,
                "deny_absolute_paths": self.app_config.security.deny_absolute_paths,
                "deny_hidden_paths": self.app_config.security.deny_hidden_paths,
                "allow_any_path": self.app_config.security.allow_any_path,
                "auto_create_allowed_roots": self.app_config.security.auto_create_allowed_roots,
                "roots_must_be_within_security_root": self.app_config.security.roots_must_be_within_security_root,
            },
            workspace_root=self.workroot,
            resolved_config_path=self.config_path,
        )

    def close(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def corpus_path(self, name: str) -> Path:
        return self.workroot / "allowed" / "corpus" / name

    def write_corpus_note(self, name: str, content: str) -> Path:
        path = self.corpus_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def scratch_path(self, name: str) -> Path:
        return self.workroot / "allowed" / "scratch" / name

    def write_scratch_note(self, name: str, content: str) -> Path:
        path = self.scratch_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def build_app(self, *, config_override: Optional[dict] = None) -> LocalAgentApp:
        if config_override:
            raw_config = deep_merge_config(self.raw_config, config_override)
            app_config = build_app_config(raw_config)
        else:
            raw_config = self.raw_config
            app_config = self.app_config
        return LocalAgentApp(
            config=app_config,
            roots=self.roots,
            raw_config=raw_config,
            resolved_config_path=self.config_path,
        )
