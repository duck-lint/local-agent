from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.__main__ import main, run_doctor, run_embed, run_memory
from agent.embed_runtime_fingerprint import build_ollama_runtime_fingerprint
from agent.embeddings_db import connect_db as connect_embeddings_db
from agent.indexer import SourceSpec, index_sources
from agent.tools import configure_tool_security


class _DummyEmbedder:
    def __init__(self, runtime_fp: str = "dummy-runtime-v1") -> None:
        self._runtime_fp = runtime_fp

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            base = float(len(text) % 10)
            out.append([base + 0.1, base + 0.2, base + 0.3, base + 0.4])
        return out

    @property
    def embed_dim(self) -> int:
        return 4

    def runtime_fingerprint(self) -> str:
        return self._runtime_fp


def _dummy_factory(provider: str, model_id: str, base_url: str, timeout_s: int) -> _DummyEmbedder:
    _ = timeout_s
    if provider == "ollama":
        runtime_fp = build_ollama_runtime_fingerprint(base_url=base_url, model_id=model_id)
    else:
        runtime_fp = f"{provider}:{model_id}"
    return _DummyEmbedder(runtime_fp=runtime_fp)


class Phase3CliContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.workroot = self.tmp_path / "workroot"
        self.allowed = self.workroot / "allowed"
        self.corpus = self.allowed / "corpus"
        self.scratch = self.allowed / "scratch"
        self.runs = self.workroot / "runs"
        self.corpus.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workroot / "index" / "index.sqlite"

        (self.corpus / "one.md").write_text("## One\nalpha\n", encoding="utf-8")
        (self.corpus / "two.md").write_text("## Two\nbeta\n", encoding="utf-8")

        self.config_path = self.tmp_path / "repo" / "configs" / "default.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text("model: test\n", encoding="utf-8")

        configure_tool_security(
            {
                "allowed_roots": ["allowed/corpus/", "allowed/scratch/", "runs/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": True,
            },
            workspace_root=self.workroot,
            resolved_config_path=self.config_path,
        )

        summary = index_sources(
            db_path=self.db_path,
            source_specs=[
                SourceSpec(name="corpus", root="allowed/corpus/", kind="corpus"),
                SourceSpec(name="scratch", root="allowed/scratch/", kind="scratch"),
            ],
            security_root=self.workroot,
            scheme="obsidian_v1",
            max_chars=120,
            overlap=20,
        )
        self.assertEqual(summary.errors, [])

        self.cfg = {
            "ollama_base_url": "http://127.0.0.1:11434",
            "timeout_s": 1,
            "security": {
                "allowed_roots": ["allowed/corpus/", "allowed/scratch/", "runs/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": False,
                "roots_must_be_within_security_root": True,
            },
            "phase2": {
                "index_db_path": "index/index.sqlite",
                "sources": [
                    {"name": "corpus", "root": "allowed/corpus/", "kind": "corpus"},
                    {"name": "scratch", "root": "allowed/scratch/", "kind": "scratch"},
                ],
                "chunking": {
                    "scheme": "obsidian_v1",
                    "max_chars": 120,
                    "overlap": 20,
                },
            },
            "phase3": {
                "embeddings_db_path": "embeddings/db/embeddings.sqlite",
                "embed": {
                    "provider": "ollama",
                    "model_id": "nomic-embed-text-v1.5",
                    "preprocess": "obsidian_v1",
                    "chunk_preprocess_sig": "",
                    "query_preprocess_sig": "",
                    "batch_size": 16,
                },
                "retrieve": {
                    "lexical_k": 5,
                    "vector_k": 5,
                    "vector_fetch_k": 0,
                    "rel_path_prefix": "",
                    "fusion": "simple_union",
                },
                "memory": {
                    "durable_db_path": "memory/durable.sqlite",
                    "enabled": True,
                },
            },
        }
        self.roots = {"security_root": self.workroot}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _capture_json(self, func, *args, **kwargs) -> dict:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = func(*args, **kwargs)
        self.assertEqual(exit_code, 0)
        text = buffer.getvalue().strip().splitlines()[-1]
        return json.loads(text)

    def test_embed_json_counts_match_db(self) -> None:
        payload = self._capture_json(
            run_embed,
            self.cfg,
            json_output=True,
            dry_run=False,
            embedder_factory=_dummy_factory,
            resolved_config_path=self.config_path,
            roots=self.roots,
        )
        self.assertIn("total_chunks", payload)
        self.assertIn("embedded_written", payload)
        self.assertIn("orphan_embeddings_before", payload)
        self.assertIn("orphan_embeddings_pruned", payload)
        self.assertIn("embeddings_total_before", payload)
        self.assertIn("embeddings_total_after", payload)

        embeddings_db = Path(payload["embeddings_db"])
        with connect_embeddings_db(embeddings_db) as conn:
            count = int(conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"])
        self.assertEqual(payload["embedded_written"], count)

    def test_doctor_json_contains_phase3_fields(self) -> None:
        self._capture_json(
            run_embed,
            self.cfg,
            json_output=True,
            dry_run=False,
            embedder_factory=_dummy_factory,
            resolved_config_path=self.config_path,
            roots=self.roots,
        )
        payload = self._capture_json(
            run_doctor,
            self.cfg,
            json_output=True,
            check_ollama=False,
            require_phase3=True,
            resolved_config_path=self.config_path,
            roots=self.roots,
        )
        self.assertIn("phase3", payload)
        phase3 = payload["phase3"]
        self.assertIn("missing_embeddings", phase3)
        self.assertIn("outdated_embeddings", phase3)
        self.assertIn("embeddings_total", phase3)
        self.assertIn("orphan_embeddings", phase3)
        self.assertIn("retrieval_smoke_ran", phase3)
        self.assertIn("retrieval_smoke_ok", phase3)
        self.assertIn("retrieval_smoke_reason", phase3)

    def test_memory_list_json_shape(self) -> None:
        add_payload = self._capture_json(
            run_memory,
            self.cfg,
            action="add",
            memory_type="preference",
            source="manual",
            content="prefer concise responses",
            chunk_keys=None,
            json_output=True,
            resolved_config_path=self.config_path,
            roots=self.roots,
        )
        self.assertIn("memory_id", add_payload)

        list_payload = self._capture_json(
            run_memory,
            self.cfg,
            action="list",
            json_output=True,
            resolved_config_path=self.config_path,
            roots=self.roots,
        )
        self.assertIn("count", list_payload)
        self.assertIn("items", list_payload)
        self.assertGreaterEqual(list_payload["count"], 1)


class MainLazyOllamaResolutionTests(unittest.TestCase):
    def test_doctor_no_ollama_ignores_invalid_ollama_url_at_startup(self) -> None:
        loaded_cfg = {"ollama_base_url": "http://example.test:11434/api?token=secret"}
        config_path = Path("/tmp/configs/default.yaml")

        with patch("agent.__main__.load_config_with_path", return_value=(loaded_cfg, config_path)):
            with patch("agent.__main__.configure_tool_security"):
                with patch("agent.__main__.run_doctor", return_value=0) as run_doctor_mock:
                    with patch.dict(
                        os.environ,
                        {"LOCAL_AGENT_OLLAMA_BASE_URL": "http://example.test:11434/api?token=secret"},
                        clear=False,
                    ):
                        with patch("sys.argv", ["agent", "doctor", "--no-ollama", "--json"]):
                            exit_code = main()

        self.assertEqual(exit_code, 0)
        cfg = run_doctor_mock.call_args.args[0]
        self.assertEqual(cfg["ollama_base_url"], loaded_cfg["ollama_base_url"])
        self.assertFalse(run_doctor_mock.call_args.kwargs["check_ollama"])

    def test_embed_torch_ignores_invalid_ollama_url_at_startup(self) -> None:
        loaded_cfg = {
            "ollama_base_url": "http://example.test:11434/api?token=secret",
            "phase3": {"embed": {"provider": "torch", "model_id": "sentence-transformers/all-MiniLM-L6-v2"}},
        }
        config_path = Path("/tmp/configs/default.yaml")

        with patch("agent.__main__.load_config_with_path", return_value=(loaded_cfg, config_path)):
            with patch("agent.__main__.configure_tool_security"):
                with patch("agent.__main__.run_embed", return_value=0) as run_embed_mock:
                    with patch.dict(
                        os.environ,
                        {"LOCAL_AGENT_OLLAMA_BASE_URL": "http://example.test:11434/api?token=secret"},
                        clear=False,
                    ):
                        with patch("sys.argv", ["agent", "embed", "--dry-run", "--json"]):
                            exit_code = main()

        self.assertEqual(exit_code, 0)
        cfg = run_embed_mock.call_args.args[0]
        self.assertEqual(cfg["ollama_base_url"], loaded_cfg["ollama_base_url"])

    def test_embed_ollama_still_fails_fast_on_invalid_ollama_url(self) -> None:
        loaded_cfg = {
            "ollama_base_url": "http://example.test:11434/api?token=secret",
            "phase3": {"embed": {"provider": "ollama", "model_id": "nomic-embed-text-v1.5"}},
        }
        config_path = Path("/tmp/configs/default.yaml")
        stderr = io.StringIO()

        with patch("agent.__main__.load_config_with_path", return_value=(loaded_cfg, config_path)):
            with patch("agent.__main__.configure_tool_security"):
                with patch("agent.__main__.run_embed", return_value=0) as run_embed_mock:
                    with patch.dict(
                        os.environ,
                        {"LOCAL_AGENT_OLLAMA_BASE_URL": "http://example.test:11434/api?token=secret"},
                        clear=False,
                    ):
                        with patch("sys.argv", ["agent", "embed", "--dry-run", "--json"]):
                            with patch("sys.stderr", stderr):
                                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertFalse(run_embed_mock.called)
        payload = json.loads(stderr.getvalue().strip())
        self.assertEqual(payload["error_code"], "CONFIG_ERROR")
        self.assertIn("must not include query", payload["error_message"])

    def test_chat_still_fails_fast_on_invalid_ollama_url(self) -> None:
        loaded_cfg = {"ollama_base_url": "http://example.test:11434/api?token=secret"}
        config_path = Path("/tmp/configs/default.yaml")
        stderr = io.StringIO()

        with patch("agent.__main__.load_config_with_path", return_value=(loaded_cfg, config_path)):
            with patch("agent.__main__.configure_tool_security"):
                with patch("agent.__main__.run_chat", return_value=0) as run_chat_mock:
                    with patch.dict(
                        os.environ,
                        {"LOCAL_AGENT_OLLAMA_BASE_URL": "http://example.test:11434/api?token=secret"},
                        clear=False,
                    ):
                        with patch("sys.argv", ["agent", "chat", "ping"]):
                            with patch("sys.stderr", stderr):
                                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertFalse(run_chat_mock.called)
        payload = json.loads(stderr.getvalue().strip())
        self.assertEqual(payload["error_code"], "CONFIG_ERROR")
        self.assertIn("must not include query", payload["error_message"])

    def test_ask_still_fails_fast_on_invalid_ollama_url(self) -> None:
        loaded_cfg = {"ollama_base_url": "http://example.test:11434/api?token=secret"}
        config_path = Path("/tmp/configs/default.yaml")
        stderr = io.StringIO()

        with patch("agent.__main__.load_config_with_path", return_value=(loaded_cfg, config_path)):
            with patch("agent.__main__.configure_tool_security"):
                with patch("agent.__main__.run_ask_grounded", return_value=0) as run_ask_mock:
                    with patch.dict(
                        os.environ,
                        {"LOCAL_AGENT_OLLAMA_BASE_URL": "http://example.test:11434/api?token=secret"},
                        clear=False,
                    ):
                        with patch("sys.argv", ["agent", "ask", "ping"]):
                            with patch("sys.stderr", stderr):
                                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertFalse(run_ask_mock.called)
        payload = json.loads(stderr.getvalue().strip())
        self.assertEqual(payload["error_code"], "CONFIG_ERROR")
        self.assertIn("must not include query", payload["error_message"])


if __name__ == "__main__":
    unittest.main()
