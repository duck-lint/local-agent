from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from agent.indexer import SourceSpec, index_sources
from agent.phase3 import build_phase3_cfg, run_embed_phase
from agent.retrieval import retrieve
from agent.tools import configure_tool_security


class _DummyEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
            out.append([float(digest[i]) / 255.0 for i in range(self.dim)])
        return out


def _dummy_factory(provider: str, model_id: str, base_url: str, timeout_s: int) -> _DummyEmbedder:
    _ = provider, model_id, base_url, timeout_s
    return _DummyEmbedder()


class Phase3RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.workroot = self.tmp_path / "workroot"
        self.corpus = self.workroot / "allowed" / "corpus"
        self.scratch = self.workroot / "allowed" / "scratch"
        self.corpus.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)

        configure_tool_security(
            {
                "allowed_roots": ["allowed/corpus/", "allowed/scratch/", "runs/"],
                "allowed_exts": [".md", ".txt", ".json"],
                "deny_absolute_paths": True,
                "deny_hidden_paths": True,
                "allow_any_path": False,
                "auto_create_allowed_roots": True,
                "roots_must_be_within_security_root": True,
            },
            workspace_root=self.workroot,
        )

        (self.corpus / "a.md").write_text("## Alpha\ncoherence token appears here\n", encoding="utf-8")
        (self.corpus / "b.md").write_text("## Beta\ncoherence token appears again\n", encoding="utf-8")

        self.phase2_db = self.workroot / "index" / "index.sqlite"
        summary = index_sources(
            db_path=self.phase2_db,
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
            "phase3": {
                "embeddings_db_path": "embeddings/db/embeddings.sqlite",
                "embed": {
                    "provider": "ollama",
                    "model_id": "nomic-embed-text-v1.5",
                    "preprocess": "obsidian_v1",
                    "preprocess_sig": "",
                    "batch_size": 16,
                },
                "retrieve": {
                    "lexical_k": 10,
                    "vector_k": 10,
                    "fusion": "simple_union",
                },
                "memory": {
                    "durable_db_path": "memory/durable.sqlite",
                    "enabled": True,
                },
            },
        }
        phase3_cfg = build_phase3_cfg(self.cfg)
        run_embed_phase(
            cfg=self.cfg,
            security_root=self.workroot,
            phase2_db_path=self.phase2_db,
            phase3_cfg=phase3_cfg,
            embedder_factory=_dummy_factory,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_retrieve_returns_provenance_and_fused_candidates(self) -> None:
        phase3_cfg = build_phase3_cfg(self.cfg)
        result = retrieve(
            "coherence token",
            index_db_path=self.phase2_db,
            embeddings_db_path=(self.workroot / phase3_cfg["embeddings_db_path"]).resolve(),
            embedder=_DummyEmbedder(),
            embed_model_id=phase3_cfg["embed"]["model_id"],
            preprocess_name=phase3_cfg["embed"]["preprocess"],
            lexical_k=phase3_cfg["retrieve"]["lexical_k"],
            vector_k=phase3_cfg["retrieve"]["vector_k"],
            fusion=phase3_cfg["retrieve"]["fusion"],
        )
        self.assertGreaterEqual(len(result.candidates), 1)
        first = result.candidates[0]
        self.assertTrue(first.chunk_key)
        self.assertTrue(first.rel_path)
        self.assertIn(first.method, {"lexical", "vector", "both"})
        self.assertEqual(result.embed_model_id, phase3_cfg["embed"]["model_id"])
        self.assertTrue(result.preprocess_sig)


if __name__ == "__main__":
    unittest.main()
