from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.embed_runtime_fingerprint import build_ollama_runtime_fingerprint
from agent.embedding_fingerprint import (
    compute_chunk_preprocess_sig,
    compute_embed_sig,
    compute_query_preprocess_sig,
    normalize_vector,
    pack_vector_f32_le,
)
from agent.embeddings_db import connect_db as connect_embeddings_db
from agent.embeddings_db import init_db as init_embeddings_db
from agent.embeddings_db import set_meta as set_embeddings_meta
from agent.embeddings_db import upsert_embedding
from agent.indexer import SourceSpec, index_sources
from agent.phase3 import build_phase3_cfg, run_embed_phase
from agent.retrieval import _compute_vector_candidates, retrieve
from agent.tools import configure_tool_security


class _DummyEmbedder:
    def __init__(self, dim: int = 8, runtime_fp: str = "dummy-runtime-v1") -> None:
        self.dim = dim
        self._runtime_fp = runtime_fp

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
            out.append([float(digest[i]) / 255.0 for i in range(self.dim)])
        return out

    @property
    def embed_dim(self) -> int:
        return self.dim

    def runtime_fingerprint(self) -> str:
        return self._runtime_fp


def _dummy_factory(provider: str, model_id: str, base_url: str, timeout_s: int) -> _DummyEmbedder:
    _ = timeout_s
    if provider == "ollama":
        runtime_fp = build_ollama_runtime_fingerprint(base_url=base_url, model_id=model_id)
    else:
        runtime_fp = f"{provider}:{model_id}"
    return _DummyEmbedder(runtime_fp=runtime_fp)


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
        for i in range(80):
            (self.corpus / f"extra_{i:02d}.md").write_text(
                f"## Extra {i}\ncoherence token extra {i}\n",
                encoding="utf-8",
            )

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
                    "chunk_preprocess_sig": "",
                    "query_preprocess_sig": "",
                    "batch_size": 16,
                },
                "retrieve": {
                    "lexical_k": 10,
                    "vector_k": 10,
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
        phase3_cfg = build_phase3_cfg(self.cfg)
        self.preprocess_name = str(phase3_cfg["embed"]["preprocess"])
        self.chunk_pre_sig = compute_chunk_preprocess_sig(self.preprocess_name)
        self.query_pre_sig = compute_query_preprocess_sig(self.preprocess_name)
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
            preprocess_name=self.preprocess_name,
            chunk_preprocess_sig=self.chunk_pre_sig,
            query_preprocess_sig=self.query_pre_sig,
            lexical_k=phase3_cfg["retrieve"]["lexical_k"],
            vector_k=phase3_cfg["retrieve"]["vector_k"],
            vector_fetch_k=phase3_cfg["retrieve"]["vector_fetch_k"],
            rel_path_prefix=phase3_cfg["retrieve"]["rel_path_prefix"],
            fusion=phase3_cfg["retrieve"]["fusion"],
        )
        self.assertGreaterEqual(len(result.candidates), 1)
        first = result.candidates[0]
        self.assertTrue(first.chunk_key)
        self.assertTrue(first.rel_path)
        self.assertIn(first.method, {"lexical", "vector", "both"})
        self.assertEqual(result.embed_model_id, phase3_cfg["embed"]["model_id"])
        self.assertTrue(result.chunk_preprocess_sig)
        self.assertTrue(result.query_preprocess_sig)

    def test_vector_fetch_filter_slice_with_rel_path_prefix(self) -> None:
        phase3_cfg = build_phase3_cfg(self.cfg)
        result = retrieve(
            "coherence token",
            index_db_path=self.phase2_db,
            embeddings_db_path=(self.workroot / phase3_cfg["embeddings_db_path"]).resolve(),
            embedder=_DummyEmbedder(),
            embed_model_id=phase3_cfg["embed"]["model_id"],
            preprocess_name=self.preprocess_name,
            chunk_preprocess_sig=self.chunk_pre_sig,
            query_preprocess_sig=self.query_pre_sig,
            lexical_k=2,
            vector_k=2,
            vector_fetch_k=0,
            rel_path_prefix="a.md",
            fusion="simple_union",
        )
        self.assertTrue(result.rel_path_prefix_applied)
        self.assertEqual(result.vector_fetch_k_used, 50)
        self.assertGreaterEqual(result.vector_candidates_prefilter, result.vector_candidates_postfilter)
        self.assertEqual(result.vector_candidates_postfilter, 1)
        self.assertIn("rel_path_prefix reduced vector results", result.vector_filter_warning)

    def test_vector_stage_only_fetches_candidate_metadata(self) -> None:
        phase3_cfg = build_phase3_cfg(self.cfg)
        embeddings_db_path = (self.workroot / phase3_cfg["embeddings_db_path"]).resolve()
        calls: list[list[str]] = []
        from agent import retrieval as retrieval_mod

        original_fetch = retrieval_mod._fetch_chunk_metadata

        def _spy(*, index_db_path: Path, chunk_keys: list[str]) -> dict[str, dict[str, str]]:
            calls.append(list(chunk_keys))
            return original_fetch(index_db_path=index_db_path, chunk_keys=chunk_keys)

        with patch("agent.retrieval._fetch_chunk_metadata", side_effect=_spy):
            result = retrieve(
                "coherence token",
                index_db_path=self.phase2_db,
                embeddings_db_path=embeddings_db_path,
                embedder=_DummyEmbedder(),
                embed_model_id=phase3_cfg["embed"]["model_id"],
                preprocess_name=self.preprocess_name,
                chunk_preprocess_sig=self.chunk_pre_sig,
                query_preprocess_sig=self.query_pre_sig,
                lexical_k=2,
                vector_k=2,
                vector_fetch_k=0,
                rel_path_prefix="extra_",
                fusion="simple_union",
            )
        self.assertGreaterEqual(len(calls), 1)
        self.assertGreaterEqual(len(result.candidates), 1)
        total_docs = len(list(self.corpus.glob("*.md")))
        for key_list in calls:
            self.assertLess(len(set(key_list)), total_docs)

    def test_vector_scoring_normalized_dot_order(self) -> None:
        embed_db = self.workroot / "embeddings" / "db" / "score.sqlite"
        init_embeddings_db(embed_db)
        with connect_embeddings_db(embed_db) as conn:
            set_embeddings_meta(conn, "vectors_normalized", "1")
            set_embeddings_meta(conn, "embed_model_id", "m")
            set_embeddings_meta(conn, "chunk_preprocess_sig", self.chunk_pre_sig)
            set_embeddings_meta(conn, "query_preprocess_sig", self.query_pre_sig)
            v1 = normalize_vector([1.0, 0.0, 0.0, 0.0])
            v2 = normalize_vector([0.5, math.sqrt(3.0) / 2.0, 0.0, 0.0])
            upsert_embedding(
                conn,
                chunk_key="k1",
                embed_sig=compute_embed_sig(
                    chunk_key="k1",
                    chunk_sha="s1",
                    model_id="m",
                    dim=4,
                    chunk_preprocess_sig=self.chunk_pre_sig,
                ),
                model_id="m",
                dim=4,
                preprocess_sig=self.chunk_pre_sig,
                vector_blob=pack_vector_f32_le(v1),
            )
            upsert_embedding(
                conn,
                chunk_key="k2",
                embed_sig=compute_embed_sig(
                    chunk_key="k2",
                    chunk_sha="s2",
                    model_id="m",
                    dim=4,
                    chunk_preprocess_sig=self.chunk_pre_sig,
                ),
                model_id="m",
                dim=4,
                preprocess_sig=self.chunk_pre_sig,
                vector_blob=pack_vector_f32_le(v2),
            )
            conn.commit()

        ranked, scored_count, vectors_normalized = _compute_vector_candidates(
            embeddings_db_path=embed_db,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            query_dim=4,
            model_id="m",
            chunk_preprocess_sig=self.chunk_pre_sig,
            fetch_k=2,
        )
        self.assertEqual(scored_count, 2)
        self.assertTrue(vectors_normalized)
        self.assertEqual([k for _, k in ranked], ["k1", "k2"])

    def test_vector_scoring_fallback_without_numpy(self) -> None:
        embed_db = self.workroot / "embeddings" / "db" / "fallback.sqlite"
        init_embeddings_db(embed_db)
        with connect_embeddings_db(embed_db) as conn:
            set_embeddings_meta(conn, "vectors_normalized", "1")
            upsert_embedding(
                conn,
                chunk_key="k1",
                embed_sig="s1",
                model_id="m",
                dim=2,
                preprocess_sig=self.chunk_pre_sig,
                vector_blob=pack_vector_f32_le([1.0, 0.0]),
            )
            upsert_embedding(
                conn,
                chunk_key="k2",
                embed_sig="s2",
                model_id="m",
                dim=2,
                preprocess_sig=self.chunk_pre_sig,
                vector_blob=pack_vector_f32_le([0.0, 1.0]),
            )
            conn.commit()

        with patch("agent.retrieval._np", None):
            ranked, scored_count, _ = _compute_vector_candidates(
                embeddings_db_path=embed_db,
                query_vector=[1.0, 0.0],
                query_dim=2,
                model_id="m",
                chunk_preprocess_sig=self.chunk_pre_sig,
                fetch_k=2,
            )
        self.assertEqual(scored_count, 2)
        self.assertEqual([k for _, k in ranked], ["k1", "k2"])

    def test_vector_scoring_numpy_path_when_available(self) -> None:
        from agent import retrieval as retrieval_mod

        if retrieval_mod._np is None:
            self.skipTest("numpy not installed")

        embed_db = self.workroot / "embeddings" / "db" / "numpy.sqlite"
        init_embeddings_db(embed_db)
        with connect_embeddings_db(embed_db) as conn:
            set_embeddings_meta(conn, "vectors_normalized", "1")
            upsert_embedding(
                conn,
                chunk_key="k1",
                embed_sig="s1",
                model_id="m",
                dim=2,
                preprocess_sig=self.chunk_pre_sig,
                vector_blob=pack_vector_f32_le([1.0, 0.0]),
            )
            conn.commit()

        with patch("agent.retrieval.unpack_vector_f32_le", side_effect=AssertionError("fallback should not be used")):
            ranked, scored_count, _ = _compute_vector_candidates(
                embeddings_db_path=embed_db,
                query_vector=[1.0, 0.0],
                query_dim=2,
                model_id="m",
                chunk_preprocess_sig=self.chunk_pre_sig,
                fetch_k=1,
            )
        self.assertEqual(scored_count, 1)
        self.assertEqual([k for _, k in ranked], ["k1"])


if __name__ == "__main__":
    unittest.main()
