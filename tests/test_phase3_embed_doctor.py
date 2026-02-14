from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from agent.__main__ import collect_doctor_checks
from agent.index_db import connect_db
from agent.indexer import SourceSpec, index_sources
from agent.memory_db import connect_db as connect_memory_db
from agent.memory_db import init_db as init_memory_db
from agent.phase3 import build_phase3_cfg, run_embed_phase
from agent.tools import configure_tool_security


class _DummyEmbedder:
    def __init__(self, dim: int = 6) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
            vec = [float(digest[i]) / 255.0 for i in range(self.dim)]
            out.append(vec)
        return out


def _dummy_factory(provider: str, model_id: str, base_url: str, timeout_s: int) -> _DummyEmbedder:
    _ = provider, model_id, base_url, timeout_s
    return _DummyEmbedder()


class Phase3EmbedDoctorTests(unittest.TestCase):
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

        (self.corpus / "a.md").write_text("## A\nalpha chunk text\n", encoding="utf-8")
        (self.corpus / "b.md").write_text("## B\nbeta chunk text\n", encoding="utf-8")

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
                    "preprocess_sig": "",
                    "batch_size": 16,
                },
                "retrieve": {
                    "lexical_k": 5,
                    "vector_k": 5,
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

    def _doctor(self, cfg: dict, require_phase3: bool) -> tuple[list[str], dict]:
        summary: dict = {}
        checks = collect_doctor_checks(
            cfg,
            resolved_config_path=self.config_path,
            roots=self.roots,
            check_ollama=False,
            require_phase3=require_phase3,
            phase3_summary_out=summary,
        )
        failed_codes = [check.error_code for check in checks if not check.ok]
        return failed_codes, summary

    def test_doctor_flags_outdated_then_embed_fixes(self) -> None:
        phase3_cfg = build_phase3_cfg(self.cfg)
        first = run_embed_phase(
            cfg=self.cfg,
            security_root=self.workroot,
            phase2_db_path=self.db_path,
            phase3_cfg=phase3_cfg,
            embedder_factory=_dummy_factory,
        )
        self.assertGreater(first.embedded_written, 0)

        failed_codes, summary = self._doctor(copy.deepcopy(self.cfg), require_phase3=True)
        self.assertEqual(failed_codes, [])
        self.assertEqual(int(summary["missing_embeddings"]), 0)
        self.assertEqual(int(summary["outdated_embeddings"]), 0)

        with connect_db(self.db_path) as conn:
            row = conn.execute("SELECT id FROM chunks ORDER BY id LIMIT 1").fetchone()
            self.assertIsNotNone(row)
            chunk_id = int(row["id"])
            new_text = "mutated chunk text"
            new_sha = hashlib.sha256(new_text.encode("utf-8", errors="replace")).hexdigest()
            conn.execute(
                "UPDATE chunks SET text = ?, sha256 = ? WHERE id = ?",
                (new_text, new_sha, chunk_id),
            )
            conn.commit()

        failed_codes, summary = self._doctor(copy.deepcopy(self.cfg), require_phase3=True)
        self.assertIn("DOCTOR_EMBED_OUTDATED_REQUIRE_PHASE3", failed_codes)
        self.assertGreater(int(summary["outdated_embeddings"]), 0)

        second = run_embed_phase(
            cfg=self.cfg,
            security_root=self.workroot,
            phase2_db_path=self.db_path,
            phase3_cfg=phase3_cfg,
            embedder_factory=_dummy_factory,
        )
        self.assertGreater(second.embedded_written, 0)

        failed_codes, summary = self._doctor(copy.deepcopy(self.cfg), require_phase3=True)
        self.assertEqual(failed_codes, [])
        self.assertEqual(int(summary["outdated_embeddings"]), 0)

    def test_doctor_require_phase3_flags_missing_embeddings(self) -> None:
        failed_codes, summary = self._doctor(copy.deepcopy(self.cfg), require_phase3=True)
        self.assertIn("DOCTOR_PHASE3_EMBEDDINGS_DB_MISSING", failed_codes)
        self.assertEqual(int(summary["missing_embeddings"]), 0)

    def test_doctor_detects_preprocess_sig_mismatch(self) -> None:
        bad_cfg = copy.deepcopy(self.cfg)
        bad_cfg["phase3"]["embed"]["preprocess_sig"] = "deadbeef"
        failed_codes, _ = self._doctor(bad_cfg, require_phase3=False)
        self.assertIn("DOCTOR_PHASE3_CONFIG_INVALID", failed_codes)

    def test_model_change_marks_outdated_until_reembed(self) -> None:
        phase3_cfg = build_phase3_cfg(self.cfg)
        run_embed_phase(
            cfg=self.cfg,
            security_root=self.workroot,
            phase2_db_path=self.db_path,
            phase3_cfg=phase3_cfg,
            embedder_factory=_dummy_factory,
        )

        changed_cfg = copy.deepcopy(self.cfg)
        changed_cfg["phase3"]["embed"]["model_id"] = "alternate-model"
        failed_codes, summary = self._doctor(changed_cfg, require_phase3=True)
        self.assertIn("DOCTOR_EMBED_OUTDATED_REQUIRE_PHASE3", failed_codes)
        self.assertGreater(int(summary["outdated_embeddings"]), 0)

        changed_phase3_cfg = build_phase3_cfg(changed_cfg)
        run_embed_phase(
            cfg=changed_cfg,
            security_root=self.workroot,
            phase2_db_path=self.db_path,
            phase3_cfg=changed_phase3_cfg,
            embedder_factory=_dummy_factory,
        )

        failed_codes, summary = self._doctor(changed_cfg, require_phase3=True)
        self.assertEqual(failed_codes, [])
        self.assertEqual(int(summary["outdated_embeddings"]), 0)

    def test_doctor_detects_dangling_memory_evidence(self) -> None:
        memory_db = self.workroot / "memory" / "durable.sqlite"
        init_memory_db(memory_db)
        with connect_memory_db(memory_db) as conn:
            conn.execute(
                """
                INSERT INTO memory(memory_id, type, content, source, created_at, updated_at)
                VALUES ('m1', 'preference', 'x', 'manual', 1.0, 1.0)
                """
            )
            conn.execute(
                """
                INSERT INTO memory_evidence(memory_id, chunk_key)
                VALUES ('m1', 'deadbeefdeadbeefdeadbeefdeadbeef')
                """
            )
            conn.commit()

        failed_codes, summary = self._doctor(copy.deepcopy(self.cfg), require_phase3=False)
        self.assertIn("DOCTOR_MEMORY_DANGLING_EVIDENCE", failed_codes)
        self.assertGreater(int(summary["dangling_memory_evidence"]), 0)


if __name__ == "__main__":
    unittest.main()
