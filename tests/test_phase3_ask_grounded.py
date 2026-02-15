from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.__main__ import _build_grounded_system_prompt, run_ask_grounded
from agent.index_db import connect_db as connect_index_db
from agent.index_db import init_db as init_index_db
from agent.retrieval import RetrievedChunk, RetrievalResult


class Phase3AskGroundedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.workroot = self.tmp_path / "workroot"
        (self.workroot / "runs").mkdir(parents=True, exist_ok=True)
        (self.workroot / "index").mkdir(parents=True, exist_ok=True)
        self.phase2_db_path = self.workroot / "index" / "index.sqlite"
        init_index_db(self.phase2_db_path)

        self.cfg = {
            "ollama_base_url": "http://127.0.0.1:11434",
            "timeout_s": 1,
            "timeout_s_big_second": 1,
            "max_tokens_big_second": 128,
            "temperature": 0.0,
            "model": "test-model",
            "model_fast": "test-model",
            "model_big": "test-model",
            "prefer_fast": True,
            "big_triggers": [],
            "phase2": {
                "index_db_path": "index/index.sqlite",
                "sources": [{"name": "corpus", "root": "allowed/corpus/", "kind": "corpus"}],
                "chunking": {"scheme": "obsidian_v1", "max_chars": 120, "overlap": 20},
            },
            "phase3": {
                "embeddings_db_path": "embeddings/db/embeddings.sqlite",
                "embed": {
                    "provider": "ollama",
                    "model_id": "nomic-embed-text-v1.5",
                    "preprocess": "obsidian_v1",
                    "chunk_preprocess_sig": "",
                    "query_preprocess_sig": "",
                    "batch_size": 8,
                },
                "retrieve": {
                    "lexical_k": 5,
                    "vector_k": 5,
                    "vector_fetch_k": 0,
                    "rel_path_prefix": "",
                    "fusion": "simple_union",
                },
                "ask": {"citation_validation": {"enabled": True, "strict": False, "require_in_snapshot": False}},
                "runs": {
                    "log_evidence_excerpts": True,
                    "max_total_evidence_chars": 200000,
                    "max_excerpt_chars": 1200,
                },
                "memory": {"durable_db_path": "memory/durable.sqlite", "enabled": True},
            },
        }
        self.roots = {"security_root": self.workroot}
        self._insert_chunk(
            chunk_key="0123456789abcdef0123456789abcdef",
            rel_path="a.md",
            heading_path="H2: Alpha",
            text="alpha",
        )

    def _key_for_index(self, i: int) -> str:
        return f"{i:032x}"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _insert_chunk(self, *, chunk_key: str, rel_path: str, heading_path: str, text: str) -> None:
        with connect_index_db(self.phase2_db_path) as conn:
            now = 1000.0
            conn.execute(
                "INSERT OR IGNORE INTO sources(name, root, kind, created_at) VALUES (?, ?, ?, ?)",
                ("corpus", "allowed/corpus/", "corpus", now),
            )
            source_row = conn.execute("SELECT id FROM sources WHERE name = ?", ("corpus",)).fetchone()
            if source_row is None:
                raise RuntimeError("expected source row")
            source_id = int(source_row["id"])

            doc_sha = hashlib.sha256(f"doc::{rel_path}".encode("utf-8")).hexdigest()
            existing_doc = conn.execute(
                "SELECT id FROM docs WHERE source_id = ? AND rel_path = ?",
                (source_id, rel_path),
            ).fetchone()
            if existing_doc is None:
                conn.execute(
                    """
                    INSERT INTO docs(
                        source_id, rel_path, abs_path, sha256, mtime, size,
                        is_markdown, yaml_present, yaml_parse_ok, yaml_error,
                        required_keys_present, frontmatter_json, discovered_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        rel_path,
                        str((self.workroot / "allowed" / "corpus" / rel_path).resolve()),
                        doc_sha,
                        now,
                        len(text),
                        1,
                        0,
                        0,
                        None,
                        0,
                        "{}",
                        now,
                        now,
                    ),
                )
                doc_row = conn.execute(
                    "SELECT id FROM docs WHERE source_id = ? AND rel_path = ?",
                    (source_id, rel_path),
                ).fetchone()
                if doc_row is None:
                    raise RuntimeError("expected doc row")
                doc_id = int(doc_row["id"])
            else:
                doc_id = int(existing_doc["id"])

            chunk_sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            existing_chunk = conn.execute(
                "SELECT id, chunk_index FROM chunks WHERE chunk_key = ?",
                (chunk_key,),
            ).fetchone()
            if existing_chunk is None:
                index_row = conn.execute(
                    "SELECT COALESCE(MAX(chunk_index), -1) AS max_idx FROM chunks WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
                chunk_index = int(index_row["max_idx"]) + 1 if index_row is not None else 0
                conn.execute(
                    """
                    INSERT INTO chunks(
                        doc_id, chunk_index, start_char, end_char, text, sha256,
                        created_at, scheme, heading_path, chunk_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        chunk_index,
                        0,
                        len(text),
                        text,
                        chunk_sha,
                        now,
                        "obsidian_v1",
                        heading_path,
                        chunk_key,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE chunks
                    SET doc_id = ?, chunk_index = ?, start_char = ?, end_char = ?, text = ?, sha256 = ?,
                        created_at = ?, scheme = ?, heading_path = ?, chunk_key = ?
                    WHERE id = ?
                    """,
                    (
                        doc_id,
                        int(existing_chunk["chunk_index"]),
                        0,
                        len(text),
                        text,
                        chunk_sha,
                        now,
                        "obsidian_v1",
                        heading_path,
                        chunk_key,
                        int(existing_chunk["id"]),
                    ),
                )
            conn.commit()

    def _latest_run_record(self) -> dict:
        run_dirs = sorted((self.workroot / "runs").glob("*"))
        if not run_dirs:
            raise AssertionError("expected run dir")
        return json.loads((run_dirs[-1] / "run.json").read_text(encoding="utf-8"))

    def _footer(self, *, missing: int, path_mismatches: int, sha_mismatches: int, not_in_snapshot: int) -> str:
        return (
            f"(missing={missing}, path_mismatches={path_mismatches}, "
            f"sha_mismatches={sha_mismatches}, not_in_snapshot={not_in_snapshot})"
        )

    def test_second_pass_prompt_contains_citation_invariants(self) -> None:
        prompt = _build_grounded_system_prompt()
        self.assertIn("CITATION INVARIANTS", prompt)
        self.assertIn("[source: <rel_path>#<heading_path> | <chunk_key>]", prompt)
        self.assertIn("INSUFFICIENT_EVIDENCE", prompt)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_missing_citations_triggers_fallback(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["enabled"] = False
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key="0123456789abcdef0123456789abcdef",
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": "Answer without citations"}}

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("Insufficient citation-grounded answer", output)
        self.assertIn("[source: a.md#H2: Alpha | 0123456789abcdef0123456789abcdef]", output)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    def test_no_candidates_returns_insufficient_evidence(
        self,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=0,
            vector_candidates_prefilter=0,
            vector_candidates_postfilter=0,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[],
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("Insufficient public evidence", output)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_non_strict_invalid_unparseable_returns_raw_answer_no_fallback(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["enabled"] = True
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = False
        key = "0123456789abcdef0123456789abcdef"
        raw_answer = "Answer with malformed citation [source: a.md | H2: Alpha]"
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key=key,
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": raw_answer}}

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        output = out.getvalue()
        self.assertIn(raw_answer, output)
        self.assertNotIn("Insufficient citation-grounded answer from model output.", output)
        footer = self._footer(missing=1, path_mismatches=0, sha_mismatches=0, not_in_snapshot=0)
        self.assertEqual(output.count(footer), 1)

        record = self._latest_run_record()
        self.assertEqual(record.get("citation_validation_footer"), footer)
        self.assertEqual(record["grounding_gate"]["reason"], "non_strict_invalid_returned")
        raw_report = record.get("citation_validation_raw_answer")
        final_report = record.get("citation_validation_final_answer")
        self.assertIsInstance(raw_report, dict)
        self.assertIsInstance(final_report, dict)
        self.assertEqual(int(raw_report.get("unparseable_citations_count", 0)), 1)
        self.assertEqual(int(final_report.get("unparseable_citations_count", 0)), 1)
        self.assertFalse(bool(raw_report.get("valid")))
        self.assertEqual(raw_report, final_report)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_run_logging_includes_evidence_excerpts_and_caps(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["runs"]["max_total_evidence_chars"] = 30
        self.cfg["phase3"]["runs"]["max_excerpt_chars"] = 12
        self._insert_chunk(
            chunk_key="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            rel_path="a.md",
            heading_path="H2: Alpha",
            text="A" * 20,
        )
        self._insert_chunk(
            chunk_key="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            rel_path="b.md",
            heading_path="H2: Beta",
            text="B" * 20,
        )
        self._insert_chunk(
            chunk_key="cccccccccccccccccccccccccccccccc",
            rel_path="c.md",
            heading_path="H2: Gamma",
            text="C" * 20,
        )
        self._insert_chunk(
            chunk_key="dddddddddddddddddddddddddddddddd",
            rel_path="d.md",
            heading_path="H2: Delta",
            text="D" * 20,
        )

        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=4,
            vector_candidates_prefilter=4,
            vector_candidates_postfilter=4,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="unused",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                ),
                RetrievedChunk(
                    chunk_key="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    rel_path="b.md",
                    heading_path="H2: Beta",
                    text="unused",
                    score=0.9,
                    method="both",
                    lexical_score=0.9,
                    vector_score=0.9,
                ),
                RetrievedChunk(
                    chunk_key="cccccccccccccccccccccccccccccccc",
                    rel_path="c.md",
                    heading_path="H2: Gamma",
                    text="unused",
                    score=0.8,
                    method="vector",
                    lexical_score=0.0,
                    vector_score=0.8,
                ),
                RetrievedChunk(
                    chunk_key="dddddddddddddddddddddddddddddddd",
                    rel_path="d.md",
                    heading_path="H2: Delta",
                    text="unused",
                    score=0.7,
                    method="lexical",
                    lexical_score=0.7,
                    vector_score=0.0,
                ),
            ],
        )
        mock_chat.return_value = {
            "message": {
                "content": "Answer [source: a.md#H2: Alpha | aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]"
            }
        }

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)

        record = self._latest_run_record()
        retrieval = record["retrieval"]
        self.assertEqual(retrieval["logging_caps"]["max_total_chars"], 30)
        self.assertEqual(retrieval["logging_caps"]["max_excerpt_chars"], 12)
        self.assertTrue(bool(retrieval["logging_truncated_total"]))
        self.assertEqual(int(retrieval["results_logged_count"]), 3)
        self.assertEqual(int(retrieval["results_omitted_count"]), 1)
        results = retrieval["results"]
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["chunk_sha"], hashlib.sha256(("A" * 20).encode("utf-8")).hexdigest())
        self.assertEqual(len(results[0]["excerpt"]), 12)
        self.assertEqual(len(results[1]["excerpt"]), 12)
        self.assertEqual(len(results[2]["excerpt"]), 6)
        self.assertTrue(bool(results[2]["excerpt_truncated"]))
        self.assertEqual(
            results[2]["excerpt_sha"],
            hashlib.sha256(results[2]["excerpt"].encode("utf-8")).hexdigest(),
        )

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_ask_non_strict_prints_footer_when_validation_valid(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = False
        key = "0123456789abcdef0123456789abcdef"
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key=key,
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": f"Answer [source: a.md#H2: Alpha | {key}]"}}

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        output = out.getvalue()
        footer = self._footer(missing=0, path_mismatches=0, sha_mismatches=0, not_in_snapshot=0)
        self.assertEqual(output.count(footer), 1)
        self.assertLess(output.index("Answer [source:"), output.index(footer))
        self.assertLess(output.index(footer), output.index("[logged]"))

        record = self._latest_run_record()
        self.assertEqual(record.get("citation_validation_footer"), footer)
        self.assertTrue(bool(record["citation_validation"]["valid"]))

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_ask_non_strict_records_warnings_but_returns_answer(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = False
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key="0123456789abcdef0123456789abcdef",
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {
            "message": {
                "content": "Answer [source: wrong.md#H2: Alpha | 0123456789abcdef0123456789abcdef]"
            }
        }

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        output = out.getvalue()
        self.assertIn("Answer [source: wrong.md#H2: Alpha | 0123456789abcdef0123456789abcdef]", output)
        footer = self._footer(missing=0, path_mismatches=1, sha_mismatches=0, not_in_snapshot=0)
        self.assertEqual(output.count(footer), 1)
        self.assertLess(output.index("Answer [source:"), output.index(footer))
        self.assertLess(output.index(footer), output.index("[logged]"))

        record = self._latest_run_record()
        self.assertTrue(bool(record.get("ok")))
        validation = record["citation_validation"]
        self.assertFalse(bool(validation["valid"]))
        self.assertEqual(len(validation["path_mismatches"]), 1)
        self.assertEqual(record.get("citation_validation_footer"), footer)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_ask_strict_mode_fails_closed_on_invalid_citations(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = True
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key="0123456789abcdef0123456789abcdef",
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {
            "message": {
                "content": "Answer [source: wrong.md#H2: Alpha | 0123456789abcdef0123456789abcdef]"
            }
        }

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 1)
        err_text = err.getvalue().strip()
        err_lines = [line for line in err_text.splitlines() if line.strip()]
        self.assertEqual(len(err_lines), 1)
        err_payload = json.loads(err_lines[0])
        self.assertEqual(err_payload.get("error_code"), "ASK_CITATION_INVALID")
        expected_footer = self._footer(missing=0, path_mismatches=1, sha_mismatches=0, not_in_snapshot=0)
        self.assertIn(expected_footer, str(err_payload.get("error_message", "")))
        self.assertNotIn(expected_footer, out.getvalue())

        record = self._latest_run_record()
        self.assertFalse(bool(record.get("ok")))
        self.assertEqual(record.get("error_code"), "ASK_CITATION_INVALID")
        validation = record["citation_validation"]
        self.assertFalse(bool(validation["valid"]))
        self.assertEqual(len(validation["path_mismatches"]), 1)
        self.assertEqual(record.get("citation_validation_footer"), expected_footer)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_strict_invalid_unparseable_fails_closed_and_logs_raw_vs_final(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["enabled"] = True
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = True
        key = "0123456789abcdef0123456789abcdef"
        raw_answer = "Answer with malformed citation [source: a.md | H2: Alpha]"
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key=key,
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": raw_answer}}

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 1)
        err_payload = json.loads(err.getvalue().strip())
        self.assertEqual(err_payload["error_code"], "ASK_CITATION_INVALID")
        footer = self._footer(missing=1, path_mismatches=0, sha_mismatches=0, not_in_snapshot=0)
        self.assertIn(footer, err_payload["error_message"])

        record = self._latest_run_record()
        self.assertEqual(record["grounding_gate"]["reason"], "strict_fail")
        self.assertIsInstance(record.get("citation_validation_raw_answer"), dict)
        self.assertIsNone(record.get("citation_validation_final_answer"))
        self.assertEqual(record.get("citation_validation_footer"), footer)
        self.assertEqual(record.get("error_code"), "ASK_CITATION_INVALID")

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_ask_strict_mode_fails_when_citation_not_in_snapshot(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = True
        self.cfg["phase3"]["ask"]["citation_validation"]["require_in_snapshot"] = True
        b_key = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        self._insert_chunk(
            chunk_key=b_key,
            rel_path="b.md",
            heading_path="H2: Beta",
            text="beta",
        )
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key="0123456789abcdef0123456789abcdef",
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": f"Answer [source: b.md#H2: Beta | {b_key}]"}}

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 1)
        self.assertIn("ASK_CITATION_INVALID", err.getvalue())

        record = self._latest_run_record()
        self.assertFalse(bool(record.get("ok")))
        validation = record["citation_validation"]
        self.assertTrue(bool(validation["require_in_snapshot"]))
        self.assertEqual(validation["not_in_snapshot_chunk_keys"], [b_key])
        self.assertFalse(bool(validation["valid"]))

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_ask_strict_mode_allows_prefix_heading_match(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        key = "cccccccccccccccccccccccccccccccc"
        rel_path = "gate.md"
        self._insert_chunk(
            chunk_key=key,
            rel_path=rel_path,
            heading_path="H2: Qualities > H3: From ChatGPT",
            text="gate",
        )
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = True
        self.cfg["phase3"]["ask"]["citation_validation"]["require_in_snapshot"] = True
        self.cfg["phase3"]["ask"]["citation_validation"]["heading_match"] = "prefix"
        self.cfg["phase3"]["ask"]["citation_validation"]["normalize_heading"] = True
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key=key,
                    rel_path=rel_path,
                    heading_path="H2: Qualities > H3: From ChatGPT",
                    text="gate",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": f"Answer [source: {rel_path}#H2: Qualities | {key}]"}}

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        self.assertNotIn("ASK_CITATION_INVALID", err.getvalue())

        record = self._latest_run_record()
        self.assertTrue(bool(record.get("ok")))
        validation = record["citation_validation"]
        self.assertTrue(bool(validation["valid"]))
        self.assertEqual(validation["path_mismatches"], [])

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_ask_non_strict_returns_answer_when_not_in_snapshot(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = False
        self.cfg["phase3"]["ask"]["citation_validation"]["require_in_snapshot"] = True
        b_key = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        self._insert_chunk(
            chunk_key=b_key,
            rel_path="b.md",
            heading_path="H2: Beta",
            text="beta",
        )
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=5,
            vector_candidates_scored=1,
            vector_candidates_prefilter=1,
            vector_candidates_postfilter=1,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=[
                RetrievedChunk(
                    chunk_key="0123456789abcdef0123456789abcdef",
                    rel_path="a.md",
                    heading_path="H2: Alpha",
                    text="alpha",
                    score=1.0,
                    method="both",
                    lexical_score=1.0,
                    vector_score=1.0,
                )
            ],
        )
        mock_chat.return_value = {"message": {"content": f"Answer [source: b.md#H2: Beta | {b_key}]"}}

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        self.assertIn(f"[source: b.md#H2: Beta | {b_key}]", out.getvalue())

        record = self._latest_run_record()
        self.assertTrue(bool(record.get("ok")))
        validation = record["citation_validation"]
        self.assertTrue(bool(validation["require_in_snapshot"]))
        self.assertEqual(validation["not_in_snapshot_chunk_keys"], [b_key])
        self.assertFalse(bool(validation["valid"]))

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_top_n_honored_and_snapshot_enforced(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["evidence"] = {"top_n": 5}
        self.cfg["phase3"]["ask"]["citation_validation"]["strict"] = False
        self.cfg["phase3"]["ask"]["citation_validation"]["require_in_snapshot"] = True

        candidates: list[RetrievedChunk] = []
        for i in range(1, 21):
            key = self._key_for_index(i)
            rel = f"n{i}.md"
            heading = f"H2: {i}"
            self._insert_chunk(
                chunk_key=key,
                rel_path=rel,
                heading_path=heading,
                text=f"text {i}",
            )
            candidates.append(
                RetrievedChunk(
                    chunk_key=key,
                    rel_path=rel,
                    heading_path=heading,
                    text=f"text {i}",
                    score=1.0 - (i * 0.01),
                    method="both",
                    lexical_score=1.0 - (i * 0.01),
                    vector_score=1.0 - (i * 0.01),
                )
            )
        cited_key = self._key_for_index(12)
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=20,
            vector_candidates_scored=20,
            vector_candidates_prefilter=20,
            vector_candidates_postfilter=20,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=candidates,
        )
        mock_chat.return_value = {"message": {"content": f"Answer [source: n12.md#H2: 12 | {cited_key}]"}}

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)

        record = self._latest_run_record()
        retrieval = record["retrieval"]
        budget = retrieval["evidence_budget"]
        self.assertEqual(int(budget["requested_top_n"]), 5)
        self.assertEqual(int(budget["effective_top_n"]), 5)
        self.assertEqual(int(budget["total_retrieval_results_available"]), 20)
        self.assertEqual(int(retrieval["evidence_selected_count"]), 5)
        self.assertEqual(int(retrieval["evidence_omitted_count"]), 15)
        validation = record["citation_validation"]
        self.assertEqual(validation["not_in_snapshot_chunk_keys"], [cited_key])
        self.assertFalse(bool(validation["valid"]))

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_top_n_clamps_when_larger_than_available(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["evidence"] = {"top_n": 10}
        self.cfg["phase3"]["ask"]["citation_validation"]["enabled"] = False

        candidates: list[RetrievedChunk] = []
        for i in range(1, 4):
            key = self._key_for_index(i)
            candidates.append(
                RetrievedChunk(
                    chunk_key=key,
                    rel_path=f"a{i}.md",
                    heading_path=f"H2: {i}",
                    text=f"text {i}",
                    score=1.0 - (i * 0.1),
                    method="both",
                    lexical_score=1.0 - (i * 0.1),
                    vector_score=1.0 - (i * 0.1),
                )
            )
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=3,
            vector_candidates_scored=3,
            vector_candidates_prefilter=3,
            vector_candidates_postfilter=3,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=candidates,
        )
        mock_chat.return_value = {"message": {"content": f"Answer [source: a1.md#H2: 1 | {self._key_for_index(1)}]"}}

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        record = self._latest_run_record()
        budget = record["retrieval"]["evidence_budget"]
        self.assertEqual(int(budget["requested_top_n"]), 10)
        self.assertEqual(int(budget["effective_top_n"]), 3)
        self.assertEqual(int(record["retrieval"]["evidence_omitted_count"]), 0)

    @patch("agent.__main__.ensure_ollama_up")
    @patch("agent.__main__.create_embedder")
    @patch("agent.__main__.retrieve")
    @patch("agent.__main__.ollama_chat")
    def test_top_n_non_positive_clamps_to_one(
        self,
        mock_chat,
        mock_retrieve,
        mock_create_embedder,
        mock_ensure_up,
    ) -> None:
        _ = mock_ensure_up, mock_create_embedder
        self.cfg["phase3"]["ask"]["evidence"] = {"top_n": 0}
        self.cfg["phase3"]["ask"]["citation_validation"]["enabled"] = False

        candidates: list[RetrievedChunk] = []
        for i in range(1, 4):
            key = self._key_for_index(i)
            candidates.append(
                RetrievedChunk(
                    chunk_key=key,
                    rel_path=f"z{i}.md",
                    heading_path=f"H2: {i}",
                    text=f"text {i}",
                    score=1.0 - (i * 0.1),
                    method="both",
                    lexical_score=1.0 - (i * 0.1),
                    vector_score=1.0 - (i * 0.1),
                )
            )
        mock_retrieve.return_value = RetrievalResult(
            query="q",
            chunker_sig="sig",
            embed_model_id="m",
            chunk_preprocess_sig="p1",
            query_preprocess_sig="p2",
            embed_db_schema_version=1,
            vector_fetch_k_used=3,
            vector_candidates_scored=3,
            vector_candidates_prefilter=3,
            vector_candidates_postfilter=3,
            rel_path_prefix_applied=False,
            vector_filter_warning="",
            candidates=candidates,
        )
        mock_chat.return_value = {"message": {"content": f"Answer [source: z1.md#H2: 1 | {self._key_for_index(1)}]"}}

        out = io.StringIO()
        with redirect_stdout(out):
            code = run_ask_grounded(self.cfg, "question", roots=self.roots)
        self.assertEqual(code, 0)
        record = self._latest_run_record()
        budget = record["retrieval"]["evidence_budget"]
        self.assertEqual(int(budget["requested_top_n"]), 0)
        self.assertEqual(int(budget["effective_top_n"]), 1)
        self.assertEqual(int(record["retrieval"]["evidence_selected_count"]), 1)
        self.assertEqual(int(record["retrieval"]["evidence_omitted_count"]), 2)


if __name__ == "__main__":
    unittest.main()
