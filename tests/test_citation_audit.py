from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from agent.citation_audit import parse_citations, validate_citations
from agent.index_db import connect_db as connect_index_db
from agent.index_db import init_db as init_index_db


class CitationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.db_path = self.tmp_path / "index.sqlite"
        init_index_db(self.db_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _insert_chunk(self, *, chunk_key: str, rel_path: str, heading_path: str, text: str) -> str:
        with connect_index_db(self.db_path) as conn:
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
            conn.execute(
                """
                INSERT INTO docs(
                    source_id, rel_path, abs_path, sha256, mtime, size,
                    is_markdown, yaml_present, yaml_parse_ok, yaml_error,
                    required_keys_present, frontmatter_json, discovered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, rel_path) DO UPDATE SET
                    abs_path=excluded.abs_path,
                    sha256=excluded.sha256,
                    mtime=excluded.mtime,
                    size=excluded.size,
                    updated_at=excluded.updated_at
                """,
                (
                    source_id,
                    rel_path,
                    str((self.tmp_path / rel_path).resolve()),
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
            chunk_sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            existing_chunk = conn.execute(
                "SELECT id, chunk_index FROM chunks WHERE chunk_key = ?",
                (chunk_key,),
            ).fetchone()
            if existing_chunk is None:
                idx_row = conn.execute(
                    "SELECT COALESCE(MAX(chunk_index), -1) AS max_idx FROM chunks WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
                chunk_index = int(idx_row["max_idx"]) + 1 if idx_row is not None else 0
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
        return chunk_sha

    def test_citation_parser_extracts_chunk_key_and_paths(self) -> None:
        answer = (
            "A [source: notes/a.md#root | 0123456789abcdef0123456789abcdef] "
            "and [source: notes/b.md#H2: Beta | fedcba9876543210fedcba9876543210] "
            "and [source: notes/c.md | aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]"
        )
        parsed = parse_citations(answer)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0].chunk_key, "0123456789abcdef0123456789abcdef")
        self.assertEqual(parsed[0].rel_path, "notes/a.md")
        self.assertEqual(parsed[0].heading_path, "")
        self.assertEqual(parsed[1].heading_path, "H2: Beta")
        self.assertEqual(parsed[2].rel_path, "notes/c.md")
        self.assertEqual(parsed[2].heading_path, "")

    def test_citation_validator_flags_missing_and_sha_mismatch(self) -> None:
        chunk_key = "0123456789abcdef0123456789abcdef"
        self._insert_chunk(
            chunk_key=chunk_key,
            rel_path="a.md",
            heading_path="H2: Alpha",
            text="alpha text",
        )
        parsed = parse_citations(
            "ok [source: a.md#H2: Alpha | 0123456789abcdef0123456789abcdef] "
            "missing [source: b.md#H2: Beta | fedcba9876543210fedcba9876543210]"
        )
        report = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={chunk_key: "deadbeef"},
            enabled=True,
            strict=False,
            require_in_snapshot=False,
        )
        self.assertFalse(bool(report["valid"]))
        self.assertEqual(report["missing_chunk_keys"], ["fedcba9876543210fedcba9876543210"])
        self.assertEqual(len(report["mismatched_sha"]), 1)
        self.assertEqual(report["mismatched_sha"][0]["chunk_key"], chunk_key)

    def test_citation_validator_require_in_snapshot_toggle(self) -> None:
        key_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        key_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        sha_a = self._insert_chunk(
            chunk_key=key_a,
            rel_path="a.md",
            heading_path="H2: Alpha",
            text="alpha text",
        )
        _ = self._insert_chunk(
            chunk_key=key_b,
            rel_path="b.md",
            heading_path="H2: Beta",
            text="beta text",
        )
        parsed = parse_citations("cite [source: b.md#H2: Beta | bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb]")

        permissive = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={key_a: sha_a},
            enabled=True,
            strict=False,
            require_in_snapshot=False,
        )
        self.assertTrue(bool(permissive["valid"]))
        self.assertEqual(permissive["not_in_snapshot_chunk_keys"], [key_b])
        self.assertEqual(permissive["sha_unchecked_chunk_keys"], [key_b])

        enforced = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={key_a: sha_a},
            enabled=True,
            strict=False,
            require_in_snapshot=True,
        )
        self.assertFalse(bool(enforced["valid"]))
        self.assertEqual(enforced["not_in_snapshot_chunk_keys"], [key_b])

    def test_heading_normalization_allows_punctuation_variation(self) -> None:
        key = "cccccccccccccccccccccccccccccccc"
        sha = self._insert_chunk(
            chunk_key=key,
            rel_path="a.md",
            heading_path="H1: Freeform Journaling:",
            text="alpha text",
        )
        parsed = parse_citations(f"cite [source: a.md#H1: Freeform Journaling | {key}]")

        exact = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={key: sha},
            enabled=True,
            strict=False,
            require_in_snapshot=False,
            heading_match="exact",
            normalize_heading=True,
        )
        self.assertTrue(bool(exact["valid"]))
        self.assertEqual(exact["path_mismatches"], [])

        prefix = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={key: sha},
            enabled=True,
            strict=False,
            require_in_snapshot=False,
            heading_match="prefix",
            normalize_heading=True,
        )
        self.assertTrue(bool(prefix["valid"]))

    def test_heading_match_prefix_vs_exact(self) -> None:
        key = "dddddddddddddddddddddddddddddddd"
        sha = self._insert_chunk(
            chunk_key=key,
            rel_path="gate.md",
            heading_path="H2: Qualities > H3: From ChatGPT",
            text="gate text",
        )
        parsed = parse_citations(f"cite [source: gate.md#H2: Qualities | {key}]")

        prefix = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={key: sha},
            enabled=True,
            strict=False,
            require_in_snapshot=False,
            heading_match="prefix",
            normalize_heading=True,
        )
        self.assertTrue(bool(prefix["valid"]))

        exact = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={key: sha},
            enabled=True,
            strict=False,
            require_in_snapshot=False,
            heading_match="exact",
            normalize_heading=True,
        )
        self.assertFalse(bool(exact["valid"]))
        self.assertEqual(len(exact["path_mismatches"]), 1)
        self.assertEqual(exact["path_mismatches"][0]["heading_mismatch_reason"], "prefix")

    def test_heading_match_ignore_logs_but_does_not_invalidate(self) -> None:
        key = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        sha = self._insert_chunk(
            chunk_key=key,
            rel_path="gate.md",
            heading_path="H2: Qualities > H3: From ChatGPT",
            text="gate text",
        )
        parsed = parse_citations(f"cite [source: gate.md#H9: Completely Different | {key}]")

        report = validate_citations(
            parsed_citations=parsed,
            index_db_path=self.db_path,
            retrieval_snapshot_sha_by_key={key: sha},
            enabled=True,
            strict=False,
            require_in_snapshot=False,
            heading_match="ignore",
            normalize_heading=True,
        )
        self.assertTrue(bool(report["valid"]))
        self.assertEqual(len(report["path_mismatches"]), 1)
        self.assertEqual(report["path_mismatches"][0]["heading_mismatch_reason"], "ignored")


if __name__ == "__main__":
    unittest.main()
