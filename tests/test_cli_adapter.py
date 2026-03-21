from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.cli import build_parser, main


class _StubApp:
    def __init__(self) -> None:
        self.ask_calls: list[tuple[str, bool, bool]] = []
        self.doctor_calls: list[tuple[bool, bool]] = []

    def answer_grounded(
        self,
        question: str,
        *,
        force_big_second: bool = False,
        force_fast: bool = False,
    ):
        self.ask_calls.append((question, force_big_second, force_fast))
        run_dir = Path(tempfile.mkdtemp())
        return SimpleNamespace(
            ok=True,
            text="grounded answer",
            model_used="test-model",
            run_dir=run_dir,
            record={},
            error_code=None,
            error_message=None,
        )

    def ingest_corpus(self, *, force_rebuild: bool = False):
        return SimpleNamespace(
            errors=[],
            corpus_db_path="index/index.sqlite",
            corpus_contract_sig="corpus-v1",
            sources_total=2,
            docs_scanned=3,
            docs_changed=2,
            docs_unchanged=1,
            docs_pruned=0,
            chunks_written=5,
            total_docs=3,
            total_chunks=5,
            force_rebuild=force_rebuild,
        )

    def doctor(self, *, check_ollama: bool = True, require_grounding: bool = False):
        self.doctor_calls.append((check_ollama, require_grounding))
        return SimpleNamespace(
            ok=True,
            summary={"require_grounding": require_grounding},
            checks=[
                SimpleNamespace(
                    ok=True,
                    code="DOCTOR_OLLAMA_SKIPPED" if not check_ollama else "DOCTOR_OLLAMA_OK",
                    message="doctor check completed",
                    suggested_fix=None,
                )
            ],
        )


class CliAdapterTests(unittest.TestCase):
    def test_help_uses_current_runtime_vocabulary(self) -> None:
        obsolete_pattern = "|".join(
            [
                "".join(["ph", "ase"]),
                "".join(["st", "age"]),
                "".join(["le", "gacy"]),
            ]
        )
        root_help = build_parser().format_help().lower()
        self.assertIsNone(re.search(rf"\b({obsolete_pattern})\b", root_help))

        stdout = io.StringIO()
        with patch.object(sys, "argv", ["agent", "doctor", "--help"]):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as ctx:
                    main()
        self.assertEqual(ctx.exception.code, 0)
        doctor_help = stdout.getvalue().lower()
        self.assertIn("--require-grounding", doctor_help)
        self.assertIsNone(re.search(rf"\b({obsolete_pattern})\b", doctor_help))

    def test_index_json_is_adapter_over_structured_result(self) -> None:
        app = _StubApp()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("agent.cli.LocalAgentApp.from_config", return_value=app):
            with patch.object(sys, "argv", ["agent", "index", "--rebuild", "--json"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = main()

        self.assertEqual(rc, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["docs_changed"], 2)
        self.assertEqual(payload["total_chunks"], 5)

    def test_doctor_forwards_require_grounding_flag(self) -> None:
        app = _StubApp()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("agent.cli.LocalAgentApp.from_config", return_value=app):
            with patch.object(sys, "argv", ["agent", "doctor", "--no-ollama", "--require-grounding", "--json"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = main()

        self.assertEqual(rc, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(app.doctor_calls, [(False, True)])
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["require_grounding"], True)

    def test_ask_forwards_model_selection_flags(self) -> None:
        app = _StubApp()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("agent.cli.LocalAgentApp.from_config", return_value=app):
            with patch.object(sys, "argv", ["agent", "ask", "--big", "where is alpha?"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = main()

        self.assertEqual(rc, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(app.ask_calls, [("where is alpha?", True, False)])
        self.assertIn("grounded answer", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
