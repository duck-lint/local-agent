from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.cli import build_parser, main
from agent.tools import ToolError


class _StubApp:
    def __init__(self) -> None:
        self.ask_calls: list[tuple[str, bool, bool]] = []
        self.doctor_calls: list[tuple[bool, bool]] = []
        self.denied_export_paths: set[str] = {"../memory-export.json"}
        self.doctor_result = SimpleNamespace(
            ok=True,
            summary={"require_grounding": False},
            checks=[
                SimpleNamespace(
                    ok=True,
                    code="DOCTOR_OLLAMA_OK",
                    message="doctor check completed",
                    suggested_fix=None,
                )
            ],
        )

    def answer_grounded(
        self,
        question: str,
        *,
        force_big_second: bool = False,
        force_fast: bool = False,
        session_id=None,
        session_store=None,
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

    def ingest_corpus(self, *, force_rebuild: bool = False, stage_dump_dir=None):
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
        result = deepcopy(self.doctor_result)
        if getattr(result, "summary", None) is not None:
            result.summary["require_grounding"] = require_grounding
        if getattr(result, "checks", None) and len(result.checks) > 0 and str(result.checks[0].code).startswith(
            "DOCTOR_OLLAMA"
        ):
            result.checks[0].code = "DOCTOR_OLLAMA_SKIPPED" if not check_ollama else "DOCTOR_OLLAMA_OK"
        return result

    def export_memory(self, path: str):
        if path in self.denied_export_paths:
            raise ToolError("PATH_DENIED", "Memory export path escapes security_root")
        return {"ok": True, "schema_version": 2, "items": []}


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
        self.assertEqual(payload["checks"][0]["state"], "ok")

    def test_doctor_human_output_labels_warning_without_failing_command(self) -> None:
        app = _StubApp()
        app.doctor_result = SimpleNamespace(
            ok=True,
            summary={"require_grounding": False},
            checks=[
                SimpleNamespace(
                    ok=False,
                    code="DOCTOR_EMBEDDINGS_MISSING_WARN",
                    message="Embeddings DB is missing.",
                    suggested_fix="Run: local-agent embed --json",
                )
            ],
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("agent.cli.LocalAgentApp.from_config", return_value=app):
            with patch.object(sys, "argv", ["agent", "doctor"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = main()

        self.assertEqual(rc, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("[warn] DOCTOR_EMBEDDINGS_MISSING_WARN: Embeddings DB is missing.", stdout.getvalue())
        self.assertIn("fix: Run: local-agent embed --json", stdout.getvalue())

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

    def test_memory_export_reports_typed_path_error(self) -> None:
        app = _StubApp()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("agent.cli.LocalAgentApp.from_config", return_value=app):
            with patch.object(sys, "argv", ["agent", "memory", "export", "../memory-export.json", "--json"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = main()

        self.assertEqual(rc, 1)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["ok"], False)
        self.assertEqual(payload["error_code"], "PATH_DENIED")


if __name__ == "__main__":
    unittest.main()
