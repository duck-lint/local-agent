from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agent.app import LocalAgentApp
from agent.runtime import print_output, render_query_results

_DOCTOR_CHECK_PREFIX = {"ok": "ok", "warning": "warn", "failure": "fail"}


def _emit_error(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


def _doctor_check_state(*, ok: bool, code: str) -> str:
    if ok:
        return "ok"
    if code.endswith("_WARN"):
        return "warning"
    return "failure"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument(
        "--workroot",
        type=str,
        default=None,
        help="Data root for runs, allowed content, embeddings, and memory.",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="Override Ollama host (scheme+host[:port]).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="Send a single prompt.")
    p_chat.add_argument("prompt", type=str)

    p_ask = sub.add_parser("ask", help="Ask using grounded retrieval.")
    p_ask.add_argument("question", type=str)
    ask_speed_group = p_ask.add_mutually_exclusive_group()
    ask_speed_group.add_argument("--big", action="store_true", help="Force the larger answer model.")
    ask_speed_group.add_argument("--fast", action="store_true", help="Force the faster model path.")

    p_index = sub.add_parser("index", help="Build or refresh the corpus index.")
    p_index.add_argument("--rebuild", action="store_true", help="Rebuild chunk state for every corpus document.")
    p_index.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    p_query = sub.add_parser("query", help="Inspect lexical corpus matches.")
    p_query.add_argument("text", type=str)
    p_query.add_argument("--limit", type=int, default=5)

    p_embed = sub.add_parser("embed", help="Build or refresh embeddings for indexed chunks.")
    p_embed.add_argument("--rebuild", action="store_true", help="Rebuild every embedding row.")
    p_embed.add_argument("--limit", type=int, default=None)
    p_embed.add_argument("--dry-run", action="store_true")
    p_embed.add_argument("--no-prune", action="store_true")
    p_embed.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    p_memory = sub.add_parser("memory", help="Manage durable memory records.")
    memory_sub = p_memory.add_subparsers(dest="memory_cmd", required=True)
    p_memory_add = memory_sub.add_parser("add", help="Add a durable memory record.")
    p_memory_add.add_argument("--type", required=True)
    p_memory_add.add_argument("--source", required=True)
    p_memory_add.add_argument("--content", required=True, type=str)
    p_memory_add.add_argument("--chunk-key", action="append", default=None)
    p_memory_add.add_argument("--json", action="store_true")

    p_memory_list = memory_sub.add_parser("list", help="List durable memory records.")
    p_memory_list.add_argument("--json", action="store_true")

    p_memory_delete = memory_sub.add_parser("delete", help="Delete a durable memory record.")
    p_memory_delete.add_argument("memory_id", type=str)
    p_memory_delete.add_argument("--json", action="store_true")

    p_memory_export = memory_sub.add_parser("export", help="Export durable memory as JSON.")
    p_memory_export.add_argument("path", type=str)
    p_memory_export.add_argument("--json", action="store_true")

    p_doctor = sub.add_parser("doctor", help="Run deterministic runtime checks.")
    p_doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_doctor.add_argument("--no-ollama", action="store_true", help="Skip Ollama reachability checks.")
    p_doctor.add_argument(
        "--require-grounding",
        action="store_true",
        help="Fail when embeddings, retrieval, or grounding invariants are not ready.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        app = LocalAgentApp.from_config(
            cli_workroot=getattr(args, "workroot", None),
            cli_ollama_base_url=getattr(args, "ollama_base_url", None),
        )
    except Exception as exc:
        return _emit_error({"ok": False, "error_code": "CONFIG_ERROR", "error_message": str(exc)})

    if args.cmd == "chat":
        result = app.chat(args.prompt)
        if not result.ok:
            return _emit_error(
                {"ok": False, "error_code": result.error_code, "error_message": result.error_message}
            )
        print_output(result.text)
        print(f"\n[logged] {result.run_dir / 'run.json'}")
        return 0

    if args.cmd == "ask":
        result = app.answer_grounded(
            args.question,
            force_big_second=bool(getattr(args, "big", False)),
            force_fast=bool(getattr(args, "fast", False)),
        )
        if not result.ok:
            return _emit_error(
                {"ok": False, "error_code": result.error_code, "error_message": result.error_message}
            )
        print_output(result.text)
        print(f"\n[logged] {result.run_dir / 'run.json'}")
        return 0

    if args.cmd == "index":
        result = app.ingest_corpus(force_rebuild=bool(getattr(args, "rebuild", False)))
        payload = {
            "ok": len(result.errors) == 0,
            "corpus_db": result.corpus_db_path,
            "corpus_contract_sig": result.corpus_contract_sig,
            "sources_total": result.sources_total,
            "docs_scanned": result.docs_scanned,
            "docs_changed": result.docs_changed,
            "docs_unchanged": result.docs_unchanged,
            "docs_pruned": result.docs_pruned,
            "chunks_written": result.chunks_written,
            "total_docs": result.total_docs,
            "total_chunks": result.total_chunks,
            "errors": result.errors,
        }
        if getattr(args, "json", False):
            print_output(json.dumps(payload, ensure_ascii=False))
        else:
            print_output(
                f"corpus sync: docs_changed={result.docs_changed} docs_unchanged={result.docs_unchanged} "
                f"docs_pruned={result.docs_pruned} total_chunks={result.total_chunks}"
            )
            if result.errors:
                print_output("\n".join(result.errors))
        return 0 if not result.errors else 1

    if args.cmd == "query":
        rows = app.lexical_query(args.text, limit=int(args.limit))
        print_output(render_query_results(rows, args.text))
        return 0

    if args.cmd == "embed":
        result = app.embed_corpus(
            rebuild=bool(getattr(args, "rebuild", False)),
            limit=getattr(args, "limit", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            no_prune=bool(getattr(args, "no_prune", False)),
        )
        payload = {
            "ok": len(result.errors) == 0,
            "embeddings_db": result.embeddings_db_path,
            "total_chunks": result.total_chunks,
            "embedded_written": result.embedded_written,
            "existing_embeddings": result.existing_embeddings,
            "embeddings_total_before": result.embeddings_total_before,
            "embeddings_total_after": result.embeddings_total_after,
            "orphan_embeddings_before": result.orphan_embeddings_before,
            "orphan_embeddings_pruned": result.orphan_embeddings_pruned,
            "missing": result.missing,
            "outdated": result.outdated,
            "skipped_ok": result.skipped_ok,
            "errors": result.errors,
        }
        if getattr(args, "json", False):
            print_output(json.dumps(payload, ensure_ascii=False))
        else:
            print_output(
                f"embedding sync: written={result.embedded_written} missing={result.missing} "
                f"outdated={result.outdated} orphans_pruned={result.orphan_embeddings_pruned}"
            )
            if result.errors:
                print_output("\n".join(result.errors))
        return 0 if not result.errors else 1

    if args.cmd == "memory":
        action = getattr(args, "memory_cmd", "")
        if action == "add":
            memory_id = app.add_memory(
                memory_type=getattr(args, "type"),
                source=getattr(args, "source"),
                content=getattr(args, "content"),
                chunk_keys=list(getattr(args, "chunk_key", None) or []),
            )
            payload = {"ok": True, "memory_id": memory_id}
            print_output(json.dumps(payload, ensure_ascii=False) if getattr(args, "json", False) else memory_id)
            return 0
        if action == "list":
            items = app.list_memory()
            payload = {"ok": True, "count": len(items), "items": items}
            if getattr(args, "json", False):
                print_output(json.dumps(payload, ensure_ascii=False))
            else:
                for item in items:
                    print_output(
                        f"{item['memory_id']} type={item['type']} source={item['source']} chunk_keys={len(item['chunk_keys'])}"
                    )
            return 0
        if action == "delete":
            deleted = app.delete_memory(getattr(args, "memory_id"))
            payload = {"ok": deleted, "deleted": deleted}
            print_output(json.dumps(payload, ensure_ascii=False) if getattr(args, "json", False) else str(deleted))
            return 0 if deleted else 1
        if action == "export":
            payload = app.export_memory(getattr(args, "path"))
            if getattr(args, "json", False):
                print_output(json.dumps(payload, ensure_ascii=False))
            else:
                print_output(f"memory exported: {getattr(args, 'path')}")
            return 0

    if args.cmd == "doctor":
        report = app.doctor(
            check_ollama=not bool(getattr(args, "no_ollama", False)),
            require_grounding=bool(getattr(args, "require_grounding", False)),
        )
        payload = {
            "ok": report.ok,
            "summary": report.summary,
            "checks": [
                {
                    "state": _doctor_check_state(ok=check.ok, code=check.code),
                    "ok": check.ok,
                    "code": check.code,
                    "message": check.message,
                    "suggested_fix": check.suggested_fix,
                }
                for check in report.checks
            ],
        }
        if getattr(args, "json", False):
            print_output(json.dumps(payload, ensure_ascii=False))
        else:
            for check in report.checks:
                state = _doctor_check_state(ok=check.ok, code=check.code)
                prefix = _DOCTOR_CHECK_PREFIX.get(state, "fail")
                print_output(f"[{prefix}] {check.code}: {check.message}")
                if check.suggested_fix:
                    print_output(f"fix: {check.suggested_fix}")
        return 0 if report.ok else 1

    parser.print_help()
    return 2
