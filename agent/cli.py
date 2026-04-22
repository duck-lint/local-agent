from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent.app import LocalAgentApp
from agent.runtime import print_output, render_query_results
from agent.tools import ToolError

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


class _DaemonStoreAdapter:
    """Bridges DaemonClient into the duck-typed SessionStore protocol used by grounding."""

    def __init__(self, client) -> None:
        self._client = client

    def get(self, session_id: str):
        return self._client.session_get(session_id)

    def save(self, state) -> None:
        self._client.session_update(state)

    def list(self) -> list[str]:
        return self._client.session_list()

    def clear(self, session_id: str) -> bool:
        return self._client.session_clear(session_id)


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
    p_ask.add_argument(
        "--session",
        type=str,
        default=None,
        help="Phase 3: attach to a session id for ephemeral memory snapshot/update.",
    )

    p_index = sub.add_parser("index", help="Build or refresh the corpus index.")
    p_index.add_argument("--rebuild", action="store_true", help="Rebuild chunk state for every corpus document.")
    p_index.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_index.add_argument(
        "--debug-stage-dump",
        type=str,
        default=None,
        metavar="DIR",
        help="Opt-in: write per-document stage_1 input + stage_2 chunks under DIR/<run_id>/.",
    )

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

    # ---- Phase 4: promotion + suggestion ----
    p_memory_promote = memory_sub.add_parser(
        "promote", help="Promote ephemeral session evidence into durable memory."
    )
    p_memory_promote.add_argument("--session", required=True, type=str, dest="session_id")
    p_memory_promote.add_argument(
        "--ref",
        action="append",
        default=None,
        dest="refs",
        help="chunk_key (repeatable). Required unless --llm-suggest is used.",
    )
    p_memory_promote.add_argument("--type", default="user_fact", type=str)
    p_memory_promote.add_argument("--content", default=None, type=str)
    p_memory_promote.add_argument(
        "--llm-suggest",
        action="store_true",
        help="Have the model propose a content draft; user must confirm via --yes.",
    )
    p_memory_promote.add_argument("--yes", action="store_true", help="Confirm an LLM suggestion.")
    p_memory_promote.add_argument("--json", action="store_true")

    p_memory_suggest = memory_sub.add_parser(
        "suggest", help="Show deterministic candidates for promotion (frequency-ranked)."
    )
    p_memory_suggest.add_argument("--session", required=True, type=str, dest="session_id")
    p_memory_suggest.add_argument("--json", action="store_true")

    p_doctor = sub.add_parser("doctor", help="Run deterministic runtime checks.")
    p_doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_doctor.add_argument("--no-ollama", action="store_true", help="Skip Ollama reachability checks.")
    p_doctor.add_argument(
        "--require-grounding",
        action="store_true",
        help="Fail when embeddings, retrieval, or grounding invariants are not ready.",
    )

    # ---- Phase 3: daemon + session inspection ----
    p_daemon = sub.add_parser("daemon", help="Manage the local session-memory daemon.")
    daemon_sub = p_daemon.add_subparsers(dest="daemon_cmd", required=True)
    p_daemon_start = daemon_sub.add_parser("start", help="Run the daemon in the foreground.")
    p_daemon_start.add_argument("--host", type=str, default=None)
    p_daemon_start.add_argument("--port", type=int, default=None)
    p_daemon_start.add_argument("--idle-timeout", type=int, default=None)
    p_daemon_start.add_argument("--json", action="store_true")
    p_daemon_status = daemon_sub.add_parser("status", help="Ping the daemon (fail-fast).")
    p_daemon_status.add_argument("--json", action="store_true")
    p_daemon_stop = daemon_sub.add_parser("stop", help="Request orderly daemon shutdown.")
    p_daemon_stop.add_argument("--json", action="store_true")

    p_session = sub.add_parser("session", help="Inspect ephemeral session state.")
    session_sub = p_session.add_subparsers(dest="session_cmd", required=True)
    p_session_show = session_sub.add_parser("show", help="Print a single session JSON.")
    p_session_show.add_argument("session_id", type=str)
    p_session_show.add_argument("--json", action="store_true")
    p_session_list = session_sub.add_parser("list", help="List known session ids.")
    p_session_list.add_argument("--json", action="store_true")
    p_session_clear = session_sub.add_parser("clear", help="Delete a session file.")
    p_session_clear.add_argument("session_id", type=str)
    p_session_clear.add_argument("--json", action="store_true")
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
        session_id = getattr(args, "session", None)
        session_store = None
        if session_id:
            try:
                from agent.session_memory import validate_session_id

                validate_session_id(session_id)
            except ValueError as exc:
                return _emit_error(
                    {"ok": False, "error_code": "BAD_SESSION_ID", "error_message": str(exc)}
                )
            if not app.config.session.enabled:
                return _emit_error(
                    {
                        "ok": False,
                        "error_code": "SESSION_DISABLED",
                        "error_message": "session.enabled is false in config; --session has no effect.",
                    }
                )
            if app.config.daemon.enabled and app.config.session.require_daemon_for_cli:
                from agent.daemon.client import DaemonClient, DaemonUnreachableError

                client = DaemonClient(
                    host=app.config.daemon.bind_host,
                    port=app.config.daemon.bind_port,
                    timeout_s=float(app.config.daemon.request_timeout_s),
                )
                try:
                    client.ping()
                except DaemonUnreachableError as exc:
                    return _emit_error(
                        {
                            "ok": False,
                            "error_code": "DAEMON_UNREACHABLE",
                            "error_message": str(exc),
                        }
                    )
                session_store = _DaemonStoreAdapter(client)
        result = app.answer_grounded(
            args.question,
            force_big_second=bool(getattr(args, "big", False)),
            force_fast=bool(getattr(args, "fast", False)),
            session_id=session_id,
            session_store=session_store,
        )
        if not result.ok:
            return _emit_error(
                {"ok": False, "error_code": result.error_code, "error_message": result.error_message}
            )
        print_output(result.text)
        print(f"\n[logged] {result.run_dir / 'run.json'}")
        return 0

    if args.cmd == "index":
        stage_dump_arg = getattr(args, "debug_stage_dump", None)
        stage_dump_path = Path(stage_dump_arg).expanduser().resolve() if stage_dump_arg else None
        result = app.ingest_corpus(
            force_rebuild=bool(getattr(args, "rebuild", False)),
            stage_dump_dir=stage_dump_path,
        )
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
            try:
                memory_id = app.add_memory(
                    memory_type=getattr(args, "type"),
                    source=getattr(args, "source"),
                    content=getattr(args, "content"),
                    chunk_keys=list(getattr(args, "chunk_key", None) or []),
                )
            except ValueError as exc:
                return _emit_error({"ok": False, "error_code": "MEMORY_ERROR", "error_message": str(exc)})
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
            try:
                payload = app.export_memory(getattr(args, "path"))
            except ToolError as exc:
                return _emit_error(
                    {
                        "ok": False,
                        "error_code": exc.code,
                        "error_message": str(exc),
                    }
                )
            except Exception as exc:
                return _emit_error(
                    {
                        "ok": False,
                        "error_code": "MEMORY_EXPORT_ERROR",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
            if getattr(args, "json", False):
                print_output(json.dumps(payload, ensure_ascii=False))
            else:
                print_output(f"memory exported: {getattr(args, 'path')}")
            return 0
        if action == "promote":
            return _handle_memory_promote(app, args)
        if action == "suggest":
            return _handle_memory_suggest(app, args)

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

    if args.cmd == "daemon":
        return _handle_daemon_cmd(app, args)

    if args.cmd == "session":
        return _handle_session_cmd(app, args)

    parser.print_help()
    return 2


def _handle_daemon_cmd(app: LocalAgentApp, args) -> int:
    sub = getattr(args, "daemon_cmd", "")
    cfg = app.config.daemon
    if sub == "start":
        from agent.daemon.server import DaemonServer
        from agent.session_memory import FileSessionStore

        host = getattr(args, "host", None) or cfg.bind_host
        port = getattr(args, "port", None)
        port = cfg.bind_port if port is None else int(port)
        idle = getattr(args, "idle_timeout", None)
        idle = cfg.idle_timeout_s if idle is None else int(idle)
        if app.roots.workroot is None:
            return _emit_error(
                {"ok": False, "error_code": "NO_WORKROOT", "error_message": "workroot is required for daemon"}
            )
        store = FileSessionStore(app.roots.workroot)
        server = DaemonServer(host=host, port=port, store=store, idle_timeout_s=idle)
        server.serve_forever_in_thread()
        bound_host, bound_port = server.address
        payload = {
            "ok": True,
            "host": bound_host,
            "port": bound_port,
            "idle_timeout_s": idle,
            "sessions_dir": str(store.sessions_dir),
        }
        if getattr(args, "json", False):
            print_output(json.dumps(payload, ensure_ascii=False))
        else:
            print_output(f"daemon listening on {bound_host}:{bound_port} (idle_timeout_s={idle})")
        # Foreground mode: keep the wrapper process alive only while the server is.
        server.wait_until_stopped()
        return 0
    if sub == "status":
        from agent.daemon.client import DaemonClient, DaemonError

        client = DaemonClient(host=cfg.bind_host, port=cfg.bind_port, timeout_s=float(cfg.request_timeout_s))
        try:
            client.ping()
        except DaemonError as exc:
            return _emit_error({"ok": False, "error_code": exc.code, "error_message": exc.message})
        payload = {"ok": True, "host": cfg.bind_host, "port": cfg.bind_port}
        print_output(json.dumps(payload, ensure_ascii=False) if getattr(args, "json", False) else "ok")
        return 0
    if sub == "stop":
        from agent.daemon.client import DaemonClient, DaemonError

        client = DaemonClient(host=cfg.bind_host, port=cfg.bind_port, timeout_s=float(cfg.request_timeout_s))
        try:
            client.shutdown()
        except DaemonError as exc:
            return _emit_error({"ok": False, "error_code": exc.code, "error_message": exc.message})
        payload = {"ok": True, "stopped": True}
        print_output(json.dumps(payload, ensure_ascii=False) if getattr(args, "json", False) else "stopped")
        return 0
    return _emit_error({"ok": False, "error_code": "BAD_COMMAND", "error_message": f"unknown daemon subcommand: {sub!r}"})


def _handle_session_cmd(app: LocalAgentApp, args) -> int:
    sub = getattr(args, "session_cmd", "")
    cfg = app.config
    use_daemon = cfg.daemon.enabled and cfg.session.require_daemon_for_cli

    def _store():
        if use_daemon:
            from agent.daemon.client import DaemonClient

            client = DaemonClient(
                host=cfg.daemon.bind_host,
                port=cfg.daemon.bind_port,
                timeout_s=float(cfg.daemon.request_timeout_s),
            )
            return _DaemonStoreAdapter(client)
        from agent.session_memory import FileSessionStore

        if app.roots.workroot is None:
            raise RuntimeError("workroot is required for session inspection")
        return FileSessionStore(app.roots.workroot)

    try:
        store = _store()
    except Exception as exc:
        return _emit_error({"ok": False, "error_code": "STORE_INIT_ERROR", "error_message": str(exc)})

    try:
        if sub == "show":
            from agent.session_memory import validate_session_id

            try:
                validate_session_id(args.session_id)
            except ValueError as exc:
                return _emit_error({"ok": False, "error_code": "BAD_SESSION_ID", "error_message": str(exc)})
            state = store.get(args.session_id)
            payload = {"ok": True, "state": state.to_dict()}
            print_output(json.dumps(payload, ensure_ascii=False, indent=2) if getattr(args, "json", False) else json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
            return 0
        if sub == "list":
            ids = store.list()
            payload = {"ok": True, "session_ids": ids}
            if getattr(args, "json", False):
                print_output(json.dumps(payload, ensure_ascii=False))
            else:
                for sid in ids:
                    print_output(sid)
            return 0
        if sub == "clear":
            from agent.session_memory import validate_session_id

            try:
                validate_session_id(args.session_id)
            except ValueError as exc:
                return _emit_error({"ok": False, "error_code": "BAD_SESSION_ID", "error_message": str(exc)})
            deleted = store.clear(args.session_id)
            payload = {"ok": True, "deleted": bool(deleted)}
            print_output(json.dumps(payload, ensure_ascii=False) if getattr(args, "json", False) else str(deleted))
            return 0 if deleted else 1
    except Exception as exc:  # noqa: BLE001
        return _emit_error({"ok": False, "error_code": "SESSION_ERROR", "error_message": f"{type(exc).__name__}: {exc}"})
    return _emit_error({"ok": False, "error_code": "BAD_COMMAND", "error_message": f"unknown session subcommand: {sub!r}"})


# ---- Phase 4: promotion helpers ---------------------------------------------


def _load_session_state_for_cli(app: LocalAgentApp, session_id: str):
    """Validate id + load SessionState directly from FileSessionStore.

    Promotion writes durable memory; we always read the on-disk truth, even if
    a daemon is running. This is consistent with the daemon's own storage.
    """
    from agent.session_memory import FileSessionStore, validate_session_id

    validate_session_id(session_id)
    if app.roots.workroot is None:
        raise RuntimeError("workroot is required for session promotion")
    store = FileSessionStore(app.roots.workroot)
    return store.get(session_id)


def _handle_memory_suggest(app: LocalAgentApp, args) -> int:
    cfg = app.config
    if not cfg.session.enabled:
        return _emit_error(
            {"ok": False, "error_code": "SESSION_DISABLED", "error_message": "session.enabled is false"}
        )
    try:
        state = _load_session_state_for_cli(app, args.session_id)
    except ValueError as exc:
        return _emit_error({"ok": False, "error_code": "BAD_SESSION_ID", "error_message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return _emit_error({"ok": False, "error_code": "SESSION_ERROR", "error_message": str(exc)})
    suggestions = [
        {
            "chunk_key": ref.chunk_key,
            "doc_key": ref.doc_key,
            "rel_path": ref.rel_path,
            "heading_path": ref.heading_path,
        }
        for ref in state.active_refs
    ]
    payload = {"ok": True, "session_id": args.session_id, "suggestions": suggestions}
    if getattr(args, "json", False):
        print_output(json.dumps(payload, ensure_ascii=False))
    else:
        if not suggestions:
            print_output("(no active refs in session)")
        else:
            for item in suggestions:
                print_output(f"{item['chunk_key']}\t{item['heading_path']}")
    return 0


def _handle_memory_promote(app: LocalAgentApp, args) -> int:
    cfg = app.config
    if not cfg.session.enabled:
        return _emit_error(
            {"ok": False, "error_code": "SESSION_DISABLED", "error_message": "session.enabled is false"}
        )
    if not cfg.session.promotion.enabled:
        return _emit_error(
            {
                "ok": False,
                "error_code": "PROMOTION_DISABLED",
                "error_message": "session.promotion.enabled is false",
            }
        )
    if bool(getattr(args, "llm_suggest", False)) and not cfg.session.promotion.llm_suggest_enabled:
        return _emit_error(
            {
                "ok": False,
                "error_code": "LLM_SUGGEST_DISABLED",
                "error_message": "session.promotion.llm_suggest_enabled is false",
            }
        )

    try:
        state = _load_session_state_for_cli(app, args.session_id)
    except ValueError as exc:
        return _emit_error({"ok": False, "error_code": "BAD_SESSION_ID", "error_message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return _emit_error({"ok": False, "error_code": "SESSION_ERROR", "error_message": str(exc)})

    active_keys = {ref.chunk_key: ref for ref in state.active_refs}
    requested_refs = list(getattr(args, "refs", None) or [])
    use_llm = bool(getattr(args, "llm_suggest", False))
    confirmed = bool(getattr(args, "yes", False))

    if use_llm:
        # Deterministic stub proposal — production wiring may later swap in an
        # actual LLM call. Either way, --yes is required to write.
        if not requested_refs and active_keys:
            requested_refs = [next(iter(active_keys))]
        suggested_content = (
            getattr(args, "content", None)
            or f"Promoted from session {args.session_id}: refs={','.join(requested_refs)}"
        )
        if not confirmed:
            payload = {
                "ok": True,
                "proposal": {
                    "session_id": args.session_id,
                    "type": getattr(args, "type", "user_fact"),
                    "content": suggested_content,
                    "refs": requested_refs,
                },
                "requires_confirmation": True,
            }
            print_output(json.dumps(payload, ensure_ascii=False) if getattr(args, "json", False) else json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        promoted_by = "llm_suggested_user_confirmed"
        content = suggested_content
    else:
        promoted_by = "user"
        content = getattr(args, "content", None) or f"Promoted from session {args.session_id}"

    if not requested_refs:
        return _emit_error(
            {
                "ok": False,
                "error_code": "PROMOTION_NEEDS_REF",
                "error_message": "at least one --ref is required",
            }
        )
    unknown = [r for r in requested_refs if r not in active_keys]
    if unknown:
        return _emit_error(
            {
                "ok": False,
                "error_code": "REF_NOT_IN_SESSION",
                "error_message": f"chunk_keys not in session.active_refs: {unknown}",
            }
        )

    from agent.corpus_db import connect_db as connect_corpus_db, fetch_existing_chunk_keys
    from agent.memory_db import add_promoted_memory, connect_db as connect_memory_db, init_db as init_memory_db

    try:
        memory_db_path = app.memory_db_path()
        init_memory_db(memory_db_path)
        corpus_db_path = app.corpus_db_path()
        with connect_corpus_db(corpus_db_path) as corpus_conn:
            allowed = fetch_existing_chunk_keys(corpus_conn, requested_refs)
        with connect_memory_db(memory_db_path) as conn:
            memory_id = add_promoted_memory(
                conn,
                memory_type=getattr(args, "type", "user_fact"),
                content=content,
                chunk_keys=requested_refs,
                session_id=args.session_id,
                triggering_query_ids=[state.last_query] if state.last_query else [],
                evidence_bundle_ids=list(state.last_evidence_bundle_ids),
                promoted_by=promoted_by,
                payload={
                    "topic_summary": list(state.topic_summary),
                    "turn_count_at_promotion": state.turn_count,
                },
                allowed_chunk_keys=allowed,
            )
            conn.commit()
    except ValueError as exc:
        return _emit_error({"ok": False, "error_code": "MEMORY_ERROR", "error_message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return _emit_error(
            {"ok": False, "error_code": "PROMOTION_ERROR", "error_message": f"{type(exc).__name__}: {exc}"}
        )

    payload = {
        "ok": True,
        "memory_id": memory_id,
        "session_id": args.session_id,
        "promoted_by": promoted_by,
        "refs": requested_refs,
    }
    print_output(json.dumps(payload, ensure_ascii=False) if getattr(args, "json", False) else memory_id)
    return 0
