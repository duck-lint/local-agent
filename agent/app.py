from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agent.app_types import (
    AppConfig,
    AppRoots,
    ChatResult,
    DoctorReport,
    EmbeddingSyncResult,
    GroundedAnswerResult,
)
from agent.config import (
    build_app_config,
    config_root_from_config_path,
    deep_merge_config,
    load_config_with_path,
    resolve_runtime_roots,
    root_log_fields,
)
from agent.corpus import lexical_query, sync_corpus
from agent.doctor import run_doctor
from agent.embeddings import (
    create_embedder,
    resolve_embeddings_db_path,
    resolve_memory_db_path,
    sync_embeddings,
)
from agent.grounding import answer_grounded
from agent.memory_db import (
    add_memory,
    connect_db as connect_memory_db,
    delete_memory,
    export_memory,
    init_db as init_memory_db,
    list_memory,
)
from agent.ollama_client import ensure_ollama_up, get_assistant_text, ollama_chat
from agent.retrieval import RetrievalResult, retrieve
from agent.runtime import make_run_dir, now_unix, select_models, strip_thinking
from agent.tools import ToolError, configure_tool_security


@dataclass(frozen=True)
class LocalAgentApp:
    config: AppConfig
    roots: AppRoots
    raw_config: dict[str, Any]
    resolved_config_path: Optional[Path]

    @classmethod
    def from_config(
        cls,
        *,
        cli_workroot: Optional[str] = None,
        cli_ollama_base_url: Optional[str] = None,
        start_dir: Optional[Path] = None,
        repo_root: Optional[Path] = None,
    ) -> "LocalAgentApp":
        raw_config, cfg_path = load_config_with_path(start_dir=start_dir, repo_root=repo_root)
        if cli_ollama_base_url:
            raw_config = deep_merge_config(raw_config, {"ollama_base_url": cli_ollama_base_url})
        roots = resolve_runtime_roots(
            resolved_config_path=cfg_path,
            cfg=raw_config,
            cli_workroot=cli_workroot,
        )
        app_config = build_app_config(raw_config)
        configure_tool_security(
            {
                "allowed_roots": app_config.security.allowed_roots,
                "allowed_exts": app_config.security.allowed_exts,
                "deny_absolute_paths": app_config.security.deny_absolute_paths,
                "deny_hidden_paths": app_config.security.deny_hidden_paths,
                "allow_any_path": app_config.security.allow_any_path,
                "auto_create_allowed_roots": app_config.security.auto_create_allowed_roots,
                "roots_must_be_within_security_root": app_config.security.roots_must_be_within_security_root,
            },
            workspace_root=roots.security_root,
            resolved_config_path=cfg_path,
        )
        return cls(config=app_config, roots=roots, raw_config=raw_config, resolved_config_path=cfg_path)

    def corpus_db_path(self) -> Path:
        candidate = Path(self.config.corpus.db_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.roots.security_root / candidate
        return candidate.resolve()

    def embeddings_db_path(self) -> Path:
        return resolve_embeddings_db_path(self.config.embeddings, self.roots.security_root)

    def memory_db_path(self) -> Path:
        return resolve_memory_db_path(self.config.memory.db_path, self.roots.security_root)

    def _resolve_memory_export_target(self, target_path: str) -> Path:
        target_text = str(target_path).strip()
        if not target_text:
            raise ToolError("INVALID_ARGS", "memory export path must be a non-empty string")

        target = Path(target_text).expanduser()
        if target.is_absolute() and self.config.security.deny_absolute_paths:
            raise ToolError("PATH_DENIED", f"Absolute export paths are denied by policy: {target_text}")

        if not target.is_absolute():
            target = self.roots.security_root / target

        try:
            resolved = target.resolve(strict=False)
        except OSError as exc:
            raise ToolError("PATH_DENIED", f"Invalid export path: {target_text}") from exc

        security_root = self.roots.security_root.resolve()
        try:
            rel = resolved.relative_to(security_root)
        except ValueError as exc:
            raise ToolError(
                "PATH_DENIED",
                f"Memory export path escapes security_root: {target_text}",
            ) from exc

        if self.config.security.deny_hidden_paths and any(part.startswith(".") for part in rel.parts):
            raise ToolError("PATH_DENIED", "Hidden export paths are denied by policy.")

        suffix = resolved.suffix.lower()
        if suffix != ".json":
            shown = suffix or "<none>"
            raise ToolError("PATH_DENIED", f"Memory exports must use a .json extension, got: {shown}")

        return resolved

    def chat(self, prompt: str) -> ChatResult:
        run_dir = make_run_dir(self.roots.security_root)
        started = now_unix()
        record: dict[str, Any] = {
            "run_id": run_dir.name,
            "mode": "chat",
            "prompt": prompt,
            "model": self.config.model,
            "ollama_base_url": self.config.ollama_base_url,
            "resolved_config_path": str(self.resolved_config_path) if self.resolved_config_path else None,
            "started_unix": started,
        }
        record.update(root_log_fields(self.roots))
        try:
            ensure_ollama_up(self.config.ollama_base_url, timeout_s=self.config.timeout_s)
            response = ollama_chat(
                base_url=self.config.ollama_base_url,
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout_s=self.config.timeout_s,
            )
            text = get_assistant_text(response)
            record["assistant_text"] = text
            record["raw_response"] = strip_thinking(response)
            record["ok"] = True
            return ChatResult(
                ok=True,
                text=text,
                model_used=self.config.model,
                run_dir=run_dir,
                record=record,
                raw_response=strip_thinking(response),
            )
        except Exception as exc:
            record["ok"] = False
            record["error_code"] = "CHAT_ERROR"
            record["error_message"] = str(exc)
            return ChatResult(
                ok=False,
                text="",
                model_used=self.config.model,
                run_dir=run_dir,
                record=record,
                error_code="CHAT_ERROR",
                error_message=str(exc),
            )
        finally:
            record["ended_unix"] = now_unix()
            record["elapsed_s"] = round(float(record["ended_unix"]) - started, 3)
            (run_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    def ingest_corpus(
        self,
        *,
        force_rebuild: bool = False,
    ):
        return sync_corpus(
            db_path=self.corpus_db_path(),
            source_specs=self.config.corpus.sources,
            security_root=self.roots.security_root,
            corpus_config=self.config.corpus,
            force_rebuild=force_rebuild,
        )

    def embed_corpus(
        self,
        *,
        rebuild: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False,
        no_prune: bool = False,
    ) -> EmbeddingSyncResult:
        return sync_embeddings(
            app_config=self.config,
            security_root=self.roots.security_root,
            corpus_db_path=self.corpus_db_path(),
            rebuild=rebuild,
            limit=limit,
            dry_run=dry_run,
            prune_orphans=not no_prune,
        )

    def retrieve(self, query: str) -> RetrievalResult:
        if self.config.embeddings.provider == "ollama":
            ensure_ollama_up(self.config.ollama_base_url, timeout_s=self.config.timeout_s)
        provider, model_id, preprocess_name, chunk_preprocess_sig, query_preprocess_sig, _ = (
            self.config.embeddings.provider,
            self.config.embeddings.model_id,
            self.config.embeddings.preprocess,
            self.config.embeddings.chunk_preprocess_sig,
            self.config.embeddings.query_preprocess_sig,
            self.config.embeddings.batch_size,
        )
        if not chunk_preprocess_sig or not query_preprocess_sig:
            from agent.embeddings import parse_embed_runtime

            provider, model_id, preprocess_name, chunk_preprocess_sig, query_preprocess_sig, _ = parse_embed_runtime(
                self.config.embeddings
            )
        embedder = create_embedder(
            embeddings_cfg=self.config.embeddings,
            base_url=self.config.ollama_base_url,
            timeout_s=self.config.timeout_s,
        )
        return retrieve(
            query,
            corpus_db_path=self.corpus_db_path(),
            embeddings_db_path=self.embeddings_db_path(),
            embedder=embedder,
            embed_model_id=model_id,
            preprocess_name=preprocess_name,
            chunk_preprocess_sig=chunk_preprocess_sig,
            query_preprocess_sig=query_preprocess_sig,
            lexical_k=self.config.retrieval.lexical_k,
            vector_k=self.config.retrieval.vector_k,
            vector_fetch_k=self.config.retrieval.vector_fetch_k,
            rel_path_prefix=self.config.retrieval.rel_path_prefix,
            fusion=self.config.retrieval.fusion,
        )

    def lexical_query(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        return lexical_query(db_path=self.corpus_db_path(), query_text=query, limit=limit)

    def answer_grounded(
        self,
        question: str,
        *,
        force_big_second: bool = False,
        force_fast: bool = False,
    ) -> GroundedAnswerResult:
        first_model, second_model = select_models(
            self.config,
            question,
            force_big_second=force_big_second,
            force_fast=force_fast,
        )
        _ = first_model
        try:
            return answer_grounded(
                app_config=self.config,
                security_root=self.roots.security_root,
                corpus_db_path=self.corpus_db_path(),
                question=question,
                answer_model=second_model,
                force_big_second=force_big_second,
                force_fast=force_fast,
            )
        except Exception as exc:
            run_dir = make_run_dir(self.roots.security_root)
            record = {
                "run_id": run_dir.name,
                "mode": "ask",
                "question": question,
                "ok": False,
                "error_code": "GROUNDING_ERROR",
                "error_message": str(exc),
            }
            record.update(root_log_fields(self.roots))
            (run_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return GroundedAnswerResult(
                ok=False,
                text="",
                model_used=second_model,
                run_dir=run_dir,
                record=record,
                error_code="GROUNDING_ERROR",
                error_message=str(exc),
            )

    def doctor(self, *, check_ollama: bool = True, require_grounding: bool = False) -> DoctorReport:
        return run_doctor(
            app_config=self.config,
            security_root=self.roots.security_root,
            corpus_db_path=self.corpus_db_path(),
            check_ollama=check_ollama,
            require_grounding=require_grounding,
        )

    def add_memory(self, *, memory_type: str, source: str, content: str, chunk_keys: list[str]) -> str:
        db_path = self.memory_db_path()
        init_memory_db(db_path)
        with connect_memory_db(db_path) as conn:
            memory_id = add_memory(
                conn,
                memory_type=memory_type,
                source=source,
                content=content,
                chunk_keys=chunk_keys,
            )
            conn.commit()
            return memory_id

    def list_memory(self) -> list[dict[str, object]]:
        db_path = self.memory_db_path()
        init_memory_db(db_path)
        with connect_memory_db(db_path) as conn:
            return list_memory(conn)

    def delete_memory(self, memory_id: str) -> bool:
        db_path = self.memory_db_path()
        init_memory_db(db_path)
        with connect_memory_db(db_path) as conn:
            deleted = delete_memory(conn, memory_id)
            conn.commit()
            return deleted

    def export_memory(self, target_path: str) -> dict[str, object]:
        db_path = self.memory_db_path()
        init_memory_db(db_path)
        target = self._resolve_memory_export_target(target_path)
        with connect_memory_db(db_path) as conn:
            payload = export_memory(conn, target)
            conn.commit()
            return payload
