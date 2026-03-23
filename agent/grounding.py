from __future__ import annotations

import json
from pathlib import Path

from agent.app_types import AppConfig, GroundedAnswerResult
from agent.citation_audit import (
    build_evidence_log_entries,
    count_citation_markers,
    fetch_chunk_rows_for_keys,
    format_citation_validation_footer,
    parse_citations,
    validate_citations,
)
from agent.embeddings import create_embedder, ensure_runtime_dirs, parse_embed_runtime, resolve_embeddings_db_path
from agent.ollama_client import ensure_ollama_up, get_assistant_text, ollama_chat
from agent.retrieval import RetrievalResult, retrieve
from agent.runtime import make_run_dir, now_unix, strip_thinking


def render_citation(rel_path: str, heading_path: str, chunk_key: str) -> str:
    anchor = heading_path.strip() if heading_path.strip() else "root"
    return f"[source: {rel_path}#{anchor} | {chunk_key}]"


def build_grounded_system_prompt() -> str:
    return (
        "You are a grounded QA assistant. Use only provided retrieval evidence. "
        "Cite claims using the exact format: [source: rel_path#heading_path | chunk_key]. "
        "If evidence is insufficient, say so explicitly and ask for a narrower query. "
        "Never fabricate citations.\n\n"
        "CITATION INVARIANTS:\n"
        "1) Citation grammar is exactly: [source: <rel_path>#<heading_path> | <chunk_key>].\n"
        "2) Treat citation spans as immutable literal text.\n"
        "3) If you cannot cite correctly, output literal 'INSUFFICIENT_EVIDENCE'.\n"
    )


def build_grounded_user_prompt(question: str, retrieval_result: RetrievalResult, top_n: int) -> str:
    lines: list[str] = ["Question:", question.strip(), "", "Evidence chunks:"]
    for index, item in enumerate(retrieval_result.candidates[:top_n], start=1):
        lines.append(
            f"[{index}] chunk_key={item.chunk_key} rel_path={item.rel_path} "
            f"heading_path={item.heading_path or 'root'} method={item.method} score={item.score:.4f}"
        )
        if item.chunk_title:
            lines.append(f"title={item.chunk_title}")
        lines.append(item.text)
        lines.append("")
    lines.append("Answer using only these chunks and include citations for each claim.")
    return "\n".join(lines)


def insufficient_evidence_text(question: str) -> str:
    return (
        "Insufficient indexed evidence for this query.\n"
        "Try narrowing scope by file, heading, or exact phrase and ask again.\n"
        f"Query: {question.strip()}"
    )


def answer_grounded(
    *,
    app_config: AppConfig,
    security_root: Path,
    corpus_db_path: Path,
    question: str,
    answer_model: str,
    force_big_second: bool = False,
    force_fast: bool = False,
) -> GroundedAnswerResult:
    _ = force_big_second, force_fast
    ensure_runtime_dirs(security_root)
    run_dir = make_run_dir(security_root)
    started = now_unix()
    record: dict[str, object] = {
        "run_id": run_dir.name,
        "mode": "ask",
        "question": question,
        "started_unix": started,
        "ollama_base_url": app_config.ollama_base_url,
        "retrieval": None,
        "citation_validation": None,
    }
    try:
        ensure_ollama_up(app_config.ollama_base_url, timeout_s=app_config.timeout_s)
        provider, model_id, preprocess_name, chunk_preprocess_sig, query_preprocess_sig, _ = parse_embed_runtime(
            app_config.embeddings
        )
        embedder = create_embedder(
            embeddings_cfg=app_config.embeddings,
            base_url=app_config.ollama_base_url,
            timeout_s=app_config.timeout_s,
        )
        retrieval_result = retrieve(
            question,
            corpus_db_path=corpus_db_path,
            embeddings_db_path=resolve_embeddings_db_path(app_config.embeddings, security_root),
            embedder=embedder,
            embed_model_id=model_id,
            preprocess_name=preprocess_name,
            chunk_preprocess_sig=chunk_preprocess_sig,
            query_preprocess_sig=query_preprocess_sig,
            lexical_k=app_config.retrieval.lexical_k,
            vector_k=app_config.retrieval.vector_k,
            vector_fetch_k=app_config.retrieval.vector_fetch_k,
            rel_path_prefix=app_config.retrieval.rel_path_prefix,
            fusion=app_config.retrieval.fusion,
        )
        prompt_candidates = retrieval_result.candidates[: app_config.grounding.evidence_top_n]
        evidence_rows = {}
        if prompt_candidates:
            evidence_rows = fetch_chunk_rows_for_keys(
                corpus_db_path=corpus_db_path,
                chunk_keys=[item.chunk_key for item in prompt_candidates],
            )
        evidence_entries, logging_truncated_total, omitted_count = build_evidence_log_entries(
            candidates=prompt_candidates,
            chunk_rows=evidence_rows,
            max_total_chars=app_config.runs.max_total_evidence_chars,
            max_excerpt_chars=app_config.runs.max_excerpt_chars,
        )
        retrieval_snapshot_hash_by_key = {
            item.chunk_key: str(evidence_rows[item.chunk_key].chunk_hash)
            for item in prompt_candidates
            if item.chunk_key in evidence_rows
        }
        record["retrieval"] = {
            "query": retrieval_result.query,
            "corpus_contract_sig": retrieval_result.corpus_contract_sig,
            "embed_model_id": retrieval_result.embed_model_id,
            "chunk_preprocess_sig": retrieval_result.chunk_preprocess_sig,
            "query_preprocess_sig": retrieval_result.query_preprocess_sig,
            "embed_db_schema_version": retrieval_result.embed_db_schema_version,
            "lexical_backend_mode": retrieval_result.lexical_backend_mode,
            "lexical_backend_warning": retrieval_result.lexical_backend_warning,
            "vector_fetch_k_used": retrieval_result.vector_fetch_k_used,
            "vector_candidates_scored": retrieval_result.vector_candidates_scored,
            "vector_candidates_prefilter": retrieval_result.vector_candidates_prefilter,
            "vector_candidates_postfilter": retrieval_result.vector_candidates_postfilter,
            "rel_path_prefix_applied": retrieval_result.rel_path_prefix_applied,
            "vector_filter_warning": retrieval_result.vector_filter_warning,
            "rerank_applied": retrieval_result.rerank_applied,
            "rerank_intent": retrieval_result.rerank_intent,
            "rerank_signals_available": retrieval_result.rerank_signals_available,
            "candidates_count": len(retrieval_result.candidates),
            "results": evidence_entries,
            "logging_truncated_total": bool(logging_truncated_total),
            "results_omitted_count": int(omitted_count),
        }

        raw_answer_text = insufficient_evidence_text(question)
        second = None
        if retrieval_result.candidates:
            prompt = build_grounded_user_prompt(
                question,
                retrieval_result,
                top_n=app_config.grounding.evidence_top_n,
            )
            second = ollama_chat(
                base_url=app_config.ollama_base_url,
                model=answer_model,
                messages=[
                    {"role": "system", "content": build_grounded_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=app_config.temperature,
                max_tokens=app_config.max_tokens_big_second,
                timeout_s=max(app_config.timeout_s, app_config.timeout_s_big_second),
            )
            raw_answer_text = get_assistant_text(second)

        parsed_raw_citations = parse_citations(raw_answer_text)
        citation_report = validate_citations(
            parsed_citations=parsed_raw_citations,
            corpus_db_path=corpus_db_path,
            retrieval_snapshot_hash_by_key=retrieval_snapshot_hash_by_key,
            enabled=app_config.grounding.citation_validation.enabled,
            strict=app_config.grounding.citation_validation.strict,
            require_in_snapshot=app_config.grounding.citation_validation.require_in_snapshot,
            heading_match=app_config.grounding.citation_validation.heading_match,
            normalize_heading=app_config.grounding.citation_validation.normalize_heading,
            citation_markers_found=count_citation_markers(raw_answer_text),
        )
        footer = format_citation_validation_footer(citation_report)
        record["citation_validation"] = citation_report
        record["citation_validation_footer"] = footer

        final_text = raw_answer_text
        if not retrieval_result.candidates:
            final_text = insufficient_evidence_text(question)
        elif not app_config.grounding.citation_validation.enabled and "source:" not in raw_answer_text:
            fallback_lines = [
                "Insufficient citation-grounded answer from model output.",
                "Evidence references:",
            ]
            for item in prompt_candidates[:5]:
                fallback_lines.append(render_citation(item.rel_path, item.heading_path, item.chunk_key))
            final_text = "\n".join(fallback_lines)

        if app_config.grounding.citation_validation.enabled and not citation_report.get("valid", False):
            if app_config.grounding.citation_validation.strict:
                error_message = f"Citation validation failed {footer}."
                record["ok"] = False
                record["error_code"] = "ASK_CITATION_INVALID"
                record["error_message"] = error_message
                if second is not None:
                    record["raw_second"] = strip_thinking(second)
                return GroundedAnswerResult(
                    ok=False,
                    text=final_text,
                    model_used=answer_model,
                    run_dir=run_dir,
                    record=record,
                    error_code="ASK_CITATION_INVALID",
                    error_message=error_message,
                )

        record["ok"] = True
        if second is not None:
            record["raw_second"] = strip_thinking(second)
        return GroundedAnswerResult(
            ok=True,
            text=f"{final_text}\n{footer}" if app_config.grounding.citation_validation.enabled else final_text,
            model_used=answer_model,
            run_dir=run_dir,
            record=record,
        )
    finally:
        record["ended_unix"] = now_unix()
        record["elapsed_s"] = round(float(record["ended_unix"]) - started, 3)
        (run_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
