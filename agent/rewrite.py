from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RewrittenQuery:
    original: str
    rewritten: str
    transforms_applied: list[str] = field(default_factory=list)
    acronyms_expanded: list[tuple[str, str]] = field(default_factory=list)
    synonyms_injected: list[tuple[str, str]] = field(default_factory=list)
    active_refs_appended: list[str] = field(default_factory=list)
    topic_keywords_appended: list[str] = field(default_factory=list)

    def is_identity(self) -> bool:
        return self.rewritten == self.original and not self.transforms_applied

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "transforms_applied": list(self.transforms_applied),
            "acronyms_expanded": [list(p) for p in self.acronyms_expanded],
            "synonyms_injected": [list(p) for p in self.synonyms_injected],
            "active_refs_appended": list(self.active_refs_appended),
            "topic_keywords_appended": list(self.topic_keywords_appended),
        }


def _norm_map(raw: Optional[Mapping[str, str]]) -> dict[str, str]:
    if not raw:
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        key = k.strip()
        val = v.strip()
        if not key or not val:
            continue
        out[key.lower()] = val
    return out


def _memory_fields(memory: object | None) -> tuple[list[str], list[object]]:
    """Best-effort extraction of (topic_summary, active_refs) from a memory object.

    Accepts SessionState (attributes) or a plain mapping. Missing fields → empty.
    """
    if memory is None:
        return [], []
    if isinstance(memory, dict):
        ts_raw = memory.get("topic_summary") or []
        refs_raw = memory.get("active_refs") or []
    else:
        ts_raw = getattr(memory, "topic_summary", None) or []
        refs_raw = getattr(memory, "active_refs", None) or []
    topic_summary = [str(t) for t in ts_raw if isinstance(t, str) and t.strip()]
    active_refs = list(refs_raw)
    return topic_summary, active_refs


def _ref_text(ref: object) -> str:
    if isinstance(ref, dict):
        return str(ref.get("heading_path") or "").strip()
    return str(getattr(ref, "heading_path", "") or "").strip()


def rule_based_rewrite(
    query: str,
    *,
    acronyms: Optional[Mapping[str, str]] = None,
    synonyms: Optional[Mapping[str, str]] = None,
    memory: object | None = None,
) -> RewrittenQuery:
    """Deterministic, LLM-free query rewrite.

    Transforms applied (in order, each appends suffix tokens):
      - Acronym expansion: token (case-insensitive) matches an entry in `acronyms`.
      - Synonym injection: same shape, separate map.
      - (Phase 4) Active-ref heading tokens: each token from a memory.active_refs
        ChunkRef.heading_path that is NOT already present in the rewritten query.
      - (Phase 4) Topic-summary keywords: each topic_summary token not already
        present in the rewritten query.

    Memory contributes lexical seeds only; it never replaces or removes any
    user-supplied tokens. If no transforms apply, returns identity.
    """
    original = query
    acro = _norm_map(acronyms)
    syn = _norm_map(synonyms)
    topic_summary, active_refs = _memory_fields(memory)

    tokens = _TOKEN_RE.findall(query)
    seen_acro: set[str] = set()
    seen_syn: set[str] = set()
    acro_hits: list[tuple[str, str]] = []
    syn_hits: list[tuple[str, str]] = []
    for tok in tokens:
        lower = tok.lower()
        if lower in acro and lower not in seen_acro:
            acro_hits.append((tok, acro[lower]))
            seen_acro.add(lower)
        if lower in syn and lower not in seen_syn:
            syn_hits.append((tok, syn[lower]))
            seen_syn.add(lower)

    # Tokens already in the (about-to-be-built) rewritten query, lowercase.
    present: set[str] = {t.lower() for t in tokens}
    for _, exp in acro_hits + syn_hits:
        for tok in _TOKEN_RE.findall(exp):
            present.add(tok.lower())

    ref_tokens_appended: list[str] = []
    refs_used: list[str] = []
    for ref in active_refs:
        text = _ref_text(ref)
        if not text:
            continue
        new_for_this_ref: list[str] = []
        for tok in _TOKEN_RE.findall(text):
            lower = tok.lower()
            if lower in present:
                continue
            present.add(lower)
            new_for_this_ref.append(tok)
            ref_tokens_appended.append(tok)
        if new_for_this_ref:
            refs_used.append(text)

    topic_kw_appended: list[str] = []
    for kw in topic_summary:
        for tok in _TOKEN_RE.findall(kw):
            lower = tok.lower()
            if lower in present:
                continue
            present.add(lower)
            topic_kw_appended.append(tok)

    transforms: list[str] = []
    if acro_hits:
        transforms.append("acronym_expansion")
    if syn_hits:
        transforms.append("synonym_injection")
    if ref_tokens_appended:
        transforms.append("memory_active_refs")
    if topic_kw_appended:
        transforms.append("memory_topic_summary")

    if not transforms:
        return RewrittenQuery(original=original, rewritten=original)

    suffix_parts: list[str] = (
        [exp for _, exp in acro_hits]
        + [exp for _, exp in syn_hits]
        + ref_tokens_appended
        + topic_kw_appended
    )
    rewritten = original.rstrip()
    if suffix_parts:
        rewritten = rewritten + " " + " ".join(suffix_parts)

    return RewrittenQuery(
        original=original,
        rewritten=rewritten,
        transforms_applied=transforms,
        acronyms_expanded=acro_hits,
        synonyms_injected=syn_hits,
        active_refs_appended=refs_used,
        topic_keywords_appended=topic_kw_appended,
    )
