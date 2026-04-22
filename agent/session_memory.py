"""Phase 3 ephemeral session memory.

Design constraints:
  * In v1 session memory does NOT influence retrieval at all (Phase 4 will turn
    on the rewrite_only memory channel). Therefore this module only owns
    persistence + deterministic state updates.
  * Memory text is NEVER allowed to enter the evidence snapshot or the prompt.
    `SessionState.to_snapshot_dict()` is the only thing that reaches the run
    record and is serialised verbatim.
  * Storage is per-session JSON under ``<workroot>/sessions/<session_id>.json``.
    This is intentionally separate from ``durable.sqlite``; promoted records
    live there (Phase 4).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_TOPIC_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

# Negation cues used by the 3-word upstream window in extract_topic_terms.
# Mirrors the requirement_analysis.rs NEGATION_CUES set; deliberately small.
_NEGATION_CUES = frozenset({
    "not", "no", "without", "never",
    "don't", "don\u2019t",
    "doesn't", "doesn\u2019t",
    "isn't", "isn\u2019t",
    "won't", "won\u2019t",
    "aren't", "aren\u2019t",
    "cannot",
})

# Static stoplist; keep small/lowercased. No NLP libs.
_TOPIC_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "are",
    "you", "your", "yours", "what", "when", "where", "which", "while", "who",
    "why", "into", "onto", "over", "under", "about", "above", "below",
    "between", "but", "not", "any", "all", "some", "per", "via", "very",
    "more", "most", "much", "many", "such", "than", "then", "they", "their",
    "them", "those", "these", "his", "her", "hers", "him", "she", "was",
    "were", "been", "being", "is", "an", "as", "at", "be", "by", "do",
    "does", "doing", "done", "if", "in", "of", "on", "or", "to", "us", "we",
    "it", "its", "itself", "let", "lets", "may", "might", "must", "shall",
    "should", "would", "could", "can", "cannot", "also", "just", "only",
    "other", "another", "each", "few", "own", "same", "too", "out",
})


def validate_session_id(session_id: str) -> str:
    """Validate that ``session_id`` is a safe filesystem-friendly slug.

    Returns the validated id; raises ``ValueError`` on rejection. Allowing
    arbitrary strings is unsafe because the session id becomes a filename
    component.
    """
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a string")
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            "session_id must match [A-Za-z0-9._-]{1,64}; "
            f"got {session_id!r}"
        )
    return session_id


@dataclass(frozen=True)
class ChunkRef:
    chunk_key: str
    doc_key: str
    rel_path: str = ""
    heading_path: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_key": self.chunk_key,
            "doc_key": self.doc_key,
            "rel_path": self.rel_path,
            "heading_path": self.heading_path,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ChunkRef":
        return cls(
            chunk_key=str(raw.get("chunk_key", "")),
            doc_key=str(raw.get("doc_key", "")),
            rel_path=str(raw.get("rel_path", "")),
            heading_path=str(raw.get("heading_path", "")),
        )


@dataclass(frozen=True)
class SessionState:
    """In-memory view of a session.

    schema_version pinned to 1 for Phase 3.
    """

    session_id: str
    schema_version: int = 1
    created_unix: float = 0.0
    updated_unix: float = 0.0
    turn_count: int = 0
    topic_summary: list[str] = field(default_factory=list)
    active_refs: list[ChunkRef] = field(default_factory=list)
    last_evidence_bundle_ids: list[str] = field(default_factory=list)
    last_query: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "created_unix": self.created_unix,
            "updated_unix": self.updated_unix,
            "turn_count": self.turn_count,
            "topic_summary": list(self.topic_summary),
            "active_refs": [r.to_dict() for r in self.active_refs],
            "last_evidence_bundle_ids": list(self.last_evidence_bundle_ids),
            "last_query": self.last_query,
        }

    def to_snapshot_dict(self) -> dict:
        """Snapshot embedded in run.json. Identical to to_dict for Phase 3."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, raw: dict) -> "SessionState":
        refs_raw = raw.get("active_refs") or []
        return cls(
            session_id=str(raw.get("session_id", "")),
            schema_version=int(raw.get("schema_version", 1)),
            created_unix=float(raw.get("created_unix", 0.0)),
            updated_unix=float(raw.get("updated_unix", 0.0)),
            turn_count=int(raw.get("turn_count", 0)),
            topic_summary=[str(t) for t in (raw.get("topic_summary") or [])],
            active_refs=[ChunkRef.from_dict(r) for r in refs_raw if isinstance(r, dict)],
            last_evidence_bundle_ids=[str(b) for b in (raw.get("last_evidence_bundle_ids") or [])],
            last_query=str(raw.get("last_query", "")),
        )

    @classmethod
    def empty(cls, session_id: str, *, now_unix: float) -> "SessionState":
        return cls(
            session_id=validate_session_id(session_id),
            created_unix=float(now_unix),
            updated_unix=float(now_unix),
        )


@dataclass(frozen=True)
class TopicTerm:
    """A single token extracted from the current turn, with provenance.

    ``source`` is "query" or "answer".
    ``polarity`` is "asserted" or "negated" (negated = the token appeared
    within 3 words after a negation cue like "not"/"never"/"without").
    """
    text: str
    source: str
    polarity: str


def extract_topic_terms(*, query: str, answer_text: str) -> list["TopicTerm"]:
    """Tokenize the current turn into typed ``TopicTerm``s.

    Lowercases, drops stopwords and tokens shorter than 3 chars. For each
    answer-side token, looks at the 3 raw words immediately preceding it; if
    any is a negation cue, the term is tagged ``polarity="negated"``. Query
    tokens are always asserted (queries do not negate themselves).
    """
    out: list[TopicTerm] = []

    if query:
        for match in _TOPIC_TOKEN_RE.finditer(query.lower()):
            tok = match.group(0)
            if tok in _TOPIC_STOPWORDS:
                continue
            out.append(TopicTerm(text=tok, source="query", polarity="asserted"))

    if answer_text:
        lowered = answer_text.lower()
        for match in _TOPIC_TOKEN_RE.finditer(lowered):
            tok = match.group(0)
            if tok in _TOPIC_STOPWORDS:
                continue
            prefix_words = lowered[: match.start()].split()
            window = prefix_words[-3:]
            polarity = "negated" if any(w in _NEGATION_CUES for w in window) else "asserted"
            out.append(TopicTerm(text=tok, source="answer", polarity=polarity))

    return out


def compute_topic_summary(
    *,
    query: str,
    answer_text: str,
    previous_summary: Sequence[str] = (),
    top_k: int = 8,
) -> list[str]:
    """Deterministic top-k tf-ish keyword summary.

    Pipeline:
      1. ``extract_topic_terms`` produces typed ``TopicTerm``s from the current
         turn (query + answer), tagged with ``source`` and ``polarity``.
      2. Negated terms are dropped — a refusal like "the evidence does not
         contain X" no longer poisons the summary with X.
      3. Asserted terms are tallied (query x2, answer x1). Tokens whose only
         evidence is a single answer occurrence are dropped (count >= 2 for
         answer-only tokens) so that one-shot refusal meta-vocabulary
         ("documents", "definition", "list", ...) cannot survive.
      4. ``previous_summary`` tokens are carried forward with +1 weight so the
         topic still drifts smoothly across turns; they are exempt from the
         answer-only threshold because they are already curated.

    Ties are broken by first-occurrence order. No LLM, no external NLP.
    """
    terms = extract_topic_terms(query=query, answer_text=answer_text)

    counts: dict[str, int] = {}
    sources: dict[str, set[str]] = {}
    order: dict[str, int] = {}

    for term in terms:
        if term.polarity == "negated":
            continue
        weight = 2 if term.source == "query" else 1
        counts[term.text] = counts.get(term.text, 0) + weight
        sources.setdefault(term.text, set()).add(term.source)
        order.setdefault(term.text, len(order))

    surviving: dict[str, int] = {
        tok: c
        for tok, c in counts.items()
        if "query" in sources[tok] or c >= 2
    }

    for prev in previous_summary:
        prev_lower = str(prev).lower().strip()
        if not prev_lower or prev_lower in _TOPIC_STOPWORDS:
            continue
        if not _TOPIC_TOKEN_RE.fullmatch(prev_lower):
            continue
        surviving[prev_lower] = surviving.get(prev_lower, 0) + 1
        order.setdefault(prev_lower, len(order))

    if not surviving:
        return []

    ranked = sorted(surviving.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    return [tok for tok, _ in ranked[: max(1, int(top_k))]]


def compute_state_update(
    *,
    previous: SessionState,
    query: str,
    answer_text: str,
    final_chunk_refs: Sequence[ChunkRef],
    bundle_id: str,
    max_active_refs: int = 10,
    max_bundle_ids: int = 5,
    topic_top_k: int = 8,
    now_unix: Optional[float] = None,
) -> SessionState:
    """Pure deterministic transition: previous + answer artefacts -> new state."""
    ts = float(now_unix if now_unix is not None else time.time())
    summary = compute_topic_summary(
        query=query,
        answer_text=answer_text,
        previous_summary=previous.topic_summary,
        top_k=topic_top_k,
    )

    # Active refs: most recent first, dedupe by chunk_key, cap at max_active_refs.
    seen_keys: set[str] = set()
    new_refs: list[ChunkRef] = []
    for ref in final_chunk_refs:
        if not ref.chunk_key or ref.chunk_key in seen_keys:
            continue
        seen_keys.add(ref.chunk_key)
        new_refs.append(ref)
    for ref in previous.active_refs:
        if ref.chunk_key in seen_keys:
            continue
        seen_keys.add(ref.chunk_key)
        new_refs.append(ref)
    new_refs = new_refs[:max_active_refs]

    bundle_ids: list[str] = []
    if bundle_id:
        bundle_ids.append(bundle_id)
    for bid in previous.last_evidence_bundle_ids:
        if bid and bid not in bundle_ids:
            bundle_ids.append(bid)
    bundle_ids = bundle_ids[:max_bundle_ids]

    return SessionState(
        session_id=previous.session_id,
        schema_version=previous.schema_version,
        created_unix=previous.created_unix or ts,
        updated_unix=ts,
        turn_count=previous.turn_count + 1,
        topic_summary=summary,
        active_refs=new_refs,
        last_evidence_bundle_ids=bundle_ids,
        last_query=query,
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class SessionStoreError(Exception):
    """Base class for session store I/O failures."""


class FileSessionStore:
    """Per-session JSON file storage under ``<root>/sessions/``.

    Thread-safe via a single in-process lock; the daemon serialises requests
    in its dispatcher anyway, but tests that exercise multiple stores in the
    same process benefit from the lock.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._sessions_dir = self._root / "sessions"
        self._lock = threading.Lock()

    @property
    def sessions_dir(self) -> Path:
        return self._sessions_dir

    def _path_for(self, session_id: str) -> Path:
        validate_session_id(session_id)
        return self._sessions_dir / f"{session_id}.json"

    def get(self, session_id: str) -> SessionState:
        path = self._path_for(session_id)
        with self._lock:
            if not path.exists():
                return SessionState.empty(session_id, now_unix=time.time())
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SessionStoreError(f"failed to read session {session_id!r}: {exc}") from exc
            if not isinstance(raw, dict):
                raise SessionStoreError(f"session file {path} is not a JSON object")
            state = SessionState.from_dict(raw)
            # Re-stamp session_id to match filename (defensive).
            if state.session_id != session_id:
                state = SessionState(
                    session_id=session_id,
                    schema_version=state.schema_version,
                    created_unix=state.created_unix,
                    updated_unix=state.updated_unix,
                    turn_count=state.turn_count,
                    topic_summary=state.topic_summary,
                    active_refs=state.active_refs,
                    last_evidence_bundle_ids=state.last_evidence_bundle_ids,
                    last_query=state.last_query,
                )
            return state

    def save(self, state: SessionState) -> None:
        path = self._path_for(state.session_id)
        payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)
        with self._lock:
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)

    def list(self) -> list[str]:
        with self._lock:
            if not self._sessions_dir.exists():
                return []
            return sorted(p.stem for p in self._sessions_dir.glob("*.json") if p.is_file())

    def clear(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        with self._lock:
            if not path.exists():
                return False
            try:
                path.unlink()
            except OSError as exc:
                raise SessionStoreError(f"failed to clear session {session_id!r}: {exc}") from exc
            return True


def make_bundle_id(*, run_id: str, candidates: Iterable) -> str:
    """Stable opaque id for the evidence bundle of a run.

    For Phase 3 we use the run_id (already deterministic). The candidates
    parameter is reserved for a future content-hash variant.
    """
    _ = candidates
    return f"bundle:{run_id}"
