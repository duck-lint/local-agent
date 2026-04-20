from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(\|([^\]]+))?\]\]")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

CHUNK_KIND_CONTENT = "content"
CHUNK_KIND_METADATA = "metadata"
METADATA_HEADING_PATH = "META: frontmatter"
METADATA_CHUNK_ANCHOR = "frontmatter"
METADATA_CHUNK_TITLE = "frontmatter"
METADATA_CHUNK_INDEX = -1
METADATA_SECTION_INDEX = -1
METADATA_PROJECTION_VERSION = "metadata_v2"
LEXICAL_PROJECTION_VERSION = "lexical_v2"


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    section_index: int
    start_char: int
    end_char: int
    text: str
    heading_path: str
    chunk_anchor: str
    chunk_title: str
    out_links: list[dict[str, str]]
    section_ordinal: Optional[int] = None


@dataclass(frozen=True)
class Section:
    section_index: int
    anchor: str
    title: str
    heading_path: list[str]
    text: str
    start_char: int
    section_ordinal: Optional[int] = None


@dataclass(frozen=True)
class MetadataProjection:
    note_type: str
    aliases: list[str]
    tags: list[str]
    journal_entry_date: Optional[str]
    canonical_name: str
    layer: str
    register: str
    text: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def stable_doc_key_from_rel_path(rel_path: str, namespace: str = "obsidian") -> str:
    rel_posix = rel_path.replace("\\", "/").strip().lower()
    return sha256_text(f"{namespace}:{rel_posix}")[:24]


def canonicalize_source_uri(source_uri: str) -> str:
    cleaned = source_uri.strip().replace("\\", "/")
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def canonicalize_heading_path(heading_path: Any) -> list[str]:
    if isinstance(heading_path, list):
        parts = heading_path
    elif isinstance(heading_path, str):
        parts = [part.strip() for part in heading_path.split(">")]
    else:
        parts = []
    out: list[str] = []
    for item in parts:
        if not isinstance(item, str):
            continue
        cleaned = _normalize_heading_text(item)
        if cleaned:
            out.append(cleaned)
    return out


def stable_chunk_key(
    *,
    source_uri: str,
    chunk_kind: str,
    heading_path: list[str],
    section_index: int,
    chunk_index: int,
    section_ordinal: Optional[int] = None,
) -> str:
    canonical_source = canonicalize_source_uri(source_uri)
    canonical_heading = " > ".join(canonicalize_heading_path(heading_path))
    kind = str(chunk_kind or CHUNK_KIND_CONTENT).strip().lower() or CHUNK_KIND_CONTENT
    if section_ordinal is None:
        material = f"{canonical_source}|{kind}|{canonical_heading}|{section_index}|{chunk_index}"
    else:
        material = (
            f"{canonical_source}|{kind}|{canonical_heading}|"
            f"{section_index}#{int(section_ordinal)}|{chunk_index}"
        )
    return sha256_text(material)[:32]


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    raw = text[1:] if text.startswith("\ufeff") else text
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", raw
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            yaml_block = "\n".join(lines[1:index]).strip()
            body = "\n".join(lines[index + 1 :]).lstrip("\n")
            return yaml_block, body
    return "", raw


def parse_yaml_frontmatter(yaml_text: str) -> tuple[dict[str, Any], Optional[str]]:
    if not yaml_text.strip():
        return {}, None
    try:
        import yaml

        loaded = yaml.safe_load(yaml_text)
        if loaded is None:
            return {}, None
        if not isinstance(loaded, dict):
            return {}, "frontmatter must be a mapping"
        return loaded, None
    except Exception as exc:
        return {}, str(exc)


def slugify(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-") or "section"


def parse_date_field(meta: dict[str, Any], key: str) -> Optional[str]:
    raw = str(meta.get(key, "")).strip()
    if not raw:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    try:
        return datetime.fromisoformat(raw).date().isoformat()
    except Exception:
        return None


def parse_source_date(meta: dict[str, Any], filename: str) -> Optional[str]:
    from_meta = parse_date_field(meta, "note_creation_date")
    if from_meta:
        return from_meta
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)
    return None


def parse_string_list_field(meta: dict[str, Any], key: str) -> list[str]:
    raw = meta.get(key)
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def parse_string_field(meta: dict[str, Any], key: str) -> str:
    raw = meta.get(key)
    if raw is None:
        return ""
    return str(raw).strip()


def normalize_doc_type(
    meta: dict[str, Any],
    *,
    folder: str,
    entry_date: Optional[str],
) -> str:
    explicit = str(meta.get("doc_type") or "").strip().lower()
    if explicit:
        return explicit
    note_type = str(meta.get("note_type") or "").strip().lower()
    if note_type:
        if note_type in {"journal", "journal_entry", "journal-entry"}:
            return "journal"
        return note_type
    note_status = str(meta.get("note_status") or "").strip().lower()
    if note_status in {"journal", "journal_entry", "journal-entry"}:
        return "journal"
    if entry_date:
        return "journal"
    return folder.lower() if folder else "note"


def build_metadata_projection(
    *,
    meta: dict[str, Any],
    document_title: str,
    entry_date: Optional[str],
) -> MetadataProjection:
    note_type = parse_string_field(meta, "note_type")
    aliases = parse_string_list_field(meta, "aliases")
    tags = parse_string_list_field(meta, "tags")
    canonical_name = parse_string_field(meta, "canonical_name") or document_title
    layer = parse_string_field(meta, "layer")
    register = parse_string_field(meta, "register")
    lines: list[str] = []
    if note_type:
        lines.append(f"note_type: {note_type}")
    if aliases:
        lines.append(f"aliases: {', '.join(aliases)}")
    if tags:
        lines.append(f"tags: {', '.join(tags)}")
    if entry_date:
        lines.append(f"journal_entry_date: {entry_date}")
    if canonical_name:
        lines.append(f"canonical_name: {canonical_name}")
    if layer:
        lines.append(f"layer: {layer}")
    if register:
        lines.append(f"register: {register}")
    return MetadataProjection(
        note_type=note_type,
        aliases=aliases,
        tags=tags,
        journal_entry_date=entry_date,
        canonical_name=canonical_name,
        layer=layer,
        register=register,
        text="\n".join(lines).strip() + "\n",
    )


def normalize_markdown_light(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code and line.lstrip().startswith(">"):
            out.append(line.lstrip()[1:].lstrip())
            continue
        out.append(line.rstrip("\n"))
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (text + "\n") if text else ""


def replace_wikilinks_and_collect(text: str) -> tuple[str, list[dict[str, str]]]:
    out_links: list[dict[str, str]] = []

    def _replace(match: re.Match[str]) -> str:
        target = (match.group(1) or "").strip()
        alias = (match.group(3) or "").strip()
        item: dict[str, str] = {"target": target}
        if alias:
            item["alias"] = alias
        out_links.append(item)
        return alias if alias else target

    cleaned = WIKILINK_RE.sub(_replace, text)
    return cleaned, out_links


def extract_wikilinks(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in WIKILINK_RE.finditer(text):
        target = (match.group(1) or "").strip()
        if not target or target in seen:
            continue
        seen.add(target)
        out.append(target)
    return out


def _format_heading_path(path: list[str]) -> str:
    return " > ".join(path)


def _pick_split_level(levels: list[int]) -> int:
    if 2 in levels:
        return 2
    return min(levels)


def split_into_sections(body_markdown: str) -> list[Section]:
    lines = body_markdown.splitlines()
    if not lines:
        return []
    levels_present: list[int] = []
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            levels_present.append(len(match.group(1)))
    if not levels_present:
        text = "\n".join(lines).strip()
        if not text:
            return []
        return [
            Section(
                section_index=0,
                anchor="preamble",
                title="preamble",
                heading_path=[],
                text=text + "\n",
                start_char=0,
            )
        ]

    target_level = _pick_split_level(levels_present)
    sections: list[Section] = []
    current_title = "preamble"
    current_path: list[str] = []
    current_lines: list[str] = []
    current_start = 0
    heading_stack: dict[int, str] = {}
    offset = 0
    section_index = 0

    def flush() -> None:
        nonlocal current_lines, section_index, current_start
        text = "\n".join(current_lines).strip()
        if current_title == "preamble" and not text:
            current_lines = []
            current_start = offset
            return
        if text:
            sections.append(
                Section(
                    section_index=section_index,
                    anchor=slugify(current_title),
                    title=current_title,
                    heading_path=list(current_path),
                    text=text + "\n",
                    start_char=current_start,
                )
            )
            section_index += 1
        current_lines = []
        current_start = offset

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[level] = title
            for key in list(heading_stack):
                if key > level:
                    del heading_stack[key]
            if level == target_level:
                flush()
                current_title = title
                current_path = [f"H{lvl}: {heading_stack[lvl]}" for lvl in sorted(heading_stack) if lvl <= target_level]
                offset += len(line) + 1
                current_start = offset
                continue
        current_lines.append(line)
        offset += len(line) + 1
    flush()
    return _assign_section_ordinals(sections)


def _assign_section_ordinals(sections: list[Section]) -> list[Section]:
    if not sections:
        return sections
    keys = [tuple(canonicalize_heading_path(section.heading_path)) for section in sections]
    counts = Counter(keys)
    seen: dict[tuple[str, ...], int] = defaultdict(int)
    out: list[Section] = []
    for section, key in zip(sections, keys):
        if counts[key] > 1:
            ordinal: Optional[int] = seen[key]
            seen[key] += 1
        else:
            ordinal = None
        out.append(replace(section, section_ordinal=ordinal))
    return out


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_paragraph_by_sentences(
    paragraph: str,
    *,
    base_offset: int,
    max_chars: int,
) -> list[tuple[str, int, int]]:
    sentences = [sent for sent in _SENTENCE_SPLIT_RE.split(paragraph) if sent]
    if not sentences:
        return [(paragraph, base_offset, base_offset + len(paragraph))]
    pieces: list[tuple[str, int, int]] = []
    buf = ""
    buf_start = base_offset
    cursor = base_offset
    for sent in sentences:
        addition = sent if not buf else f" {sent}"
        if buf and len(buf) + len(addition) > max_chars:
            pieces.append((buf, buf_start, buf_start + len(buf)))
            buf = sent
            buf_start = cursor
        else:
            buf += addition
        cursor += len(sent) + 1
    if buf:
        pieces.append((buf, buf_start, buf_start + len(buf)))
    return pieces


def build_markdown_chunks(
    *,
    body_text: str,
    max_chars: int,
) -> list[ChunkDraft]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if not body_text.strip():
        return []

    sections = split_into_sections(body_text)
    chunks: list[ChunkDraft] = []
    global_chunk_index = 0

    for section in sections:
        normalized = normalize_markdown_light(section.text)
        paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            continue

        cursor = section.start_char
        for paragraph in paragraphs:
            paragraph_offset = cursor
            paragraph_len = len(paragraph)
            if paragraph_len <= max_chars:
                pieces = [(paragraph, paragraph_offset, paragraph_offset + paragraph_len)]
            else:
                pieces = _split_paragraph_by_sentences(
                    paragraph,
                    base_offset=paragraph_offset,
                    max_chars=max_chars,
                )
            for piece_text, start_char, end_char in pieces:
                cleaned_text, out_links = replace_wikilinks_and_collect(piece_text)
                cleaned_text = cleaned_text.strip()
                if not cleaned_text:
                    continue
                chunks.append(
                    ChunkDraft(
                        chunk_index=global_chunk_index,
                        section_index=section.section_index,
                        start_char=start_char,
                        end_char=end_char,
                        text=cleaned_text,
                        heading_path=_format_heading_path(section.heading_path),
                        chunk_anchor=section.anchor,
                        chunk_title=section.title,
                        out_links=out_links,
                        section_ordinal=section.section_ordinal,
                    )
                )
                global_chunk_index += 1
            cursor += paragraph_len + 2

    return chunks


def infer_document_title(meta: dict[str, Any], rel_path: str, sections: list[Section]) -> str:
    explicit = str(meta.get("title") or "").strip()
    if explicit:
        return explicit
    for section in sections:
        if section.title and section.title != "preamble":
            return section.title
    return Path(rel_path).stem
