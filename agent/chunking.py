from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
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
METADATA_PROJECTION_VERSION = "metadata_v1"
LEXICAL_PROJECTION_VERSION = "lexical_v1"


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


@dataclass(frozen=True)
class Section:
    section_index: int
    anchor: str
    title: str
    heading_path: list[str]
    text: str
    start_char: int


@dataclass(frozen=True)
class MetadataProjection:
    document_title: str
    aliases: list[str]
    tags: list[str]
    doc_type: str
    entry_date: Optional[str]
    source_date: Optional[str]
    text: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def stable_doc_key_from_rel_path(rel_path: str, namespace: str = "obsidian", *, source_name: str = "") -> str:
    rel_posix = rel_path.replace("\\", "/").strip().lower()
    if source_name:
        return sha256_text(f"{namespace}:{source_name}:{rel_posix}")[:24]
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
) -> str:
    canonical_source = canonicalize_source_uri(source_uri)
    canonical_heading = " > ".join(canonicalize_heading_path(heading_path))
    kind = str(chunk_kind or CHUNK_KIND_CONTENT).strip().lower() or CHUNK_KIND_CONTENT
    return sha256_text(f"{canonical_source}|{kind}|{canonical_heading}|{section_index}|{chunk_index}")[:32]


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
    doc_type: str,
    entry_date: Optional[str],
    source_date: Optional[str],
) -> MetadataProjection:
    aliases = parse_string_list_field(meta, "aliases")
    tags = parse_string_list_field(meta, "tags")
    lines = [f"title: {document_title}"]
    if aliases:
        lines.append(f"aliases: {', '.join(aliases)}")
    if tags:
        lines.append(f"tags: {', '.join(tags)}")
    lines.append(f"doc_type: {doc_type}")
    if entry_date:
        lines.append(f"entry_date: {entry_date}")
    if source_date:
        lines.append(f"source_date: {source_date}")
    return MetadataProjection(
        document_title=document_title,
        aliases=aliases,
        tags=tags,
        doc_type=doc_type,
        entry_date=entry_date,
        source_date=source_date,
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
    return sections


def _split_large_paragraph(
    text: str,
    *,
    base_offset: int,
    max_chars: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    if len(text) <= max_chars:
        return [(text, base_offset, base_offset + len(text))]
    boundary_re = re.compile(r"(?<=[.!?])(?:['\")\]]+)?\s+|[;:]\s+|\n")
    min_boundary = max(1, int(max_chars * 0.5))
    fragments: list[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)
    step = max(1, max_chars - overlap)
    while start < text_len:
        window_end = min(text_len, start + max_chars)
        if window_end >= text_len:
            end = text_len
        else:
            window = text[start:window_end]
            candidates = [match.end() for match in boundary_re.finditer(window)]
            good = [index for index in candidates if index >= min_boundary]
            if good:
                end = start + good[-1]
            else:
                ws = window.rfind(" ")
                if ws >= min_boundary:
                    end = start + ws + 1
                else:
                    end = window_end
        if end <= start:
            end = min(text_len, start + step)
        piece = text[start:end].strip()
        if piece:
            fragments.append((piece, base_offset + start, base_offset + end))
        if end >= text_len:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return fragments


def build_markdown_chunks(
    *,
    body_text: str,
    max_chars: int,
    overlap: int,
) -> list[ChunkDraft]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
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

        draft_parts: list[tuple[str, int, int, list[dict[str, str]]]] = []
        section_offset = section.start_char
        cursor = section_offset
        for paragraph in paragraphs:
            paragraph_offset = cursor
            paragraph_pieces = _split_large_paragraph(
                paragraph,
                base_offset=paragraph_offset,
                max_chars=max_chars,
                overlap=overlap,
            )
            for piece_text, start_char, end_char in paragraph_pieces:
                cleaned_text, out_links = replace_wikilinks_and_collect(piece_text)
                cleaned_text = cleaned_text.strip()
                if cleaned_text:
                    draft_parts.append((cleaned_text, start_char, end_char, out_links))
            cursor += len(paragraph) + 2

        current_text = ""
        current_links: list[dict[str, str]] = []
        current_start = 0
        current_end = 0
        local_chunk_index = 0

        def flush_current() -> None:
            nonlocal current_text, current_links, current_start, current_end, local_chunk_index, global_chunk_index
            if not current_text:
                return
            chunks.append(
                ChunkDraft(
                    chunk_index=global_chunk_index,
                    section_index=section.section_index,
                    start_char=current_start,
                    end_char=current_end,
                    text=current_text,
                    heading_path=_format_heading_path(section.heading_path),
                    chunk_anchor=section.anchor,
                    chunk_title=section.title,
                    out_links=list(current_links),
                )
            )
            local_chunk_index += 1
            global_chunk_index += 1
            current_text = ""
            current_links = []
            current_start = 0
            current_end = 0

        for piece_text, start_char, end_char, out_links in draft_parts:
            if not current_text:
                current_text = piece_text
                current_start = start_char
                current_end = end_char
                current_links = list(out_links)
                continue
            combined = f"{current_text}\n\n{piece_text}"
            if len(combined) <= max_chars:
                current_text = combined
                current_end = end_char
                current_links.extend(out_links)
            else:
                flush_current()
                current_text = piece_text
                current_start = start_char
                current_end = end_char
                current_links = list(out_links)
        flush_current()

    return chunks


def infer_document_title(meta: dict[str, Any], rel_path: str, sections: list[Section]) -> str:
    explicit = str(meta.get("title") or "").strip()
    if explicit:
        return explicit
    for section in sections:
        if section.title and section.title != "preamble":
            return section.title
    return Path(rel_path).stem
