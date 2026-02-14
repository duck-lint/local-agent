from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    section_ordinal: int
    chunk_ordinal: int
    heading_path: Optional[str]


def split_frontmatter(text: str) -> Tuple[str, str]:
    """
    Return (frontmatter_block, body_text).
    frontmatter_block includes only the YAML lines between --- delimiters.
    """
    if not text:
        return "", ""

    raw = text
    if raw.startswith("\ufeff"):
        raw = raw[1:]

    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", raw

    delimiter_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            delimiter_index = i
            break

    if delimiter_index < 0:
        return "", raw

    frontmatter = "".join(lines[1:delimiter_index])
    body = "".join(lines[delimiter_index + 1 :])
    if body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
    return frontmatter, body


def _validate_chunk_args(max_chars: int, overlap: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")


def _format_heading_path(stack: Dict[int, str]) -> Optional[str]:
    if not stack:
        return None
    parts: List[str] = []
    for level in sorted(stack):
        title = stack[level].strip()
        if title:
            parts.append(f"H{level}: {title}")
    if not parts:
        return None
    return " > ".join(parts)


def _split_large_text_sentenceish(
    text: str,
    *,
    base_offset: int,
    max_chars: int,
    overlap: int,
) -> List[Tuple[str, int, int]]:
    if len(text) <= max_chars:
        return [(text, base_offset, base_offset + len(text))]

    boundary_re = re.compile(r"(?<=[.!?])(?:['\")\]]+)?\s+|[;:]\s+|\n")
    min_boundary = max(1, int(max_chars * 0.5))
    fragments: List[Tuple[str, int, int]] = []
    start = 0
    text_len = len(text)
    step = max_chars - overlap

    while start < text_len:
        window_end = min(text_len, start + max_chars)
        if window_end >= text_len:
            end = text_len
        else:
            window = text[start:window_end]
            candidates = [m.end() for m in boundary_re.finditer(window)]
            good = [idx for idx in candidates if idx >= min_boundary]
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
        piece = text[start:end]
        if piece:
            fragments.append((piece, base_offset + start, base_offset + end))
        if end >= text_len:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return fragments


def chunk_markdown_fixed_window_v1(
    body_text: str,
    max_chars: int,
    overlap: int,
) -> List[Chunk]:
    _validate_chunk_args(max_chars, overlap)
    if not body_text:
        return []

    step = max_chars - overlap
    chunks: List[Chunk] = []
    chunk_index = 0
    start = 0
    text_len = len(body_text)

    while start < text_len:
        end = min(text_len, start + max_chars)
        piece = body_text[start:end]
        if piece:
            chunks.append(
                Chunk(
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    text=piece,
                    section_ordinal=0,
                    chunk_ordinal=chunk_index,
                    heading_path=None,
                )
            )
            chunk_index += 1
        if end >= text_len:
            break
        start += step

    return chunks


@dataclass(frozen=True)
class _HeadingEvent:
    line_index: int
    start_char: int
    end_char: int
    level: int
    title: str


@dataclass(frozen=True)
class _Paragraph:
    text: str
    start_char: int
    end_char: int
    heading_path: Optional[str]


def _parse_heading_events(lines: List[str]) -> List[_HeadingEvent]:
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    out: List[_HeadingEvent] = []
    offset = 0
    for idx, line in enumerate(lines):
        m = heading_re.match(line.rstrip("\r\n"))
        end_offset = offset + len(line)
        if m:
            out.append(
                _HeadingEvent(
                    line_index=idx,
                    start_char=offset,
                    end_char=end_offset,
                    level=len(m.group(1)),
                    title=m.group(2).strip(),
                )
            )
        offset = end_offset
    return out


def _section_boundaries(headings: List[_HeadingEvent]) -> Tuple[int, List[_HeadingEvent]]:
    if not headings:
        return 0, []
    has_h2 = any(h.level == 2 for h in headings)
    split_level = 2 if has_h2 else min(h.level for h in headings)
    splits = [h for h in headings if h.level == split_level]
    return split_level, splits


def _collect_paragraphs_for_section(
    lines: List[str],
    headings_by_line: Dict[int, _HeadingEvent],
    start_line: int,
    end_line_exclusive: int,
    initial_stack: Dict[int, str],
) -> List[_Paragraph]:
    paragraphs: List[_Paragraph] = []
    stack = dict(initial_stack)

    buffer: List[str] = []
    para_start: Optional[int] = None
    para_end: Optional[int] = None
    offset = 0
    for i in range(start_line):
        offset += len(lines[i])

    def flush() -> None:
        nonlocal buffer, para_start, para_end
        if not buffer or para_start is None or para_end is None:
            buffer = []
            para_start = None
            para_end = None
            return
        text = "".join(buffer)
        if text.strip():
            paragraphs.append(
                _Paragraph(
                    text=text,
                    start_char=para_start,
                    end_char=para_end,
                    heading_path=_format_heading_path(stack),
                )
            )
        buffer = []
        para_start = None
        para_end = None

    for line_index in range(start_line, end_line_exclusive):
        line = lines[line_index]
        line_start = offset
        line_end = offset + len(line)
        offset = line_end

        event = headings_by_line.get(line_index)
        if event is not None:
            flush()
            stack = {lvl: title for lvl, title in stack.items() if lvl < event.level}
            stack[event.level] = event.title
            continue

        if not line.strip():
            flush()
            continue

        if para_start is None:
            para_start = line_start
        para_end = line_end
        buffer.append(line)

    flush()
    return paragraphs


def chunk_markdown_obsidian_v1(
    body_text: str,
    max_chars: int,
    overlap: int,
) -> List[Chunk]:
    _validate_chunk_args(max_chars, overlap)
    if not body_text:
        return []

    lines = body_text.splitlines(keepends=True)
    headings = _parse_heading_events(lines)
    headings_by_line = {h.line_index: h for h in headings}
    split_level, splits = _section_boundaries(headings)

    section_ranges: List[Tuple[int, int, Optional[_HeadingEvent], Dict[int, str]]] = []
    if not splits:
        section_ranges.append((0, len(lines), None, {}))
    else:
        first_split = splits[0]
        if first_split.line_index > 0:
            section_ranges.append((0, first_split.line_index, None, {}))

        stack: Dict[int, str] = {}
        split_ranges: List[Tuple[_HeadingEvent, Dict[int, str]]] = []
        for event in headings:
            stack = {lvl: title for lvl, title in stack.items() if lvl < event.level}
            stack[event.level] = event.title
            if event.level == split_level:
                split_ranges.append((event, dict(stack)))

        for idx, (split_event, split_stack) in enumerate(split_ranges):
            next_start = split_ranges[idx + 1][0].line_index if idx + 1 < len(split_ranges) else len(lines)
            section_start = split_event.line_index + 1
            section_ranges.append((section_start, next_start, split_event, split_stack))

    chunks: List[Chunk] = []
    global_chunk_index = 0

    for section_ordinal, (start_line, end_line, split_event, section_stack) in enumerate(section_ranges):
        _ = split_event
        paragraphs = _collect_paragraphs_for_section(
            lines=lines,
            headings_by_line=headings_by_line,
            start_line=start_line,
            end_line_exclusive=end_line,
            initial_stack=section_stack,
        )
        if not paragraphs:
            continue

        expanded: List[_Paragraph] = []
        for paragraph in paragraphs:
            if len(paragraph.text) <= max_chars:
                expanded.append(paragraph)
                continue
            pieces = _split_large_text_sentenceish(
                paragraph.text,
                base_offset=paragraph.start_char,
                max_chars=max_chars,
                overlap=overlap,
            )
            for piece_text, piece_start, piece_end in pieces:
                expanded.append(
                    _Paragraph(
                        text=piece_text,
                        start_char=piece_start,
                        end_char=piece_end,
                        heading_path=paragraph.heading_path,
                    )
                )

        chunk_ordinal = 0
        current_text = ""
        current_start = 0
        current_end = 0
        current_heading: Optional[str] = None

        def flush_current() -> None:
            nonlocal chunk_ordinal, global_chunk_index, current_text, current_start, current_end, current_heading
            if not current_text:
                return
            chunks.append(
                Chunk(
                    chunk_index=global_chunk_index,
                    start_char=current_start,
                    end_char=current_end,
                    text=current_text,
                    section_ordinal=section_ordinal,
                    chunk_ordinal=chunk_ordinal,
                    heading_path=current_heading,
                )
            )
            global_chunk_index += 1
            chunk_ordinal += 1
            current_text = ""
            current_start = 0
            current_end = 0
            current_heading = None

        for para in expanded:
            para_text = para.text
            if not current_text:
                current_text = para_text
                current_start = para.start_char
                current_end = para.end_char
                current_heading = para.heading_path
                continue

            combined_len = len(current_text) + 2 + len(para_text)
            if current_heading == para.heading_path and combined_len <= max_chars:
                current_text = f"{current_text}\n\n{para_text}"
                current_end = para.end_char
            else:
                flush_current()
                current_text = para_text
                current_start = para.start_char
                current_end = para.end_char
                current_heading = para.heading_path

        flush_current()

    return chunks


def chunk_markdown(text: str, max_chars: int, overlap: int) -> List[Chunk]:
    # Backward-compatible legacy alias: full markdown -> strip frontmatter -> fixed window v1.
    _, body = split_frontmatter(text)
    return chunk_markdown_fixed_window_v1(body_text=body, max_chars=max_chars, overlap=overlap)
