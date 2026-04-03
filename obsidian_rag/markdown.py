from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(slots=True)
class ChunkPayload:
    chunk_index: int
    title: str
    heading_path: str
    aliases: str
    tags: str
    text: str
    chunk_hash: str
    word_count: int


@dataclass(slots=True)
class ParsedDocument:
    title: str
    aliases: list[str]
    tags: list[str]
    chunks: list[ChunkPayload]


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return "", text
    return parts[0][4:], parts[1]


def _strip_scalar(value: str) -> str:
    return value.strip().strip("'").strip('"')


def _parse_scalar_or_list(value: str) -> str | list[str]:
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        items = [
            _strip_scalar(item)
            for item in stripped[1:-1].split(",")
            if _strip_scalar(item)
        ]
        return items
    return _strip_scalar(stripped)


def _append_frontmatter_value(target: dict[str, object], key: str, value: str) -> None:
    parsed = _parse_scalar_or_list(value)
    existing = target.get(key)
    if isinstance(parsed, list):
        existing_values = existing if isinstance(existing, list) else []
        target[key] = [*existing_values, *parsed]
        return
    if isinstance(existing, list):
        existing.append(parsed)
        return
    target[key] = parsed


def _parse_frontmatter(frontmatter: str) -> tuple[str | None, list[str], list[str]]:
    parsed: dict[str, object] = {}
    current_key: str | None = None
    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_key:
            existing = parsed.setdefault(current_key, [])
            if isinstance(existing, list):
                existing.append(_strip_scalar(stripped[2:]))
            continue
        key, separator, value = line.partition(":")
        if not separator:
            current_key = None
            continue
        current_key = key.strip().lower()
        if value.strip():
            _append_frontmatter_value(parsed, current_key, value)
        else:
            parsed.setdefault(current_key, [])

    title = parsed.get("title")
    term = parsed.get("term")
    aliases = parsed.get("aliases") or []
    tags = parsed.get("tags") or []

    normalized_title = title if isinstance(title, str) and title else None
    normalized_term = term if isinstance(term, str) and term else None
    normalized_aliases = [alias for alias in aliases if isinstance(alias, str) and alias]
    normalized_tags = [tag for tag in tags if isinstance(tag, str) and tag]

    return normalized_title or normalized_term, normalized_aliases, normalized_tags


def _normalize_tags(text: str) -> list[str]:
    tags = {match.group(1) for match in re.finditer(r"(?:^|\s)#([\w/-]+)", text)}
    return sorted(tags)


def _make_heading_path(stack: list[str]) -> str:
    return " > ".join(part for part in stack if part)


def _yield_sections(body: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading = ""
    for line in body.splitlines():
        heading_match = HEADING_RE.match(line)
        if heading_match:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(heading_text)
            current_heading = _make_heading_path(heading_stack)
            current_lines.append(line)
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def _paragraph_chunks(section_text: str, max_words: int = 380, overlap_words: int = 40) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", section_text) if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        word_count = len(words)
        if word_count > max_words:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_words = 0
            start = 0
            while start < word_count:
                end = min(start + max_words, word_count)
                chunks.append(" ".join(words[start:end]).strip())
                if end == word_count:
                    break
                start = max(end - overlap_words, start + 1)
            continue
        if current and current_words + word_count > max_words:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += word_count
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def parse_markdown(relative_path: str, text: str) -> ParsedDocument:
    frontmatter, body = _split_frontmatter(text)
    frontmatter_title, frontmatter_aliases, frontmatter_tags = _parse_frontmatter(frontmatter)
    title = frontmatter_title or Path(relative_path).stem
    aliases = sorted({alias for alias in frontmatter_aliases if alias and alias != title})
    inline_tags = _normalize_tags(body)
    all_tags = sorted({*frontmatter_tags, *inline_tags})

    chunks: list[ChunkPayload] = []
    chunk_index = 0
    for heading_path, section_text in _yield_sections(body):
        section_chunks = _paragraph_chunks(section_text)
        for chunk_text in section_chunks:
            if not chunk_text.strip():
                continue
            word_count = len(chunk_text.split())
            chunk_hash = ""
            chunks.append(
                ChunkPayload(
                    chunk_index=chunk_index,
                    title=title,
                    heading_path=heading_path,
                    aliases=", ".join(aliases),
                    tags=", ".join(all_tags),
                    text=chunk_text,
                    chunk_hash=chunk_hash,
                    word_count=word_count,
                )
            )
            chunk_index += 1
    if not chunks and body.strip():
        chunks.append(
            ChunkPayload(
                chunk_index=0,
                title=title,
                heading_path="",
                aliases=", ".join(aliases),
                tags=", ".join(all_tags),
                text=body.strip(),
                chunk_hash="",
                word_count=len(body.split()),
            )
        )
    return ParsedDocument(title=title, aliases=aliases, tags=all_tags, chunks=chunks)
