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
    tags: str
    text: str
    chunk_hash: str
    word_count: int


@dataclass(slots=True)
class ParsedDocument:
    title: str
    tags: list[str]
    chunks: list[ChunkPayload]


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return "", text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return "", text
    return parts[0][4:], parts[1]


def _parse_frontmatter(frontmatter: str) -> tuple[str | None, list[str]]:
    title = None
    tags: list[str] = []
    for line in frontmatter.splitlines():
        key, _, value = line.partition(":")
        if not _:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "title" and value:
            title = value.strip("'").strip('"')
        elif key == "tags":
            raw_tags = value.strip("[]")
            tags.extend(tag.strip("- ").strip("'").strip('"') for tag in raw_tags.split(",") if tag.strip())
    return title, [tag for tag in tags if tag]


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
    frontmatter_title, frontmatter_tags = _parse_frontmatter(frontmatter)
    title = frontmatter_title or Path(relative_path).stem
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
                tags=", ".join(all_tags),
                text=body.strip(),
                chunk_hash="",
                word_count=len(body.split()),
            )
        )
    return ParsedDocument(title=title, tags=all_tags, chunks=chunks)
