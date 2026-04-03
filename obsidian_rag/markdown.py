from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
WIKILINK_RE = re.compile(r"!?"
                         r"\[\["
                         r"([^\]|#]+(?:#[^\]|]+)?)"
                         r"(?:\|([^\]]+))?"
                         r"\]\]")


@dataclass(slots=True)
class ChunkPayload:
    chunk_index: int
    title: str
    heading_path: str
    aliases: str
    tags: str
    related_terms: str
    text: str
    chunk_hash: str
    word_count: int


@dataclass(slots=True)
class ParsedDocument:
    title: str
    aliases: list[str]
    tags: list[str]
    related_terms: list[str]
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


def _parse_frontmatter(frontmatter: str) -> dict[str, object]:
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

    return parsed


def _normalize_tags(text: str) -> list[str]:
    tags = {match.group(1) for match in re.finditer(r"(?:^|\s)#([\w/-]+)", text)}
    return sorted(tags)


def _normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    if isinstance(value, str) and value:
        return [value]
    return []


def _normalize_relation_term(term: str) -> list[str]:
    normalized = term.strip().strip("|").strip()
    if not normalized:
        return []
    variants = {normalized}
    if normalized.startswith("@") and len(normalized) > 1:
        variants.add(normalized[1:])
    if "/" in normalized:
        leaf = Path(normalized).name.strip()
        if leaf:
            variants.add(leaf)
    return sorted(variants)


def _extract_wikilink_terms(text: str) -> list[str]:
    terms: set[str] = set()
    for target, alias in WIKILINK_RE.findall(text):
        for candidate in (target, alias):
            if not candidate:
                continue
            terms.update(_normalize_relation_term(candidate))
    return sorted(terms)


def _plain_relation_value(value: str) -> str:
    stripped = value.strip()
    if not stripped or WIKILINK_RE.search(stripped):
        return ""
    return stripped


def _interesting_frontmatter_key(key: str) -> bool:
    if key in {
        "author",
        "booktitle",
        "category",
        "editor",
        "field",
        "institution",
        "journal",
        "journaltitle",
        "key",
        "organization",
        "publisher",
        "see also",
        "type",
    }:
        return True
    return bool(re.fullmatch(r"(?:author|editor)\s*-\s*\d+", key))


def _extract_related_terms(frontmatter: dict[str, object], body: str) -> list[str]:
    terms: set[str] = set()
    for key, value in frontmatter.items():
        values = _normalize_list(value)
        for item in values:
            wikilink_terms = _extract_wikilink_terms(item)
            if wikilink_terms:
                terms.update(wikilink_terms)
            if _interesting_frontmatter_key(key):
                plain_value = _plain_relation_value(item)
                if plain_value:
                    terms.update(_normalize_relation_term(plain_value))
    terms.update(_extract_wikilink_terms(body))
    return sorted(terms)


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
    parsed_frontmatter = _parse_frontmatter(frontmatter)
    title_value = parsed_frontmatter.get("title")
    term_value = parsed_frontmatter.get("term")
    frontmatter_aliases = _normalize_list(parsed_frontmatter.get("aliases"))
    frontmatter_tags = _normalize_list(parsed_frontmatter.get("tags"))
    frontmatter_title = title_value if isinstance(title_value, str) and title_value else None
    frontmatter_term = term_value if isinstance(term_value, str) and term_value else None
    title = frontmatter_title or Path(relative_path).stem
    if frontmatter_term and frontmatter_term != title:
        frontmatter_aliases.append(frontmatter_term)
    aliases = sorted({alias for alias in frontmatter_aliases if alias and alias != title})
    inline_tags = _normalize_tags(body)
    all_tags = sorted({*frontmatter_tags, *inline_tags})
    related_terms = sorted(
        {
            term
            for term in _extract_related_terms(parsed_frontmatter, body)
            if term and term != title and term not in aliases
        }
    )

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
                    related_terms=", ".join(related_terms),
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
                related_terms=", ".join(related_terms),
                text=body.strip(),
                chunk_hash="",
                word_count=len(body.split()),
            )
        )
    return ParsedDocument(title=title, aliases=aliases, tags=all_tags, related_terms=related_terms, chunks=chunks)
