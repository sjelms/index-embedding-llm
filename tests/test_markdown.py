import unittest

from obsidian_rag.markdown import parse_markdown
from obsidian_rag.service import FilterSpec


class MarkdownParsingTests(unittest.TestCase):
    def test_parse_markdown_uses_frontmatter_title_and_tags(self) -> None:
        text = """---
title: Example Note
tags: [one, two]
---
# Intro

First paragraph.

## Details

Second paragraph with #inline/tag.
"""
        parsed = parse_markdown("Folder/example.md", text)
        self.assertEqual(parsed.title, "Example Note")
        self.assertIn("one", parsed.tags)
        self.assertIn("inline/tag", parsed.tags)
        self.assertEqual(len(parsed.chunks), 2)
        self.assertEqual(parsed.chunks[0].heading_path, "Intro")
        self.assertEqual(parsed.chunks[1].heading_path, "Intro > Details")

    def test_parse_markdown_reads_multiline_aliases_and_term(self) -> None:
        text = """---
term: Working as Learning Framework (WALF)
aliases:
  - WALF
  - Working as Learning Framework
tags:
  - thesis
---
Short body about workplace learning.
"""
        parsed = parse_markdown("Terminology/Working as Learning Framework (WALF).md", text)
        self.assertEqual(parsed.title, "Working as Learning Framework (WALF)")
        self.assertEqual(parsed.aliases, ["WALF", "Working as Learning Framework"])
        self.assertEqual(parsed.chunks[0].aliases, "WALF, Working as Learning Framework")
        self.assertIn("thesis", parsed.tags)

    def test_filter_spec_normalizes_include_and_exclude_values(self) -> None:
        filters = FilterSpec.from_raw(
            {"include_dirs": [" AI + Code/ "], "exclude_dirs": ["Readwise/"]},
            include=["AI + Code"],
        )
        self.assertEqual(filters.include, ("AI + Code",))
        self.assertEqual(filters.exclude, ("Readwise",))


if __name__ == "__main__":
    unittest.main()
