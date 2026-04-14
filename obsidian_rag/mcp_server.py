from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from .service import FilterSpec, VaultService


service = VaultService()
mcp = FastMCP("VaultSearch")
startup_status = service.index_status()


@mcp.tool()
def vault_index_status() -> dict[str, object]:
    """Report index counts, config, and LM Studio health."""
    return service.index_status()


@mcp.tool()
def vault_get_note(path: str) -> dict[str, object]:
    """Return the full contents of one indexed note by relative vault path."""
    return service.get_note(path)


@mcp.tool()
def vault_semantic_search(query: str, top_k: int | None = None, filters: dict | None = None) -> dict[str, object]:
    """Perform semantic-only retrieval across the indexed vault.
    
    When gathering evidence for concepts or literature reviews, you MUST extract from multiple distinct files.
    Do not stop at the first highly ranked file. Use vault_get_note on multiple secondary matches to ensure diverse sources.
    """
    filter_spec = FilterSpec.from_raw(filters)
    return service.semantic_search(query, top_k=top_k or service.config.top_k_default, filters=filter_spec)


@mcp.tool()
def vault_hybrid_search(query: str, top_k: int | None = None, filters: dict | None = None) -> dict[str, object]:
    """Perform hybrid semantic and FTS retrieval across the indexed vault.
    
    When gathering evidence for concepts or literature reviews, you MUST extract from multiple distinct files.
    Do not stop at the first highly ranked file. Use vault_get_note on multiple secondary matches to ensure diverse sources.
    """
    filter_spec = FilterSpec.from_raw(filters)
    return service.hybrid_search(query, top_k=top_k or service.config.top_k_default, filters=filter_spec)


def main() -> None:
    if startup_status["issues"]:
        print("VaultSearch startup issues detected:", file=sys.stderr)
        for issue in startup_status["issues"]:
            print(f"- {issue}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
