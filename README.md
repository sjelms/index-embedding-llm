# Obsidian Vault RAG

This project builds a local RAG index for an Obsidian vault and exposes it through one MCP server that both LM Studio and Gemini CLI can use.

The high-level flow is:

1. Read Markdown files directly from the vault filesystem.
2. Split notes into Markdown-aware chunks.
3. Create embeddings with LM Studio.
4. Store chunks, vectors, and full-text search data in a local SQLite database.
5. Query that index from LM Studio or Gemini CLI through a shared MCP server.

`mcp-obsidian` is still useful for direct note access, but it is not the primary semantic retrieval layer in this repo.

## What You Need

- [LM Studio](https://lmstudio.ai/)
- [`uv`](https://docs.astral.sh/uv/)
- An Obsidian vault on your machine
- Optional: the Obsidian Local REST API plugin, only if you also want `mcp-obsidian`

## 1. Configure the Project

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` and set:

- `VAULT_PATH`: absolute path to your Obsidian vault
- `INDEX_DB_PATH`: where you want the local SQLite index to live
- `LM_STUDIO_BASE_URL`: usually `http://127.0.0.1:1234`
- `EMBEDDING_MODEL`: the LM Studio embedding model to use
- `TOP_K_DEFAULT`: default number of search results

Example values:

```env
VAULT_PATH=/path/to/your/Obsidian Vault
INDEX_DB_PATH=/path/to/your/project/vault-index.db
LM_STUDIO_BASE_URL=http://127.0.0.1:1234
EMBEDDING_MODEL=google/embedding-gemma-300m
TOP_K_DEFAULT=5
```

## 2. Install the Local Environment

From the project directory:

```bash
uv venv .venv
uv sync
```

## 3. First-Time Build

Before you build the index:

1. Open LM Studio.
2. Start the local server.
3. Make sure the embedding model `google/embedding-gemma-300m` is available.

Check the current status:

```bash
uv run obsidian-rag index status --json
```

For the first full build, run:

```bash
uv run obsidian-rag index build --full
```

What this does:

- scans the vault
- chunks notes
- creates embeddings through LM Studio
- writes a new SQLite index
- records which embedding model was used

If you want to test on a smaller subset first:

```bash
uv run obsidian-rag index build --full --include "Some Folder"
```

If you only want to preview what would happen:

```bash
uv run obsidian-rag index build --full --dry-run --json
```

## 4. Updating the Index Later

When your vault changes, you usually do not need another full rebuild.

Run an incremental sync:

```bash
uv run obsidian-rag index sync
```

What sync does:

- adds new notes
- updates changed notes
- removes deleted notes from the index

Preview the next sync without changing anything:

```bash
uv run obsidian-rag index sync --dry-run --json
```

Use a full rebuild again only if:

- you changed the embedding model
- you want to rebuild from a clean index
- the index metadata is damaged or out of sync

## 5. Searching from the Terminal

Hybrid search:

```bash
uv run obsidian-rag search "working as learning framework" --top-k 5
```

Semantic-only search:

```bash
uv run obsidian-rag search "modular construction learning environments" --mode semantic
```

Limit the search to specific folders:

```bash
uv run obsidian-rag search "WALF" --include "Research" --exclude "Readwise"
```

## 6. Using It in LM Studio

After the index has been built, LM Studio can use it through MCP.

Steps:

1. Keep LM Studio running.
2. Make sure LM Studio is using the MCP config in `lm-studio/lm-studio_mcp.json`.
3. Restart or reload LM Studio so it picks up the MCP server config.
4. In chat, use the `vault-search` tools to search or fetch notes.

The shared MCP server exposes:

- `vault_hybrid_search`
- `vault_semantic_search`
- `vault_get_note`
- `vault_index_status`

Recommended setup in LM Studio:

- use `google/embedding-gemma-300m` for indexing and query embeddings
- use your preferred chat model, such as `gemma-4-e4b-it`, for answer generation

The chat model does not need to match the embedding model. The important rule is that the embedding model used for querying must match the embedding model used when the index was built.

## 7. Using It in Gemini CLI

Gemini CLI can use the same MCP server.

The example config is in `.gemini/settings.json`.

Once that config is active, Gemini CLI can:

- call `vault_hybrid_search`
- call `vault_semantic_search`
- fetch full notes with `vault_get_note`

This means LM Studio and Gemini CLI can share the same local index instead of maintaining separate retrieval systems.

## Health Checks

Run:

```bash
uv run obsidian-rag index status --json
```

A healthy setup should report:

- the index database exists
- schema metadata exists
- embedding model metadata exists
- LM Studio is reachable

## Security and Public Repo Notes

- Do not commit your `.env` file.
- Do not put your personal vault path into tracked files.
- Do not commit API keys or tokens.
- If you previously exposed an Obsidian Local REST API token, rotate it.

The tracked example files in this repo should stay generic and reusable by other people.

## Compatibility Notes

- The current index file is `vault-index.db`.
- The old `vault-embeddings.db` is left untouched for reference.
- The retrieval service refuses semantic search if the configured embedding model does not match the indexed model.
- Docker is optional and not required for v1 because the index is stored locally in SQLite.

## Legacy Entry Points

These filenames still exist for compatibility:

- `vault-index.py`
- `mcp-vault-search.py`

They now delegate to the new package-based implementation.
