from __future__ import annotations

import json
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "3"


@dataclass(slots=True)
class FileFingerprint:
    relative_path: str
    absolute_path: str
    mtime_ns: int
    size: int
    content_hash: str
    indexed_at: str
    status: str = "indexed"


class IndexDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

    def close(self) -> None:
        self.conn.close()

    def initialize(self, reset: bool = False) -> None:
        if reset:
            self.conn.executescript(
                """
                DROP TABLE IF EXISTS embeddings;
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS files;
                DROP TABLE IF EXISTS index_metadata;
                DROP TABLE IF EXISTS fts_chunks;
                """
            )
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                relative_path TEXT NOT NULL UNIQUE,
                absolute_path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'indexed'
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                title TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL,
                related_terms TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(file_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                relative_path,
                title,
                heading_path,
                aliases,
                tags,
                related_terms,
                text,
                tokenize='unicode61'
            );
            """
        )
        self._migrate_schema()
        self.set_metadata({"schema_version": SCHEMA_VERSION})

    def _migrate_schema(self) -> None:
        chunk_columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "aliases" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN aliases TEXT NOT NULL DEFAULT ''")
        if "related_terms" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN related_terms TEXT NOT NULL DEFAULT ''")
            self.conn.commit()

        fts_columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(fts_chunks)").fetchall()
        }
        if "aliases" not in fts_columns or "related_terms" not in fts_columns:
            self.conn.execute("DROP TABLE IF EXISTS fts_chunks")
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE fts_chunks USING fts5(
                    relative_path,
                    title,
                    heading_path,
                    aliases,
                    tags,
                    related_terms,
                    text,
                    tokenize='unicode61'
                )
                """
            )
            self.conn.execute(
                """
                INSERT INTO fts_chunks(rowid, relative_path, title, heading_path, aliases, tags, related_terms, text)
                SELECT chunks.id, files.relative_path, chunks.title, chunks.heading_path, chunks.aliases, chunks.tags, chunks.related_terms, chunks.text
                FROM chunks
                JOIN files ON files.id = chunks.file_id
                """
            )
            self.conn.commit()

    def set_metadata(self, values: dict[str, str]) -> None:
        self.conn.executemany(
            """
            INSERT INTO index_metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            list(values.items()),
        )
        self.conn.commit()

    def metadata(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM index_metadata").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def file_rows(self, include_prefixes: Iterable[str] = (), exclude_prefixes: Iterable[str] = ()) -> dict[str, sqlite3.Row]:
        where_sql, params = self._prefix_filter_sql(include_prefixes, exclude_prefixes, table_alias="files")
        rows = self.conn.execute(
            f"SELECT * FROM files {where_sql}",
            params,
        ).fetchall()
        return {row["relative_path"]: row for row in rows}

    def delete_paths(self, relative_paths: Iterable[str]) -> int:
        paths = list(relative_paths)
        if not paths:
            return 0
        file_rows = self.conn.execute(
            f"SELECT id FROM files WHERE relative_path IN ({','.join('?' for _ in paths)})",
            paths,
        ).fetchall()
        file_ids = [row["id"] for row in file_rows]
        if file_ids:
            chunk_rows = self.conn.execute(
                f"SELECT id FROM chunks WHERE file_id IN ({','.join('?' for _ in file_ids)})",
                file_ids,
            ).fetchall()
            chunk_ids = [row["id"] for row in chunk_rows]
            if chunk_ids:
                self.conn.executemany("DELETE FROM fts_chunks WHERE rowid = ?", [(chunk_id,) for chunk_id in chunk_ids])
        self.conn.execute(
            f"DELETE FROM files WHERE relative_path IN ({','.join('?' for _ in paths)})",
            paths,
        )
        self.conn.commit()
        return len(paths)

    def touch_file(self, relative_path: str, absolute_path: str, mtime_ns: int, size: int, content_hash: str, indexed_at: str) -> None:
        self.conn.execute(
            """
            UPDATE files
            SET absolute_path = ?, mtime_ns = ?, size = ?, content_hash = ?, indexed_at = ?, status = 'indexed'
            WHERE relative_path = ?
            """,
            (absolute_path, mtime_ns, size, content_hash, indexed_at, relative_path),
        )
        self.conn.commit()

    def replace_file(self, fingerprint: FileFingerprint, chunk_payloads: list[dict[str, str | int]], vectors: list[list[float]], embedding_model: str, embedding_dim: int) -> None:
        with self.conn:
            row = self.conn.execute(
                "SELECT id FROM files WHERE relative_path = ?",
                (fingerprint.relative_path,),
            ).fetchone()
            if row is None:
                cursor = self.conn.execute(
                    """
                    INSERT INTO files(relative_path, absolute_path, mtime_ns, size, content_hash, indexed_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fingerprint.relative_path,
                        fingerprint.absolute_path,
                        fingerprint.mtime_ns,
                        fingerprint.size,
                        fingerprint.content_hash,
                        fingerprint.indexed_at,
                        fingerprint.status,
                    ),
                )
                file_id = cursor.lastrowid
            else:
                file_id = row["id"]
                chunk_ids = self.conn.execute(
                    "SELECT id FROM chunks WHERE file_id = ?",
                    (file_id,),
                ).fetchall()
                if chunk_ids:
                    self.conn.executemany("DELETE FROM fts_chunks WHERE rowid = ?", [(chunk_id["id"],) for chunk_id in chunk_ids])
                self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                cursor = self.conn.execute(
                    """
                    INSERT INTO files(id, relative_path, absolute_path, mtime_ns, size, content_hash, indexed_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        fingerprint.relative_path,
                        fingerprint.absolute_path,
                        fingerprint.mtime_ns,
                        fingerprint.size,
                        fingerprint.content_hash,
                        fingerprint.indexed_at,
                        fingerprint.status,
                    ),
                )
                file_id = cursor.lastrowid or file_id

            for payload, vector in zip(chunk_payloads, vectors):
                cursor = self.conn.execute(
                    """
                    INSERT INTO chunks(file_id, chunk_index, title, heading_path, aliases, tags, related_terms, text, chunk_hash, word_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        payload["chunk_index"],
                        payload["title"],
                        payload["heading_path"],
                        payload["aliases"],
                        payload["tags"],
                        payload["related_terms"],
                        payload["text"],
                        payload["chunk_hash"],
                        payload["word_count"],
                        fingerprint.indexed_at,
                    ),
                )
                chunk_id = cursor.lastrowid
                self.conn.execute(
                    """
                    INSERT INTO embeddings(chunk_id, embedding_model, embedding_dim, vector)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        embedding_model,
                        embedding_dim,
                        self.vector_to_blob(vector),
                    ),
                )
                self.conn.execute(
                    """
                    INSERT INTO fts_chunks(rowid, relative_path, title, heading_path, aliases, tags, related_terms, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        fingerprint.relative_path,
                        payload["title"],
                        payload["heading_path"],
                        payload["aliases"],
                        payload["tags"],
                        payload["related_terms"],
                        payload["text"],
                    ),
                )

    def status(self) -> dict[str, object]:
        file_count = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunk_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedding_count = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        metadata = self.metadata()
        return {
            "db_path": str(self.db_path),
            "file_count": file_count,
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
            "metadata": metadata,
        }

    def iter_embeddings(self, include_prefixes: Iterable[str] = (), exclude_prefixes: Iterable[str] = ()) -> list[sqlite3.Row]:
        where_sql, params = self._prefix_filter_sql(include_prefixes, exclude_prefixes, table_alias="files")
        rows = self.conn.execute(
            f"""
            SELECT
                chunks.id AS chunk_id,
                files.relative_path,
                files.mtime_ns,
                chunks.chunk_index,
                chunks.title,
                chunks.heading_path,
                chunks.aliases,
                chunks.tags,
                chunks.related_terms,
                chunks.text,
                embeddings.vector
            FROM embeddings
            JOIN chunks ON chunks.id = embeddings.chunk_id
            JOIN files ON files.id = chunks.file_id
            {where_sql}
            """,
            params,
        ).fetchall()
        return rows

    def keyword_search(self, fts_query: str, limit: int, include_prefixes: Iterable[str] = (), exclude_prefixes: Iterable[str] = ()) -> list[sqlite3.Row]:
        where_sql, params = self._prefix_filter_sql(include_prefixes, exclude_prefixes, table_alias="files", prepend_where=False)
        base_sql = f"""
            SELECT
                chunks.id AS chunk_id,
                files.relative_path,
                files.mtime_ns,
                chunks.chunk_index,
                chunks.title,
                chunks.heading_path,
                chunks.aliases,
                chunks.tags,
                chunks.related_terms,
                chunks.text,
                bm25(fts_chunks, 2.0, 4.0, 3.0, 3.0, 2.0, 3.0, 6.0) AS keyword_rank
            FROM fts_chunks
            JOIN chunks ON chunks.id = fts_chunks.rowid
            JOIN files ON files.id = chunks.file_id
            WHERE fts_chunks MATCH ?
        """
        if where_sql:
            base_sql += f" AND {where_sql.removeprefix('WHERE ')}"
        base_sql += " ORDER BY keyword_rank LIMIT ?"
        return self.conn.execute(base_sql, [fts_query, *params, limit]).fetchall()

    def read_note(self, relative_path: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT relative_path, absolute_path, indexed_at, mtime_ns, status FROM files WHERE relative_path = ?",
            (relative_path,),
        ).fetchone()

    @staticmethod
    def vector_to_blob(vector: list[float]) -> bytes:
        return array("f", vector).tobytes()

    @staticmethod
    def blob_to_vector(blob: bytes) -> list[float]:
        values = array("f")
        values.frombytes(blob)
        return list(values)

    @staticmethod
    def _prefix_filter_sql(include_prefixes: Iterable[str], exclude_prefixes: Iterable[str], table_alias: str, prepend_where: bool = True) -> tuple[str, list[str]]:
        clauses: list[str] = []
        params: list[str] = []
        include = [prefix.strip("/ ") for prefix in include_prefixes if prefix]
        exclude = [prefix.strip("/ ") for prefix in exclude_prefixes if prefix]
        if include:
            include_clauses = []
            for prefix in include:
                include_clauses.append(f"{table_alias}.relative_path = ? OR {table_alias}.relative_path LIKE ?")
                params.extend([prefix, f"{prefix}/%"])
            clauses.append("(" + " OR ".join(include_clauses) + ")")
        for prefix in exclude:
            clauses.append(f"{table_alias}.relative_path != ?")
            clauses.append(f"{table_alias}.relative_path NOT LIKE ?")
            params.extend([prefix, f"{prefix}/%"])
        if not clauses:
            return ("", params)
        joined = " AND ".join(clauses)
        return ((f"WHERE {joined}" if prepend_where else joined), params)


def metadata_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True)
