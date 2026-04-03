from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig
from .db import FileFingerprint, IndexDatabase, metadata_json
from .lm_studio import LMStudioClient, LMStudioError
from .markdown import parse_markdown


@dataclass(slots=True)
class FilterSpec:
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @classmethod
    def from_raw(cls, raw_filters: dict | None = None, include: list[str] | None = None, exclude: list[str] | None = None) -> "FilterSpec":
        include_values = include or []
        exclude_values = exclude or []
        if raw_filters:
            include_values = [*include_values, *(raw_filters.get("include") or raw_filters.get("include_dirs") or [])]
            exclude_values = [*exclude_values, *(raw_filters.get("exclude") or raw_filters.get("exclude_dirs") or [])]
        normalized_include = tuple(sorted({value.strip("/ ") for value in include_values if value}))
        normalized_exclude = tuple(sorted({value.strip("/ ") for value in exclude_values if value}))
        return cls(include=normalized_include, exclude=normalized_exclude)


class VaultService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.load()
        self.db = IndexDatabase(self.config.index_db_path)
        self.db.initialize(reset=False)
        self.lm_studio = LMStudioClient(self.config.lm_studio_base_url)

    def close(self) -> None:
        self.db.close()

    def build_index(self, *, full: bool, dry_run: bool, filters: FilterSpec) -> dict[str, object]:
        if full and not dry_run:
            self.db.initialize(reset=True)
        indexed_rows = self.db.file_rows(filters.include, filters.exclude)
        live_files = self._discover_files(filters)
        now = self._now()

        to_delete = sorted(set(indexed_rows) - set(live_files))
        to_index: list[tuple[str, Path, os.stat_result]] = []
        touched = 0
        model_key: str | None = None
        embedding_dim: int | None = None

        for relative_path, absolute_path in sorted(live_files.items()):
            stat = absolute_path.stat()
            existing = indexed_rows.get(relative_path)
            if existing and existing["mtime_ns"] == stat.st_mtime_ns and existing["size"] == stat.st_size:
                continue
            content_hash = self._hash_file(absolute_path)
            if existing and existing["content_hash"] == content_hash:
                if not dry_run:
                    self.db.touch_file(
                        relative_path=relative_path,
                        absolute_path=str(absolute_path),
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                        content_hash=content_hash,
                        indexed_at=now,
                    )
                touched += 1
                continue
            to_index.append((relative_path, absolute_path, stat))

        if dry_run:
            return {
                "mode": "dry-run",
                "full": full,
                "indexed": len(to_index),
                "deleted": len(to_delete),
                "touched": touched,
                "unchanged": max(len(live_files) - len(to_index) - touched, 0),
                "filters": {"include": list(filters.include), "exclude": list(filters.exclude)},
                "db_path": str(self.config.index_db_path),
            }

        if to_delete:
            self.db.delete_paths(to_delete)

        if to_index:
            metadata = self.db.metadata()
            indexed_model = metadata.get("embedding_model")
            model_key = self.lm_studio.resolve_embedding_model(self.config.embedding_model_hint, indexed_model if not full else None)
            for relative_path, absolute_path, stat in to_index:
                text = absolute_path.read_text(encoding="utf-8")
                parsed = parse_markdown(relative_path, text)
                chunk_dicts: list[dict[str, str | int]] = []
                chunk_texts: list[str] = []
                for chunk in parsed.chunks:
                    chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                    chunk_dicts.append(
                        {
                            "chunk_index": chunk.chunk_index,
                            "title": chunk.title,
                            "heading_path": chunk.heading_path,
                            "tags": chunk.tags,
                            "text": chunk.text,
                            "chunk_hash": chunk_hash,
                            "word_count": chunk.word_count,
                        }
                    )
                    chunk_texts.append(chunk.text)
                vectors = self.lm_studio.embed_texts(chunk_texts, model_key) if chunk_texts else []
                if vectors:
                    embedding_dim = len(vectors[0])
                self.db.replace_file(
                    fingerprint=FileFingerprint(
                        relative_path=relative_path,
                        absolute_path=str(absolute_path),
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                        content_hash=self._hash_file(absolute_path),
                        indexed_at=now,
                    ),
                    chunk_payloads=chunk_dicts,
                    vectors=vectors,
                    embedding_model=model_key,
                    embedding_dim=embedding_dim or 0,
                )

        metadata_updates = {
            "vault_path": str(self.config.vault_path),
            "lm_studio_base_url": self.config.lm_studio_base_url,
            "last_sync_at": now,
            "filters": metadata_json({"include": list(filters.include), "exclude": list(filters.exclude)}),
        }
        if model_key:
            metadata_updates["embedding_model"] = model_key
        elif "embedding_model" not in self.db.metadata() and self.config.embedding_model_hint:
            metadata_updates["embedding_model_hint"] = self.config.embedding_model_hint
        if embedding_dim:
            metadata_updates["embedding_dim"] = str(embedding_dim)
        if full:
            metadata_updates["last_full_rebuild_at"] = now
            metadata_updates.setdefault("created_at", now)
        self.db.set_metadata(metadata_updates)

        return {
            "mode": "build" if full else "sync",
            "full": full,
            "indexed": len(to_index),
            "deleted": len(to_delete),
            "touched": touched,
            "unchanged": max(len(live_files) - len(to_index) - touched, 0),
            "embedding_model": model_key or self.db.metadata().get("embedding_model"),
            "embedding_dim": embedding_dim or self.db.metadata().get("embedding_dim"),
            "filters": {"include": list(filters.include), "exclude": list(filters.exclude)},
            "db_path": str(self.config.index_db_path),
        }

    def index_status(self) -> dict[str, object]:
        db_status = self.db.status()
        metadata = db_status["metadata"]
        lm_health = self.lm_studio.health()
        issues: list[str] = []
        if not Path(self.config.index_db_path).exists():
            issues.append("Index database does not exist yet.")
        if metadata.get("schema_version") != "1":
            issues.append("Index schema version is missing or unsupported.")
        if "embedding_model" not in metadata:
            issues.append("Index metadata is missing the embedding model.")
        if not lm_health["ok"]:
            issues.append(f"LM Studio is unavailable: {lm_health['error']}")
        return {
            "config": self.config.as_dict(),
            "index": db_status,
            "lm_studio": lm_health,
            "issues": issues,
            "ready": not issues,
        }

    def get_note(self, relative_path: str) -> dict[str, object]:
        note_row = self.db.read_note(relative_path)
        if note_row is None:
            raise FileNotFoundError(f"Note '{relative_path}' is not present in the current index.")
        note_path = Path(note_row["absolute_path"]).resolve()
        vault_root = self.config.vault_path.resolve()
        if vault_root not in note_path.parents and note_path != vault_root:
            raise ValueError(f"Resolved note path '{note_path}' is outside the configured vault.")
        content = note_path.read_text(encoding="utf-8")
        return {
            "path": note_row["relative_path"],
            "absolute_path": str(note_path),
            "indexed_at": note_row["indexed_at"],
            "content": content,
        }

    def semantic_search(self, query: str, top_k: int, filters: FilterSpec) -> dict[str, object]:
        indexed_model = self._indexed_model()
        model_key = self.lm_studio.resolve_embedding_model(self.config.embedding_model_hint, indexed_model)
        query_vector = self.lm_studio.embed_texts([query], model_key)[0]
        rows = self.db.iter_embeddings(filters.include, filters.exclude)
        if not rows:
            return {"query": query, "results": [], "summary": "Index is empty."}
        chunk_hits = []
        for row in rows:
            score = self._cosine_similarity(query_vector, self.db.blob_to_vector(row["vector"]))
            chunk_hits.append(
                {
                    "chunk_id": row["chunk_id"],
                    "relative_path": row["relative_path"],
                    "mtime_ns": row["mtime_ns"],
                    "title": row["title"],
                    "heading_path": row["heading_path"],
                    "text": row["text"],
                    "score": score,
                }
            )
        ranked = sorted(chunk_hits, key=lambda item: item["score"], reverse=True)[: max(top_k * 8, 20)]
        grouped = self._group_file_results(ranked, top_k, query=query, include_keyword_scores=False)
        return {
            "query": query,
            "mode": "semantic",
            "embedding_model": indexed_model,
            "results": grouped,
            "summary": self._summary(grouped),
        }

    def hybrid_search(self, query: str, top_k: int, filters: FilterSpec) -> dict[str, object]:
        indexed_model = self._indexed_model()
        model_key = self.lm_studio.resolve_embedding_model(self.config.embedding_model_hint, indexed_model)
        query_vector = self.lm_studio.embed_texts([query], model_key)[0]
        semantic_rows = self.db.iter_embeddings(filters.include, filters.exclude)
        semantic_hits = sorted(
            (
                {
                    "chunk_id": row["chunk_id"],
                    "relative_path": row["relative_path"],
                    "mtime_ns": row["mtime_ns"],
                    "title": row["title"],
                    "heading_path": row["heading_path"],
                    "text": row["text"],
                    "semantic_score": self._cosine_similarity(query_vector, self.db.blob_to_vector(row["vector"])),
                }
                for row in semantic_rows
            ),
            key=lambda item: item["semantic_score"],
            reverse=True,
        )[:50]
        keyword_query = self._fts_query(query)
        keyword_rows = self.db.keyword_search(keyword_query, 50, filters.include, filters.exclude)
        keyword_hits = [
            {
                "chunk_id": row["chunk_id"],
                "relative_path": row["relative_path"],
                "mtime_ns": row["mtime_ns"],
                "title": row["title"],
                "heading_path": row["heading_path"],
                "text": row["text"],
                "keyword_rank": row["keyword_rank"],
            }
            for row in keyword_rows
        ]

        merged: dict[int, dict[str, object]] = {}
        for rank, hit in enumerate(semantic_hits, start=1):
            entry = merged.setdefault(hit["chunk_id"], dict(hit))
            entry["combined_score"] = entry.get("combined_score", 0.0) + self._rrf(rank)
            entry["semantic_score"] = hit["semantic_score"]
        for rank, hit in enumerate(keyword_hits, start=1):
            entry = merged.setdefault(hit["chunk_id"], dict(hit))
            entry["combined_score"] = entry.get("combined_score", 0.0) + self._rrf(rank)
            entry["keyword_rank"] = hit["keyword_rank"]
        merged_hits = list(merged.values())
        for hit in merged_hits:
            hit["combined_score"] = float(hit.get("combined_score", 0.0)) + self._recency_boost(int(hit["mtime_ns"]))
        merged_hits.sort(key=lambda item: float(item["combined_score"]), reverse=True)
        grouped = self._group_file_results(merged_hits, top_k, query=query, include_keyword_scores=True)
        return {
            "query": query,
            "mode": "hybrid",
            "embedding_model": indexed_model,
            "results": grouped,
            "summary": self._summary(grouped),
        }

    def _discover_files(self, filters: FilterSpec) -> dict[str, Path]:
        results: dict[str, Path] = {}
        for root, _, files in os.walk(self.config.vault_path):
            for name in files:
                if not name.endswith(".md"):
                    continue
                absolute_path = Path(root) / name
                relative_path = absolute_path.relative_to(self.config.vault_path).as_posix()
                if filters.include and not any(relative_path == prefix or relative_path.startswith(f"{prefix}/") for prefix in filters.include):
                    continue
                if any(relative_path == prefix or relative_path.startswith(f"{prefix}/") for prefix in filters.exclude):
                    continue
                results[relative_path] = absolute_path
        return results

    def _indexed_model(self) -> str:
        metadata = self.db.metadata()
        indexed_model = metadata.get("embedding_model")
        if not indexed_model:
            raise LMStudioError("Index metadata is missing the embedding model. Run a full build first.")
        return indexed_model

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    @staticmethod
    def _recency_boost(mtime_ns: int) -> float:
        age_days = max((time_now_ns() - mtime_ns) / 1_000_000_000 / 86400, 0)
        return max(0.0, 0.03 * (1 - min(age_days / 365, 1)))

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[\w/-]+", query)
        if not tokens:
            raise ValueError("The query must include at least one alphanumeric term.")
        return " OR ".join(f'"{token}"*' for token in tokens if len(token) > 1)

    @staticmethod
    def _snippet(text: str, limit: int = 200) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _group_file_results(self, hits: list[dict[str, object]], top_k: int, query: str, include_keyword_scores: bool) -> list[dict[str, object]]:
        grouped: dict[str, dict[str, object]] = {}
        for hit in hits:
            relative_path = str(hit["relative_path"])
            file_entry = grouped.setdefault(
                relative_path,
                {
                    "path": relative_path,
                    "title": hit["title"],
                    "score": 0.0,
                    "chunks": [],
                },
            )
            contribution = float(
                hit.get("combined_score", hit.get("semantic_score", hit.get("score", 0.0)))
            )
            file_entry["score"] = max(float(file_entry["score"]), contribution)
            file_entry["chunks"].append(
                {
                    "chunk_id": hit["chunk_id"],
                    "heading_path": hit.get("heading_path") or "",
                    "snippet": self._snippet(str(hit["text"])),
                    "semantic_score": hit.get("semantic_score", hit.get("score")),
                    "combined_score": hit.get("combined_score", hit.get("score")),
                    "keyword_rank": hit.get("keyword_rank") if include_keyword_scores else None,
                }
            )
        ranked_files = sorted(grouped.values(), key=lambda item: float(item["score"]), reverse=True)[:top_k]
        for item in ranked_files:
            item["chunks"] = sorted(
                item["chunks"],
                key=lambda chunk: float(chunk.get("combined_score") or chunk.get("semantic_score") or 0.0),
                reverse=True,
            )[:3]
            item["query"] = query
        return ranked_files

    @staticmethod
    def _summary(grouped_results: list[dict[str, object]]) -> str:
        if not grouped_results:
            return "No matching notes found."
        lines = []
        for index, result in enumerate(grouped_results, start=1):
            top_chunk = result["chunks"][0] if result["chunks"] else {}
            snippet = top_chunk.get("snippet", "")
            lines.append(f"{index}. {result['path']} ({result['score']:.4f}) - {snippet}")
        return "\n".join(lines)


def time_now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000_000)
