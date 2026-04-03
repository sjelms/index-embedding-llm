from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TextIO

from .config import AppConfig
from .db import SCHEMA_VERSION, FileFingerprint, IndexDatabase, metadata_json
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


@dataclass(slots=True)
class PendingFile:
    relative_path: str
    absolute_path: Path
    stat: os.stat_result
    content_hash: str


@dataclass(slots=True)
class FailureRecord:
    relative_path: str
    stage: str
    error: str


class BuildRunLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("a", encoding="utf-8")

    def event(self, event_type: str, **payload: object) -> None:
        record = {
            "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "event": event_type,
            **payload,
        }
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class ProgressReporter:
    def __init__(self, *, output: TextIO | None, logger: BuildRunLogger | None, enabled: bool) -> None:
        self.output = output
        self.logger = logger
        self.enabled = enabled and output is not None
        self.interactive = bool(self.enabled and getattr(output, "isatty", lambda: False)())
        self.start_time = perf_counter()
        self.processed_files = 0
        self.total_files = 0
        self.completed_chunks = 0
        self.failed_files = 0
        self.last_path = ""
        self.last_stage = ""

    def start_run(self, *, mode: str, total_files: int, unchanged: int, touched: int, deleted: int, filters: FilterSpec, log_path: Path) -> None:
        self.total_files = total_files
        summary = (
            f"{mode}: {total_files} files to index, {unchanged} unchanged, "
            f"{touched} metadata-only updates, {deleted} deletions"
        )
        self._emit(summary)
        if filters.include or filters.exclude:
            self._emit(
                f"filters: include={list(filters.include) or ['*']} exclude={list(filters.exclude) or []}"
            )
        self._emit(f"run log: {log_path}")
        if self.logger:
            self.logger.event(
                "run_started",
                mode=mode,
                total_files=total_files,
                unchanged=unchanged,
                touched=touched,
                deleted=deleted,
                filters={"include": list(filters.include), "exclude": list(filters.exclude)},
            )

    def file_stage(self, *, index: int, relative_path: str, stage: str) -> None:
        self.last_path = relative_path
        self.last_stage = stage
        percent = self._percent(index - 1)
        eta = self._eta(index - 1)
        message = (
            f"[{index}/{self.total_files}] {percent:6.2f}% "
            f"stage={stage:<8} elapsed={self._elapsed():>7} eta={eta:>7} "
            f"path={relative_path}"
        )
        self._emit(message, transient=True)
        if self.logger:
            self.logger.event(
                "file_stage",
                index=index,
                total=self.total_files,
                stage=stage,
                relative_path=relative_path,
            )

    def file_done(self, *, index: int, relative_path: str, chunks: int, seconds: float) -> None:
        self.processed_files = index
        self.completed_chunks += chunks
        files_per_sec = self.processed_files / max(perf_counter() - self.start_time, 0.001)
        chunks_per_sec = self.completed_chunks / max(perf_counter() - self.start_time, 0.001)
        message = (
            f"[{index}/{self.total_files}] {self._percent(index):6.2f}% "
            f"done      chunks={chunks:<4} file_time={seconds:5.1f}s "
            f"rate={files_per_sec:4.2f} files/s {chunks_per_sec:5.2f} chunks/s "
            f"path={relative_path}"
        )
        self._emit(message)
        if self.logger:
            self.logger.event(
                "file_done",
                index=index,
                total=self.total_files,
                relative_path=relative_path,
                chunks=chunks,
                file_seconds=round(seconds, 3),
                elapsed_seconds=round(perf_counter() - self.start_time, 3),
            )

    def file_failed(self, *, index: int, relative_path: str, stage: str, error: str) -> None:
        self.processed_files = index
        self.failed_files += 1
        message = (
            f"[{index}/{self.total_files}] {self._percent(index):6.2f}% "
            f"ERROR     stage={stage:<8} path={relative_path} error={error}"
        )
        self._emit(message)
        if self.logger:
            self.logger.event(
                "file_failed",
                index=index,
                total=self.total_files,
                relative_path=relative_path,
                stage=stage,
                error=error,
            )

    def finish(self, *, indexed: int, failed: list[FailureRecord]) -> None:
        elapsed = perf_counter() - self.start_time
        eta = self._eta(self.processed_files)
        self._emit(
            f"finished: indexed={indexed}, failed={len(failed)}, chunks={self.completed_chunks}, "
            f"elapsed={self._format_seconds(elapsed)}, remaining_eta={eta}"
        )
        if self.logger:
            self.logger.event(
                "run_finished",
                indexed=indexed,
                failed=len(failed),
                chunks=self.completed_chunks,
                elapsed_seconds=round(elapsed, 3),
            )

    def _percent(self, processed: int) -> float:
        if not self.total_files:
            return 100.0
        return processed / self.total_files * 100

    def _elapsed(self) -> str:
        return self._format_seconds(perf_counter() - self.start_time)

    def _eta(self, processed: int) -> str:
        if processed <= 0 or processed >= self.total_files:
            return "--:--"
        elapsed = perf_counter() - self.start_time
        per_file = elapsed / processed
        remaining = per_file * max(self.total_files - processed, 0)
        return self._format_seconds(remaining)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        total = max(int(seconds), 0)
        minutes, sec = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _emit(self, message: str, transient: bool = False) -> None:
        if not self.enabled or self.output is None:
            return
        if self.interactive and transient:
            print(message[:200], end="\r", file=self.output, flush=True)
            return
        if self.interactive and self.last_stage:
            print(" " * 220, end="\r", file=self.output)
        print(message, file=self.output, flush=True)


class VaultService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.load()
        self.db = IndexDatabase(self.config.index_db_path)
        self.db.initialize(reset=False)
        self.lm_studio = LMStudioClient(self.config.lm_studio_base_url)

    def close(self) -> None:
        self.db.close()

    def build_index(
        self,
        *,
        full: bool,
        dry_run: bool,
        filters: FilterSpec,
        progress_output: TextIO | None = None,
        show_progress: bool = False,
    ) -> dict[str, object]:
        if full and not dry_run:
            self.db.initialize(reset=True)
        indexed_rows = self.db.file_rows(filters.include, filters.exclude)
        live_files = self._discover_files(filters)
        now = self._now()

        to_delete = sorted(set(indexed_rows) - set(live_files))
        to_index: list[PendingFile] = []
        touched = 0
        model_key: str | None = None
        embedding_dim: int | None = None
        failures: list[FailureRecord] = []
        log_path = self._build_log_path(now)
        logger = BuildRunLogger(log_path)
        progress = ProgressReporter(output=progress_output, logger=logger, enabled=show_progress and not dry_run)

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
            to_index.append(
                PendingFile(
                    relative_path=relative_path,
                    absolute_path=absolute_path,
                    stat=stat,
                    content_hash=content_hash,
                )
            )

        unchanged = max(len(live_files) - len(to_index) - touched, 0)

        if dry_run:
            logger.event(
                "dry_run",
                full=full,
                indexed=len(to_index),
                deleted=len(to_delete),
                touched=touched,
                unchanged=unchanged,
            )
            logger.close()
            return {
                "mode": "dry-run",
                "full": full,
                "indexed": len(to_index),
                "deleted": len(to_delete),
                "touched": touched,
                "unchanged": unchanged,
                "filters": {"include": list(filters.include), "exclude": list(filters.exclude)},
                "db_path": str(self.config.index_db_path),
                "log_path": str(log_path),
            }

        progress.start_run(
            mode="build" if full else "sync",
            total_files=len(to_index),
            unchanged=unchanged,
            touched=touched,
            deleted=len(to_delete),
            filters=filters,
            log_path=log_path,
        )

        try:
            if to_delete:
                self.db.delete_paths(to_delete)
                logger.event("deleted_paths", count=len(to_delete))

            if to_index:
                metadata = self.db.metadata()
                indexed_model = metadata.get("embedding_model")
                model_key = self.lm_studio.resolve_embedding_model(
                    self.config.embedding_model_hint,
                    indexed_model if not full else None,
                )
                self.db.set_metadata(
                    {
                        "embedding_model": model_key,
                        "vault_path": str(self.config.vault_path),
                        "lm_studio_base_url": self.config.lm_studio_base_url,
                        "filters": metadata_json({"include": list(filters.include), "exclude": list(filters.exclude)}),
                        "created_at": self.db.metadata().get("created_at", now),
                    }
                )
                indexed_successes = 0
                for index, pending in enumerate(to_index, start=1):
                    file_started_at = perf_counter()
                    try:
                        progress.file_stage(index=index, relative_path=pending.relative_path, stage="read")
                        text = pending.absolute_path.read_text(encoding="utf-8")
                        progress.file_stage(index=index, relative_path=pending.relative_path, stage="parse")
                        parsed = parse_markdown(pending.relative_path, text)
                        chunk_dicts: list[dict[str, str | int]] = []
                        chunk_texts: list[str] = []
                        for chunk in parsed.chunks:
                            chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                            chunk_dicts.append(
                                {
                                    "chunk_index": chunk.chunk_index,
                                    "title": chunk.title,
                                    "heading_path": chunk.heading_path,
                                    "aliases": chunk.aliases,
                                    "tags": chunk.tags,
                                    "text": chunk.text,
                                    "chunk_hash": chunk_hash,
                                    "word_count": chunk.word_count,
                                }
                            )
                            chunk_texts.append(
                                self._embedding_input(
                                    title=chunk.title,
                                    aliases=chunk.aliases,
                                    heading_path=chunk.heading_path,
                                    tags=chunk.tags,
                                    text=chunk.text,
                                )
                            )
                        progress.file_stage(index=index, relative_path=pending.relative_path, stage="embed")
                        vectors = self.lm_studio.embed_texts(chunk_texts, model_key) if chunk_texts else []
                        if vectors:
                            embedding_dim = len(vectors[0])
                        progress.file_stage(index=index, relative_path=pending.relative_path, stage="write")
                        self.db.replace_file(
                            fingerprint=FileFingerprint(
                                relative_path=pending.relative_path,
                                absolute_path=str(pending.absolute_path),
                                mtime_ns=pending.stat.st_mtime_ns,
                                size=pending.stat.st_size,
                                content_hash=pending.content_hash,
                                indexed_at=now,
                            ),
                            chunk_payloads=chunk_dicts,
                            vectors=vectors,
                            embedding_model=model_key,
                            embedding_dim=embedding_dim or 0,
                        )
                        indexed_successes += 1
                        progress.file_done(
                            index=index,
                            relative_path=pending.relative_path,
                            chunks=len(chunk_dicts),
                            seconds=perf_counter() - file_started_at,
                        )
                    except Exception as exc:  # noqa: BLE001
                        stage = progress.last_stage or "unknown"
                        error_message = str(exc)
                        failures.append(
                            FailureRecord(
                                relative_path=pending.relative_path,
                                stage=stage,
                                error=error_message,
                            )
                        )
                        progress.file_failed(
                            index=index,
                            relative_path=pending.relative_path,
                            stage=stage,
                            error=error_message,
                        )
                progress.finish(indexed=indexed_successes, failed=failures)

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
                "indexed": len(to_index) - len(failures),
                "deleted": len(to_delete),
                "touched": touched,
                "unchanged": unchanged,
                "failed": len(failures),
                "failures": [
                    {
                        "path": failure.relative_path,
                        "stage": failure.stage,
                        "error": failure.error,
                    }
                    for failure in failures
                ],
                "embedding_model": model_key or self.db.metadata().get("embedding_model"),
                "embedding_dim": embedding_dim or self.db.metadata().get("embedding_dim"),
                "filters": {"include": list(filters.include), "exclude": list(filters.exclude)},
                "db_path": str(self.config.index_db_path),
                "log_path": str(log_path),
            }
        finally:
            logger.close()

    def index_status(self) -> dict[str, object]:
        db_status = self.db.status()
        metadata = db_status["metadata"]
        lm_health = self.lm_studio.health()
        issues: list[str] = []
        if not Path(self.config.index_db_path).exists():
            issues.append("Index database does not exist yet.")
        if metadata.get("schema_version") != SCHEMA_VERSION:
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

    @staticmethod
    def _embedding_input(*, title: str, aliases: str, heading_path: str, tags: str, text: str) -> str:
        preface_parts = [f"title: {title}"]
        if aliases:
            preface_parts.append(f"aliases: {aliases}")
        if heading_path:
            preface_parts.append(f"heading: {heading_path}")
        if tags:
            preface_parts.append(f"tags: {tags}")
        preface = "\n".join(preface_parts)
        return f"{preface}\n\n{text}".strip()

    def _build_log_path(self, timestamp: str) -> Path:
        safe_timestamp = timestamp.replace(":", "-")
        return self.config.project_root / ".obsidian-rag" / "runs" / f"{safe_timestamp}.jsonl"


def time_now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000_000)
