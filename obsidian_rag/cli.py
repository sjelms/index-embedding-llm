from __future__ import annotations

import argparse
import json
import sys

from .service import FilterSpec, VaultService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and search an Obsidian vault for local RAG workflows.")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Manage the local vault index")
    index_subparsers = index_parser.add_subparsers(dest="index_command")

    build_parser_ = index_subparsers.add_parser("build", help="Create or rebuild the index")
    build_parser_.add_argument("--full", action="store_true", help="Force a full rebuild")
    build_parser_.add_argument("--dry-run", action="store_true", help="Show what would change without modifying the index")
    build_parser_.add_argument("--include", action="append", default=[], help="Only include a vault directory prefix")
    build_parser_.add_argument("--exclude", action="append", default=[], help="Exclude a vault directory prefix")
    build_parser_.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    sync_parser = index_subparsers.add_parser("sync", help="Incrementally sync the index")
    sync_parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying the index")
    sync_parser.add_argument("--include", action="append", default=[], help="Only include a vault directory prefix")
    sync_parser.add_argument("--exclude", action="append", default=[], help="Exclude a vault directory prefix")
    sync_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    status_parser = index_subparsers.add_parser("status", help="Report index and LM Studio health")
    status_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    search_parser = subparsers.add_parser("search", help="Search the local index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=None, help="Number of file-level results to return")
    search_parser.add_argument("--mode", choices=("hybrid", "semantic"), default="hybrid", help="Search mode")
    search_parser.add_argument("--include", action="append", default=[], help="Only include a vault directory prefix")
    search_parser.add_argument("--exclude", action="append", default=[], help="Exclude a vault directory prefix")
    search_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return

    service = VaultService()
    try:
        if args.command == "index":
            _handle_index_command(service, args)
        elif args.command == "search":
            _handle_search_command(service, args)
        else:
            parser.error(f"Unknown command '{args.command}'")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        service.close()


def _handle_index_command(service: VaultService, args: argparse.Namespace) -> None:
    if args.index_command == "build":
        filters = FilterSpec.from_raw(include=args.include, exclude=args.exclude)
        result = service.build_index(
            full=True,
            dry_run=args.dry_run,
            filters=filters,
            progress_output=sys.stderr,
            show_progress=not args.json,
        )
        _emit(args.json, result)
        _exit_if_failures(result)
        return
    if args.index_command == "sync":
        filters = FilterSpec.from_raw(include=args.include, exclude=args.exclude)
        result = service.build_index(
            full=False,
            dry_run=args.dry_run,
            filters=filters,
            progress_output=sys.stderr,
            show_progress=not args.json,
        )
        _emit(args.json, result)
        _exit_if_failures(result)
        return
    if args.index_command == "status":
        result = service.index_status()
        _emit(args.json, result)
        return
    raise ValueError("Missing index subcommand. Use build, sync, or status.")


def _handle_search_command(service: VaultService, args: argparse.Namespace) -> None:
    filters = FilterSpec.from_raw(include=args.include, exclude=args.exclude)
    top_k = args.top_k or service.config.top_k_default
    if args.mode == "semantic":
        result = service.semantic_search(args.query, top_k=top_k, filters=filters)
    else:
        result = service.hybrid_search(args.query, top_k=top_k, filters=filters)
    _emit(args.json, result)


def _emit(as_json: bool, payload: dict[str, object]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"{key}:")
            print(json.dumps(value, indent=2, sort_keys=True))
        else:
            print(f"{key}: {value}")


def _exit_if_failures(result: dict[str, object]) -> None:
    failed = int(result.get("failed", 0) or 0)
    if failed <= 0:
        return
    log_path = result.get("log_path")
    message = f"Indexing completed with {failed} failed file(s)."
    if log_path:
        message += f" See {log_path} for details."
    raise SystemExit(message)
