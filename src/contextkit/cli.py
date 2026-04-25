"""ContextKit CLI — Command-line interface for context management.

Provides utilities for inspecting, compressing, searching, and
benchmarking AI agent contexts.

Usage:
    contextkit stats <file>
    contextkit compress <file> [--output <output>]
    contextkit search <file> <query>
    contextkit export <file> <output>
    contextkit bench
    contextkit mcp
    contextkit version
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _load_messages(path: str) -> list[dict[str, Any]]:
    """Load messages from a contextkit storage directory or JSON file."""
    p = Path(path)

    # If it's a JSON file, load directly
    if p.is_file() and p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "messages" in data:
            return data["messages"]
        elif isinstance(data, list):
            return data
        else:
            print(f"Error: Unrecognized JSON format in {path}", file=sys.stderr)
            sys.exit(1)

    # If it's a directory, look for messages.json
    if p.is_dir():
        messages_file = p / "messages.json"
        if messages_file.exists():
            return json.loads(messages_file.read_text())
        else:
            print(f"Error: No messages.json found in {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Error: {path} is not a valid file or directory", file=sys.stderr)
    sys.exit(1)


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: rough estimate (~4 chars per token)
        return len(text) // 4


def cmd_stats(args: argparse.Namespace) -> None:
    """Display context statistics."""
    messages = _load_messages(args.file)

    total_chars = 0
    total_tokens = 0
    role_counts: dict[str, int] = {}
    summary_count = 0
    oldest_ts = float("inf")
    newest_ts = 0.0

    for msg in messages:
        content = msg.get("content", "")
        total_chars += len(content)
        total_tokens += _count_tokens(content)

        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

        if msg.get("metadata", {}).get("type") == "summary":
            summary_count += 1

        ts = msg.get("timestamp", 0)
        if ts > 0:
            oldest_ts = min(oldest_ts, ts)
            newest_ts = max(newest_ts, ts)

    print(f"\n{'='*50}")
    print(f"  ContextKit Stats: {args.file}")
    print(f"{'='*50}")
    print(f"  Messages:        {len(messages)}")
    print(f"  Characters:      {total_chars:,}")
    print(f"  Est. Tokens:     {total_tokens:,}")
    print(f"  Avg Tokens/M msg: {total_tokens // len(messages) if messages else 0:,}")
    print()

    if role_counts:
        print("  Role Distribution:")
        for role, count in sorted(role_counts.items()):
            print(f"    {role:15s} {count:>6}")
    print()

    if summary_count > 0:
        print(f"  Compressed:      {summary_count} summary messages")

    if oldest_ts < float("inf"):
        from datetime import datetime, timezone
        oldest = datetime.fromtimestamp(oldest_ts, tz=timezone.utc)
        newest = datetime.fromtimestamp(newest_ts, tz=timezone.utc)
        print(f"  Time Range:      {oldest.strftime('%Y-%m-%d %H:%M')} → {newest.strftime('%Y-%m-%d %H:%M')}")

    print(f"{'='*50}\n")


def cmd_compress(args: argparse.Namespace) -> None:
    """Compress context by summarizing old messages."""
    from contextkit.compressor import ContextCompressor

    messages = _load_messages(args.file)
    compressor = ContextCompressor(model=args.model or "gpt-4o-mini")

    # Separate old and new messages
    cutoff = time.time() - (args.hours * 3600)
    old_messages = [m for m in messages if m.get("timestamp", 0) < cutoff
                    and m.get("metadata", {}).get("type") != "summary"]
    new_messages = [m for m in messages if m.get("timestamp", 0) >= cutoff
                    or m.get("metadata", {}).get("type") == "summary"]

    if len(old_messages) < 3:
        print(f"Only {len(old_messages)} old messages found. Nothing to compress.")
        return

    print(f"Compressing {len(old_messages)} messages older than {args.hours}h...")

    # Token count before
    before_tokens = sum(_count_tokens(m.get("content", "")) for m in messages)

    # Generate summary
    summary = compressor.summarize(old_messages)
    summary_msg = compressor.create_summary_message(
        summary=summary,
        original_count=len(old_messages),
        start_time=old_messages[0].get("timestamp"),
    )

    # Rebuild: summary + new messages
    compressed = [summary_msg] + new_messages
    compressed.sort(key=lambda m: m.get("timestamp", 0))

    after_tokens = sum(_count_tokens(m.get("content", "")) for m in compressed)

    # Write output
    output_path = args.output or args.file
    output_p = Path(output_path)

    if output_p.is_dir():
        output_p = output_p / "messages.json"

    output_p.parent.mkdir(parents=True, exist_ok=True)
    output_p.write_text(json.dumps(compressed, indent=2, ensure_ascii=False))

    saved_pct = ((before_tokens - after_tokens) / before_tokens * 100) if before_tokens > 0 else 0

    print(f"\n  Before:    {before_tokens:,} tokens ({len(messages)} messages)")
    print(f"  After:     {after_tokens:,} tokens ({len(compressed)} messages)")
    print(f"  Saved:     {saved_pct:.1f}% ({before_tokens - after_tokens:,} tokens)")
    print(f"  Output:    {output_path}\n")


def cmd_search(args: argparse.Namespace) -> None:
    """Search context semantically."""
    messages = _load_messages(args.file)
    query = args.query

    print(f"\nSearching for: \"{query}\"")
    print(f"{'-'*50}")

    # Simple TF-IDF-like keyword search (no API needed)
    query_words = set(query.lower().split())

    scored: list[tuple[float, dict]] = []
    for msg in messages:
        content = msg.get("content", "")
        if not content.strip():
            continue
        content_lower = content.lower()
        # Score: word overlap + recency
        words = set(content_lower.split())
        overlap = len(query_words & words)
        tf_score = overlap / len(query_words) if query_words else 0

        # Recency boost
        ts = msg.get("timestamp", 0)
        recency = 1.0 if ts > time.time() - 3600 else 0.5

        total_score = tf_score * recency
        if total_score > 0:
            scored.append((total_score, msg))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = args.limit or 5

    if not scored:
        print("  No matching messages found.\n")
        return

    for i, (score, msg) in enumerate(scored[:top_k]):
        role = msg.get("role", "?")
        content = msg.get("content", "")[:200]
        print(f"  [{i+1}] score={score:.3f} | [{role}]")
        print(f"      {content}{'...' if len(msg.get('content', '')) > 200 else ''}")
        print()

    print(f"  Found {min(len(scored), top_k)} results (total scored: {len(scored)})\n")


def cmd_export(args: argparse.Namespace) -> None:
    """Export context to a file."""
    messages = _load_messages(args.file)

    output = {
        "version": "0.2.0",
        "exported_at": time.time(),
        "source": args.file,
        "messages": messages,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    size = output_path.stat().st_size
    print(f"\n  Exported {len(messages)} messages to {args.output} ({size:,} bytes)\n")


def cmd_import(args: argparse.Namespace) -> None:
    """Import messages from a JSON export file into a context store."""
    import_path = Path(args.file)
    if not import_path.exists():
        print(f"Error: {args.file} does not exist", file=sys.stderr)
        sys.exit(1)

    # Parse the export file
    data = json.loads(import_path.read_text())
    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    elif isinstance(data, list):
        messages = data
    else:
        print("Error: Unrecognized JSON format", file=sys.stderr)
        sys.exit(1)

    # Determine target storage
    storage_dir = Path(args.storage) if args.storage else Path(".contextkit")
    storage_dir.mkdir(parents=True, exist_ok=True)
    messages_file = storage_dir / "messages.json"

    # Merge: load existing, skip duplicates
    existing: list[dict] = []
    if messages_file.exists():
        try:
            existing = json.loads(messages_file.read_text())
        except Exception:
            existing = []

    existing_ids = {m.get("id") for m in existing}
    imported_count = 0
    for msg in messages:
        if msg.get("id") not in existing_ids:
            existing.append(msg)
            existing_ids.add(msg.get("id"))
            imported_count += 1

    # Sort by timestamp and write
    existing.sort(key=lambda m: m.get("timestamp", 0))
    messages_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    total_tokens = sum(_count_tokens(m.get("content", "")) for m in existing)
    print(f"\n  Imported {imported_count} messages from {args.file}")
    print(f"  Storage:  {storage_dir}")
    print(f"  Total:    {len(existing)} messages, ~{total_tokens:,} tokens\n")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run the benchmark suite."""
    import importlib.util

    # Try installed package path first, then source tree
    try:
        from benchmarks.benchmark import run_benchmarks  # type: ignore[import]

        run_benchmarks()
        return
    except ImportError:
        pass

    # Locate benchmark.py relative to this file (works in source tree)
    bench_file = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "benchmark.py"
    if bench_file.exists():
        spec = importlib.util.spec_from_file_location("benchmark", bench_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            mod.run_benchmarks()
            return

    print("Error: benchmarks/benchmark.py not found. Run from the project root.", file=sys.stderr)
    sys.exit(1)


def cmd_mcp(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    from contextkit.mcp_server import main
    main()


def cmd_version(args: argparse.Namespace) -> None:
    """Print version info."""
    from contextkit import __version__
    print(f"ContextKit v{__version__}")
    print(f"Python {sys.version.split()[0]}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="contextkit",
        description="ContextKit — The missing context layer for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show context statistics")
    p_stats.add_argument("file", help="Context file or storage directory")
    p_stats.set_defaults(func=cmd_stats)

    # compress
    p_compress = subparsers.add_parser("compress", help="Compress old messages")
    p_compress.add_argument("file", help="Context file or storage directory")
    p_compress.add_argument("-o", "--output", help="Output path (default: overwrite)")
    p_compress.add_argument("--hours", type=int, default=2, help="Compress messages older than N hours (default: 2)")
    p_compress.add_argument("--model", default="gpt-4o-mini", help="LLM model for summarization")
    p_compress.set_defaults(func=cmd_compress)

    # search
    p_search = subparsers.add_parser("search", help="Search context")
    p_search.add_argument("file", help="Context file or storage directory")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", "--limit", type=int, default=5, help="Max results")
    p_search.set_defaults(func=cmd_search)

    # export
    p_export = subparsers.add_parser("export", help="Export context to JSON")
    p_export.add_argument("file", help="Source context file or directory")
    p_export.add_argument("output", help="Output file path")
    p_export.set_defaults(func=cmd_export)

    # import
    p_import = subparsers.add_parser("import", help="Import context from a JSON export file")
    p_import.add_argument("file", help="JSON export file to import")
    p_import.add_argument(
        "--storage",
        default="",
        help="Target storage directory (default: .contextkit)",
    )
    p_import.set_defaults(func=cmd_import)

    # bench
    p_bench = subparsers.add_parser("bench", help="Run benchmarks")
    p_bench.set_defaults(func=cmd_bench)

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Start MCP server (stdio)")
    p_mcp.set_defaults(func=cmd_mcp)

    # version
    p_version = subparsers.add_parser("version", help="Show version")
    p_version.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
