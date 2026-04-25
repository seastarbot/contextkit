"""MCP Server for ContextKit.

Exposes ContextKit capabilities as MCP tools, enabling any AI agent
(Claude Desktop, Cursor, Windsurf) to manage context through the
Model Context Protocol.

Usage:
    # Run directly
    python -m contextkit.mcp_server

    # Or via installed CLI
    contextkit mcp
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from contextkit.core import ContextManager

# Default storage location
DEFAULT_STORAGE = os.environ.get(
    "CONTEXTKIT_STORAGE",
    str(Path.home() / ".contextkit" / "mcp_store"),
)

# Global context manager instance (initialized per-session)
_ctx: ContextManager | None = None


def _get_ctx(
    storage: str | None = None,
    max_tokens: int = 200000,
) -> ContextManager:
    """Get or create the global ContextManager instance."""
    global _ctx
    store = storage or DEFAULT_STORAGE
    if _ctx is None or str(_ctx.storage_dir) != store:
        _ctx = ContextManager(
            storage=store,
            max_tokens=max_tokens,
            compress_ratio=0.3,
            embedding_model=None,  # No embedding by default in MCP mode
            compression_model="gpt-4o-mini",
        )
    return _ctx


# Create the MCP server
mcp = FastMCP(
    "ContextKit",
    description="The missing context layer for AI agents — smart compression, semantic search, and token budgeting.",
)


@mcp.tool()
def ctx_add(
    role: str,
    content: str,
    storage: str = "",
    metadata_json: str = "{}",
) -> str:
    """Add a message to the context store.

    Args:
        role: Message role — "user", "assistant", or "system".
        content: The message text content.
        storage: Optional custom storage directory path.
        metadata_json: Optional JSON string with metadata key-value pairs.

    Returns:
        A confirmation message with the message ID and current token usage.
    """
    ctx = _get_ctx(storage=storage or None)
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except json.JSONDecodeError:
        metadata = {}

    msg_id = ctx.add(role=role, content=content, metadata=metadata)
    budget = ctx.token_budget

    return json.dumps(
        {
            "status": "ok",
            "message_id": msg_id,
            "token_usage": {
                "used": budget["used"],
                "total": budget["total"],
                "utilization": budget["utilization"],
            },
            "total_messages": len(ctx.messages),
        },
        indent=2,
    )


@mcp.tool()
def ctx_search(
    query: str,
    max_tokens: int = 50000,
    max_results: int = 10,
    storage: str = "",
) -> str:
    """Search context semantically to find relevant past messages.

    Args:
        query: Natural language query to search for.
        max_tokens: Maximum total tokens in returned results.
        max_results: Maximum number of messages to return.
        storage: Optional custom storage directory path.

    Returns:
        JSON with matching messages ranked by relevance.
    """
    ctx = _get_ctx(storage=storage or None)
    results = ctx.get_relevant(query=query, max_tokens=max_tokens)

    output = []
    for msg in results[:max_results]:
        output.append(
            {
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", "")[:2000],
                "relevance_score": msg.get("relevance_score", 0),
                "timestamp": msg.get("timestamp", 0),
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "query": query,
            "results_count": len(output),
            "results": output,
        },
        indent=2,
    )


@mcp.tool()
def ctx_compress(
    hours: int = 1,
    storage: str = "",
) -> str:
    """Compress old messages into summaries to save token budget.

    Summarizes messages older than the specified number of hours,
    dramatically reducing token usage while preserving key information.

    Args:
        hours: Only compress messages older than this many hours. Use 0 for aggressive compression.
        storage: Optional custom storage directory path.

    Returns:
        A summary of the compression operation.
    """
    ctx = _get_ctx(storage=storage or None)

    before_budget = ctx.token_budget
    compressed_count = ctx.summarize_older_than(hours=hours)
    after_budget = ctx.token_budget

    return json.dumps(
        {
            "status": "ok",
            "messages_compressed": compressed_count,
            "before": {
                "used": before_budget["used"],
                "utilization": before_budget["utilization"],
            },
            "after": {
                "used": after_budget["used"],
                "utilization": after_budget["utilization"],
            },
        },
        indent=2,
    )


@mcp.tool()
def ctx_stats(
    storage: str = "",
) -> str:
    """Get current context store statistics.

    Returns token usage, message count, budget status, and storage info.

    Args:
        storage: Optional custom storage directory path.

    Returns:
        JSON with detailed context statistics.
    """
    ctx = _get_ctx(storage=storage or None)
    budget = ctx.token_budget

    # Check if storage exists and its size
    storage_path = Path(str(ctx.storage_dir))
    storage_size = 0
    if storage_path.exists():
        for f in storage_path.rglob("*"):
            if f.is_file():
                storage_size += f.stat().st_size

    # Count message types
    role_counts: dict[str, int] = {}
    for msg in ctx.messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    # Check for summaries
    summary_count = sum(
        1 for m in ctx.messages
        if m.get("metadata", {}).get("type") == "summary"
    )

    return json.dumps(
        {
            "status": "ok",
            "token_budget": budget,
            "message_count": len(ctx.messages),
            "role_distribution": role_counts,
            "summary_messages": summary_count,
            "storage_path": str(ctx.storage_dir),
            "storage_size_bytes": storage_size,
        },
        indent=2,
    )


@mcp.tool()
def ctx_export(
    output_path: str,
    storage: str = "",
) -> str:
    """Export the entire context store to a JSON file.

    Args:
        output_path: File path for the exported JSON.
        storage: Optional custom storage directory path.

    Returns:
        A confirmation with export details.
    """
    ctx = _get_ctx(storage=storage or None)
    ctx.export(output_path)

    file_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0

    return json.dumps(
        {
            "status": "ok",
            "output_path": output_path,
            "messages_exported": len(ctx.messages),
            "file_size_bytes": file_size,
        },
        indent=2,
    )


@mcp.tool()
def ctx_import(
    input_path: str,
    storage: str = "",
) -> str:
    """Import messages from a JSON export file into the context store.

    Args:
        input_path: Path to the JSON file to import.
        storage: Optional custom storage directory path.

    Returns:
        A confirmation with import details.
    """
    ctx = _get_ctx(storage=storage or None)
    count = ctx.import_(input_path)

    return json.dumps(
        {
            "status": "ok",
            "messages_imported": count,
            "total_messages": len(ctx.messages),
        },
        indent=2,
    )


@mcp.tool()
def ctx_list(
    limit: int = 20,
    offset: int = 0,
    storage: str = "",
) -> str:
    """List messages in the context store with pagination.

    Args:
        limit: Maximum number of messages to return (default 20).
        offset: Number of messages to skip (for pagination).
        storage: Optional custom storage directory path.

    Returns:
        JSON with a paginated list of messages.
    """
    ctx = _get_ctx(storage=storage or None)
    total = len(ctx.messages)
    messages = ctx.messages[offset : offset + limit]

    output = []
    for msg in messages:
        output.append(
            {
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", "")[:1000],
                "timestamp": msg.get("timestamp", 0),
                "is_summary": msg.get("metadata", {}).get("type") == "summary",
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "total": total,
            "offset": offset,
            "limit": limit,
            "returned": len(output),
            "messages": output,
        },
        indent=2,
    )


@mcp.tool()
def ctx_clear(storage: str = "") -> str:
    """Clear all messages from the context store.

    This is irreversible. Use with caution.

    Args:
        storage: Optional custom storage directory path.

    Returns:
        A confirmation message.
    """
    ctx = _get_ctx(storage=storage or None)
    count = len(ctx.messages)
    ctx._messages.clear()
    if ctx._index:
        ctx._index.clear()
    ctx._save()

    return json.dumps(
        {
            "status": "ok",
            "messages_cleared": count,
        },
        indent=2,
    )


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
