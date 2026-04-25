"""MCP Server for ContextKit.

Exposes ContextKit capabilities as MCP tools, enabling any AI agent
(Claude Desktop, Cursor, Windsurf) to manage context through the
Model Context Protocol.

Uses the mcp.server.Server low-level API for full protocol control.

Usage:
    # Run directly
    python -m contextkit.mcp_server

    # Or via installed CLI
    contextkit mcp
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from contextkit.core import ContextManager

# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

DEFAULT_STORAGE = os.environ.get(
    "CONTEXTKIT_STORAGE",
    str(Path.home() / ".contextkit" / "mcp_store"),
)

_ctx: ContextManager | None = None


def _get_ctx(storage: str | None = None, max_tokens: int = 200_000) -> ContextManager:
    """Return (or create) the global ContextManager instance."""
    global _ctx
    store = storage or DEFAULT_STORAGE
    if _ctx is None or str(_ctx.storage_dir) != store:
        _ctx = ContextManager(
            storage=store,
            max_tokens=max_tokens,
            compress_ratio=0.3,
            embedding_model=None,           # no embeddings by default in MCP mode
            compression_model="gpt-4o-mini",
        )
    return _ctx


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="ctx_add",
        description=(
            "Add a message to the ContextKit store. "
            "Persists to disk automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": 'Message role: "user", "assistant", or "system".',
                },
                "content": {
                    "type": "string",
                    "description": "The message text content.",
                },
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path (overrides default).",
                    "default": "",
                },
                "metadata_json": {
                    "type": "string",
                    "description": "Optional JSON string of metadata key-value pairs.",
                    "default": "{}",
                },
            },
            "required": ["role", "content"],
        },
    ),
    Tool(
        name="ctx_search",
        description=(
            "Search the context store for messages relevant to a query. "
            "Uses semantic search when embeddings are available, "
            "otherwise falls back to keyword matching."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search for.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum total tokens in returned results.",
                    "default": 50000,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of messages to return.",
                    "default": 10,
                },
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path.",
                    "default": "",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="ctx_compress",
        description=(
            "Compress old messages into a summary to reclaim token budget. "
            "Messages older than `hours` are summarized into one system message."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": (
                        "Compress messages older than this many hours. "
                        "Use 0 for aggressive compression."
                    ),
                    "default": 1,
                },
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path.",
                    "default": "",
                },
            },
        },
    ),
    Tool(
        name="ctx_stats",
        description=(
            "Get statistics about the context store: token usage, "
            "message counts, role distribution, and storage size."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path.",
                    "default": "",
                },
            },
        },
    ),
    Tool(
        name="ctx_export",
        description="Export the entire context store to a JSON file.",
        inputSchema={
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Destination file path for the exported JSON.",
                },
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path.",
                    "default": "",
                },
            },
            "required": ["output_path"],
        },
    ),
    Tool(
        name="ctx_import",
        description=(
            "Import messages from a previously exported JSON file "
            "into the context store. Skips duplicate message IDs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the JSON export file to import.",
                },
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path.",
                    "default": "",
                },
            },
            "required": ["input_path"],
        },
    ),
    Tool(
        name="ctx_list",
        description="List messages in the context store with pagination.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return.",
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of messages to skip (for pagination).",
                    "default": 0,
                },
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path.",
                    "default": "",
                },
            },
        },
    ),
    Tool(
        name="ctx_clear",
        description=(
            "Clear ALL messages from the context store. "
            "This is irreversible — use with caution."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "storage": {
                    "type": "string",
                    "description": "Optional storage directory path.",
                    "default": "",
                },
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool implementations (synchronous helpers)
# ---------------------------------------------------------------------------

def _tool_ctx_add(args: dict[str, Any]) -> str:
    role = str(args.get("role", "user"))
    content = str(args.get("content", ""))
    storage = str(args.get("storage", "")) or None
    metadata_json = str(args.get("metadata_json", "{}"))

    ctx = _get_ctx(storage=storage)

    try:
        metadata = json.loads(metadata_json) if metadata_json.strip() else {}
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


def _tool_ctx_search(args: dict[str, Any]) -> str:
    query = str(args.get("query", ""))
    max_tokens = int(args.get("max_tokens", 50000))
    max_results = int(args.get("max_results", 10))
    storage = str(args.get("storage", "")) or None

    ctx = _get_ctx(storage=storage)
    results = ctx.get_relevant(query=query, max_tokens=max_tokens)

    output = []
    for msg in results[:max_results]:
        output.append(
            {
                "id": msg.get("id", ""),
                "role": msg.get("role", ""),
                "content": msg.get("content", "")[:2000],
                "relevance_score": round(float(msg.get("relevance_score", 0)), 4),
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


def _tool_ctx_compress(args: dict[str, Any]) -> str:
    hours = int(args.get("hours", 1))
    storage = str(args.get("storage", "")) or None

    ctx = _get_ctx(storage=storage)
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
            "tokens_saved": int(before_budget["used"]) - int(after_budget["used"]),
        },
        indent=2,
    )


def _tool_ctx_stats(args: dict[str, Any]) -> str:
    storage = str(args.get("storage", "")) or None
    ctx = _get_ctx(storage=storage)
    budget = ctx.token_budget

    storage_path = Path(str(ctx.storage_dir))
    storage_size = 0
    if storage_path.exists():
        for f in storage_path.rglob("*"):
            if f.is_file():
                storage_size += f.stat().st_size

    role_counts: dict[str, int] = {}
    for msg in ctx.messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    summary_count = sum(
        1
        for m in ctx.messages
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


def _tool_ctx_export(args: dict[str, Any]) -> str:
    output_path = str(args.get("output_path", ""))
    storage = str(args.get("storage", "")) or None

    if not output_path:
        return json.dumps({"status": "error", "message": "output_path is required"})

    ctx = _get_ctx(storage=storage)
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


def _tool_ctx_import(args: dict[str, Any]) -> str:
    input_path = str(args.get("input_path", ""))
    storage = str(args.get("storage", "")) or None

    if not input_path:
        return json.dumps({"status": "error", "message": "input_path is required"})

    if not Path(input_path).exists():
        return json.dumps(
            {"status": "error", "message": f"File not found: {input_path}"}
        )

    ctx = _get_ctx(storage=storage)
    count = ctx.import_(input_path)

    return json.dumps(
        {
            "status": "ok",
            "messages_imported": count,
            "total_messages": len(ctx.messages),
        },
        indent=2,
    )


def _tool_ctx_list(args: dict[str, Any]) -> str:
    limit = int(args.get("limit", 20))
    offset = int(args.get("offset", 0))
    storage = str(args.get("storage", "")) or None

    ctx = _get_ctx(storage=storage)
    total = len(ctx.messages)
    page = ctx.messages[offset: offset + limit]

    output = []
    for msg in page:
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


def _tool_ctx_clear(args: dict[str, Any]) -> str:
    storage = str(args.get("storage", "")) or None
    ctx = _get_ctx(storage=storage)
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


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

server = Server("contextkit")

_TOOL_HANDLERS: dict[str, Any] = {
    "ctx_add": _tool_ctx_add,
    "ctx_search": _tool_ctx_search,
    "ctx_compress": _tool_ctx_compress,
    "ctx_stats": _tool_ctx_stats,
    "ctx_export": _tool_ctx_export,
    "ctx_import": _tool_ctx_import,
    "ctx_list": _tool_ctx_list,
    "ctx_clear": _tool_ctx_clear,
}


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Return all available ContextKit tools."""
    return TOOLS


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict[str, Any] | None,
) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"status": "error", "message": f"Unknown tool: {name}"}
                ),
            )
        ]

    args = arguments or {}
    try:
        result = handler(args)
    except Exception as exc:
        result = json.dumps({"status": "error", "message": str(exc)})

    return [TextContent(type="text", text=result)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="contextkit",
                server_version="0.2.0",
            ),
        )


def main() -> None:
    """Synchronous entry point — called by the CLI and __main__."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
