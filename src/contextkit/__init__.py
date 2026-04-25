"""ContextKit — The missing context layer for AI agents.

Smart compression, vector indexing, semantic search, and token budgeting
for AI agent context management. Ships as a Python library, CLI, and MCP server.

Usage:
    # As a library
    from contextkit import ContextManager

    # As a CLI
    contextkit stats <file>
    contextkit compress <file>
    contextkit search <file> <query>

    # As an MCP server
    contextkit mcp
"""

from contextkit.core import ContextManager

__version__ = "0.2.0"
__all__ = ["ContextManager"]
