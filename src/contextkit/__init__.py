"""ContextKit — The missing context layer for AI agents.

Smart compression, vector indexing, semantic search, and token budgeting
for AI agent context management. Ships as a Python library, CLI, and MCP server.

Usage:
    # As a library
    from contextkit import ContextManager, TokenBudget, ContextCompressor, VectorIndex

    # As a CLI
    contextkit stats <file>
    contextkit compress <file>
    contextkit search <file> <query>
    contextkit export <file> <output>
    contextkit import <file> [--storage <dir>]

    # As an MCP server
    contextkit mcp
"""

from contextkit.budget import TokenBudget
from contextkit.compressor import ContextCompressor
from contextkit.core import ContextManager
from contextkit.indexer import VectorIndex

__version__ = "0.2.0"
__all__ = [
    "ContextManager",
    "TokenBudget",
    "ContextCompressor",
    "VectorIndex",
]
