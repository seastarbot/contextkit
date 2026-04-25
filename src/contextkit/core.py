"""Core context management engine.

Orchestrates compression, vector indexing, and token budget management
to provide intelligent context retrieval for AI agents.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from contextkit.budget import TokenBudget
from contextkit.compressor import ContextCompressor
from contextkit.indexer import VectorIndex


class ContextManager:
    """The main context management engine.

    Combines intelligent compression, vector indexing, and token budget
    management to optimize context windows for AI agents.

    Example:
        >>> ctx = ContextManager(max_tokens=128000)
        >>> ctx.add("user", "How do I sort a list in Python?")
        >>> ctx.add("assistant", "You can use sorted() or list.sort().")
        >>> ctx.get_relevant("sorting")
        [{'role': 'user', 'content': 'How do I sort a list in Python?', ...}]
    """

    def __init__(
        self,
        storage: str = "./.contextkit",
        max_tokens: int = 200000,
        compress_ratio: float = 0.3,
        embedding_model: str = "text-embedding-3-small",
        compression_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the context manager.

        Args:
            storage: Directory for persistent storage (vectors, messages).
            max_tokens: Maximum token budget for the context window.
            compress_ratio: Trigger compression when utilization exceeds
                          (1 - compress_ratio) of the budget. E.g., 0.3 means
                          compress when 70% full.
            embedding_model: OpenAI embedding model name. Set to None to
                           disable vector indexing.
            compression_model: LLM model for summarization during compression.
        """
        self.storage_dir = Path(storage)
        self.max_tokens = max_tokens
        self.compress_ratio = compress_ratio

        # Initialize sub-modules
        self._budget = TokenBudget(max_tokens=max_tokens)
        self._messages: list[dict[str, Any]] = []
        self._embedding_model = embedding_model

        # Vector index (optional)
        self._index: VectorIndex | None = None
        if embedding_model:
            self._index = VectorIndex(
                storage_dir=str(self.storage_dir),
                embedding_model=embedding_model,
            )

        # Compressor
        self._compressor = ContextCompressor(model=compression_model)

        # Try to load existing state
        self._load()

    # ─── Public API ──────────────────────────────────────────────

    def add(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a message to the context.

        Args:
            role: Message role ("user", "assistant", "system").
            content: Message content text.
            metadata: Optional metadata dict attached to the message.

        Returns:
            The unique message ID.
        """
        msg_id = str(uuid.uuid4())[:12]
        message = {
            "id": msg_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._messages.append(message)

        # Add to vector index
        if self._index and content.strip():
            try:
                self._index.add(msg_id, content)
            except Exception:
                # Don't fail if embedding fails — index is best-effort
                pass

        # Auto-save periodically (every 20 messages)
        if len(self._messages) % 20 == 0:
            self._save()

        return msg_id

    def get_relevant(
        self,
        query: str,
        max_tokens: int = 50000,
    ) -> list[dict[str, Any]]:
        """Retrieve context relevant to a query via semantic search.

        Combines vector-similarity search with recency weighting to return
        the most useful context within the token budget.

        Args:
            query: The query text to find relevant context for.
            max_tokens: Maximum total tokens for returned context.

        Returns:
            List of message dicts, sorted by relevance.
        """
        results: list[dict[str, Any]] = []

        # Semantic search via vector index
        if self._index and self._index.size > 0:
            try:
                search_results = self._index.search(query, top_k=max_tokens // 100 + 20)
                id_to_msg = {m["id"]: m for m in self._messages}

                for msg_id, score in search_results:
                    if msg_id in id_to_msg:
                        msg = id_to_msg[msg_id].copy()
                        msg["relevance_score"] = score
                        results.append(msg)
            except Exception:
                # Fall back to recent messages
                pass

        # Add recent messages (last 10) to ensure continuity
        recent = self._messages[-10:] if self._messages else []
        recent_ids = {m["id"] for m in results}
        for msg in recent:
            if msg["id"] not in recent_ids:
                msg_copy = msg.copy()
                msg_copy["relevance_score"] = 0.5  # Moderate relevance
                results.append(msg_copy)
                recent_ids.add(msg["id"])

        # Sort by relevance score (descending)
        results.sort(key=lambda m: m.get("relevance_score", 0), reverse=True)

        # Trim to token budget
        budget = TokenBudget(max_tokens=max_tokens)
        selected: list[dict[str, Any]] = []
        used = 0
        for msg in results:
            msg_tokens = budget.count_message_tokens(msg["role"], msg["content"])
            if used + msg_tokens > max_tokens:
                break
            selected.append(msg)
            used += msg_tokens

        return selected

    def get_recent(self, max_tokens: int = 50000) -> list[dict[str, Any]]:
        """Get the most recent messages within a token budget.

        Args:
            max_tokens: Maximum total tokens for returned context.

        Returns:
            List of message dicts, most recent last.
        """
        budget = TokenBudget(max_tokens=max_tokens)
        selected: list[dict[str, Any]] = []
        used = 0

        # Walk backwards from the most recent message
        for msg in reversed(self._messages):
            msg_tokens = budget.count_message_tokens(msg["role"], msg["content"])
            if used + msg_tokens > max_tokens:
                break
            selected.append(msg)
            used += msg_tokens

        # Reverse to chronological order
        selected.reverse()
        return selected

    def summarize_older_than(self, hours: int = 2) -> int:
        """Compress messages older than a specified time threshold.

        Messages older than `hours` hours are summarized into a single
        system message, dramatically reducing token usage.

        Args:
            hours: Only compress messages older than this many hours.

        Returns:
            Number of messages that were compressed.
        """
        cutoff = time.time() - (hours * 3600)
        old_messages: list[dict[str, Any]] = []
        old_indices: list[int] = []

        for i, msg in enumerate(self._messages):
            if msg.get("timestamp", 0) < cutoff:
                # Don't re-compress existing summaries
                meta = msg.get("metadata", {})
                if meta.get("type") == "summary":
                    continue
                old_messages.append(msg)
                old_indices.append(i)

        if len(old_messages) < 3:
            # Not enough old messages to warrant compression
            return 0

        # Generate summary
        summary = self._compressor.summarize(old_messages)

        # Create summary message
        summary_msg = self._compressor.create_summary_message(
            summary=summary,
            original_count=len(old_messages),
            start_time=old_messages[0].get("timestamp"),
        )
        summary_msg["id"] = str(uuid.uuid4())[:12]

        # Remove old messages from vector index
        if self._index:
            old_ids = [m["id"] for m in old_messages]
            self._index.remove_batch(old_ids)

        # Replace old messages with summary (in reverse index order to maintain positions)
        for idx in reversed(old_indices):
            self._messages.pop(idx)

        # Insert summary at the position of the first removed message
        insert_pos = old_indices[0] if old_indices else 0
        self._messages.insert(insert_pos, summary_msg)

        # Add summary to index
        if self._index:
            try:
                self._index.add(summary_msg["id"], summary)
            except Exception:
                pass

        # Save after compression
        self._save()

        return len(old_messages)

    def auto_compress(self) -> int:
        """Automatically compress context when budget is nearly full.

        Triggers compression when token utilization exceeds
        (1 - compress_ratio) of the maximum budget.

        Returns:
            Number of messages compressed, or 0 if no compression was needed.
        """
        utilization = self.token_count / self.max_tokens if self.max_tokens > 0 else 0
        threshold = 1.0 - self.compress_ratio

        if utilization < threshold:
            return 0

        # Try time-based compression first (compress older messages)
        compressed = self.summarize_older_than(hours=1)

        # If still over threshold, compress more aggressively
        new_utilization = self.token_count / self.max_tokens if self.max_tokens > 0 else 0
        if new_utilization >= threshold:
            compressed += self.summarize_older_than(hours=0)

        return compressed

    def export(self, path: str) -> None:
        """Export all messages and metadata to a JSON file.

        Args:
            path: Output file path.
        """
        output = {
            "version": "0.1.0",
            "exported_at": time.time(),
            "config": {
                "max_tokens": self.max_tokens,
                "compress_ratio": self.compress_ratio,
                "embedding_model": self._embedding_model,
            },
            "messages": self._messages,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    def import_(self, path: str) -> int:
        """Import messages from a JSON export file.

        Args:
            path: Path to the exported JSON file.

        Returns:
            Number of messages imported.
        """
        data = json.loads(Path(path).read_text())
        imported_messages = data.get("messages", [])

        existing_ids = {m["id"] for m in self._messages}
        count = 0

        for msg in imported_messages:
            if msg["id"] not in existing_ids:
                self._messages.append(msg)

                # Add to vector index
                if self._index and msg.get("content", "").strip():
                    try:
                        self._index.add(msg["id"], msg["content"])
                    except Exception:
                        pass

                count += 1

        # Sort by timestamp
        self._messages.sort(key=lambda m: m.get("timestamp", 0))

        self._save()
        return count

    @property
    def token_count(self) -> int:
        """Total tokens used by all messages."""
        total = 0
        for msg in self._messages:
            total += self._budget.count_message_tokens(msg["role"], msg["content"])
        return total

    @property
    def token_budget(self) -> dict[str, object]:
        """Current budget utilization status.

        Returns:
            Dictionary with total, used, remaining, utilization, and message count.
        """
        used = self.token_count
        status = self._budget.budget_status(used)
        status["messages"] = len(self._messages)
        status["tokens_per_message_avg"] = (
            used // len(self._messages) if self._messages else 0
        )
        return status

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Access to the raw message list."""
        return self._messages

    # ─── Internal ────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist messages and index to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Save messages
        messages_file = self.storage_dir / "messages.json"
        messages_file.write_text(
            json.dumps(self._messages, indent=2, ensure_ascii=False)
        )

        # Save vector index
        if self._index:
            try:
                self._index.save()
            except Exception:
                pass

        # Save config
        config_file = self.storage_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "max_tokens": self.max_tokens,
                    "compress_ratio": self.compress_ratio,
                    "embedding_model": self._embedding_model,
                    "saved_at": time.time(),
                },
                indent=2,
            )
        )

    def _load(self) -> None:
        """Load persisted state from disk."""
        messages_file = self.storage_dir / "messages.json"

        if messages_file.exists():
            try:
                self._messages = json.loads(messages_file.read_text())
            except Exception:
                self._messages = []

        # Load vector index
        if self._index:
            try:
                self._index.load()
            except Exception:
                pass

    def __len__(self) -> int:
        """Number of messages in context."""
        return len(self._messages)

    def __repr__(self) -> str:
        budget = self.token_budget
        return (
            f"ContextManager(messages={len(self._messages)}, "
            f"tokens={budget['used']}/{budget['total']}, "
            f"utilization={budget['utilization']})"
        )
