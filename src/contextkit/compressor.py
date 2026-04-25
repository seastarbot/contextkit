"""Intelligent context compression using LLM-powered summarization.

When context budgets run low, this module uses an LLM to generate concise
summaries of older conversation turns, preserving key information while
dramatically reducing token usage.
"""

from __future__ import annotations

import os
import time
from typing import Any

SUMMARIZATION_PROMPT = """You are a context compression engine for an AI assistant.
Summarize the following conversation segment into a concise paragraph that preserves:
1. Key decisions and conclusions
2. Important facts, code references, or technical details
3. User preferences or requirements mentioned
4. Any unresolved questions or ongoing tasks

Rules:
- Be extremely concise — aim for 20-30% of the original length
- Use bullet points for lists of facts
- Preserve technical terms, code snippets references, and names exactly
- Do NOT add your own commentary or opinions
- Output only the summary, no preamble

Conversation segment:
{messages}"""


class ContextCompressor:
    """LLM-powered context compressor.

    Generates summaries of older conversation turns to reduce token usage
    while preserving essential context and decision history.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        max_summary_tokens: int = 500,
    ) -> None:
        """Initialize the compressor.

        Args:
            model: LLM model name for summarization.
            base_url: Optional custom API base URL for OpenAI-compatible services.
            max_summary_tokens: Maximum tokens for each summary output.
        """
        self.model = model
        self.base_url = base_url
        self.max_summary_tokens = max_summary_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package required for compression. "
                    "Install with: pip install openai"
                )
            kwargs: dict[str, str] = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages into a readable string for summarization.

        Args:
            messages: List of message dicts with 'role', 'content', and optional 'timestamp'.

        Returns:
            Formatted string of the conversation.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp")
            prefix = f"[{role}]"
            if timestamp:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    prefix = f"[{role} @ {dt.strftime('%H:%M')}]"
                except (OSError, ValueError):
                    pass
            parts.append(f"{prefix} {content}")
        return "\n".join(parts)

    def summarize(self, messages: list[dict[str, Any]]) -> str:
        """Generate a summary of a list of messages using an LLM.

        Args:
            messages: List of message dicts to summarize.

        Returns:
            A concise summary string.
        """
        if not messages:
            return ""

        if len(messages) == 1:
            content = messages[0].get("content", "")
            # Don't summarize a single short message
            if len(content.split()) < 50:
                return content

        formatted = self._format_messages(messages)
        prompt = SUMMARIZATION_PROMPT.format(messages=formatted)

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_summary_tokens,
                temperature=0.3,
            )
            summary = response.choices[0].message.content
            return summary.strip() if summary else self._fallback_summary(messages)
        except Exception:
            # If LLM call fails, use extractive fallback
            return self._fallback_summary(messages)

    def _fallback_summary(self, messages: list[dict[str, Any]]) -> str:
        """Extractive fallback when LLM is unavailable.

        Takes the first and last message content as a simple summary.

        Args:
            messages: List of message dicts.

        Returns:
            A basic extractive summary.
        """
        if not messages:
            return ""

        parts: list[str] = ["[Conversation Summary]"]

        if len(messages) <= 2:
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                parts.append(f"{role}: {content}")
        else:
            first = messages[0]
            last = messages[-1]
            parts.append(
                f"Started with {first.get('role', 'unknown')}: "
                f"{first.get('content', '')[:150]}..."
            )
            parts.append(f"Last message from {last.get('role', 'unknown')}: {last.get('content', '')[:150]}...")
            parts.append(f"({len(messages)} messages total)")

        return "\n".join(parts)

    def summarize_batch(
        self, message_groups: list[list[dict[str, Any]]]
    ) -> list[str]:
        """Summarize multiple groups of messages.

        Args:
            message_groups: List of message groups to summarize independently.

        Returns:
            List of summary strings, one per group.
        """
        return [self.summarize(group) for group in message_groups]

    def create_summary_message(
        self, summary: str, original_count: int, start_time: float | None = None
    ) -> dict[str, Any]:
        """Create a summary message dict that can replace original messages.

        Args:
            summary: The summary text.
            original_count: Number of original messages this summarizes.
            start_time: Timestamp of the first summarized message.

        Returns:
            A message dict in the standard format with metadata.
        """
        return {
            "role": "system",
            "content": f"[Context Summary — {original_count} messages compressed]\n{summary}",
            "metadata": {
                "type": "summary",
                "compressed_count": original_count,
                "created_at": time.time(),
                "start_time": start_time,
            },
        }
