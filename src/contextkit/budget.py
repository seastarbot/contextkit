"""Token budget management using tiktoken.

Provides accurate token counting and budget tracking for context windows.
"""

from __future__ import annotations

import tiktoken


class TokenBudget:
    """Tracks token usage against a configurable budget.

    Uses tiktoken for fast, accurate token counting. Supports per-model
    encoding selection and provides real-time budget utilization metrics.
    """

    # Model name → tiktoken encoding mapping
    MODEL_ENCODINGS: dict[str, str] = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude-3-opus": "cl100k_base",
        "claude-3-sonnet": "cl100k_base",
        "claude-3-haiku": "cl100k_base",
        "claude-3.5-sonnet": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }

    def __init__(self, max_tokens: int = 200000, encoding_name: str | None = None) -> None:
        """Initialize the token budget manager.

        Args:
            max_tokens: Maximum token budget for the context window.
            encoding_name: tiktoken encoding name. Auto-detected from model
                          name if not provided.
        """
        self.max_tokens = max_tokens
        self._encoding_name = encoding_name or "cl100k_base"
        try:
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        except Exception:
            # Fallback to cl100k_base if the requested encoding is unavailable
            self._encoding = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def for_model(cls, model_name: str, max_tokens: int = 200000) -> "TokenBudget":
        """Create a budget manager auto-configured for a specific model.

        Args:
            model_name: The model name (e.g., "gpt-4o", "claude-3.5-sonnet").
            max_tokens: Maximum token budget.

        Returns:
            A TokenBudget configured with the correct encoding for the model.
        """
        encoding_name = cls.MODEL_ENCODINGS.get(model_name, "cl100k_base")
        return cls(max_tokens=max_tokens, encoding_name=encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message_tokens(self, role: str, content: str) -> int:
        """Count tokens for a chat message (role + content + overhead).

        Chat messages have per-message overhead (~4 tokens) plus the
        role token (~1-2 tokens). This method accounts for that.

        Args:
            role: The message role (e.g., "user", "assistant", "system").
            content: The message content.

        Returns:
            Total tokens including overhead.
        """
        # Per-message overhead: <|start|>{role}\n ... <|end|> = ~4 tokens
        role_tokens = self.count_tokens(role)
        content_tokens = self.count_tokens(content)
        return role_tokens + content_tokens + 4

    def budget_status(self, used_tokens: int) -> dict[str, object]:
        """Get current budget utilization status.

        Args:
            used_tokens: Number of tokens currently used.

        Returns:
            Dictionary with total, used, remaining, and utilization percentage.
        """
        remaining = max(0, self.max_tokens - used_tokens)
        utilization = (used_tokens / self.max_tokens * 100) if self.max_tokens > 0 else 0.0

        return {
            "total": self.max_tokens,
            "used": used_tokens,
            "remaining": remaining,
            "utilization": f"{utilization:.1f}%",
            "is_over_budget": used_tokens > self.max_tokens,
        }

    def headroom(self, used_tokens: int) -> int:
        """Return the number of tokens available before hitting the budget.

        Args:
            used_tokens: Number of tokens currently used.

        Returns:
            Number of remaining tokens (0 if over budget).
        """
        return max(0, self.max_tokens - used_tokens)

    def is_near_limit(self, used_tokens: int, threshold: float = 0.85) -> bool:
        """Check if token usage is near the budget limit.

        Args:
            used_tokens: Number of tokens currently used.
            threshold: Utilization ratio to consider "near limit" (0.0-1.0).

        Returns:
            True if utilization exceeds the threshold.
        """
        if self.max_tokens <= 0:
            return True
        return (used_tokens / self.max_tokens) >= threshold

    def max_content_tokens(self, overhead_tokens: int = 0) -> int:
        """Calculate max content tokens after reserving overhead.

        Useful for determining how much content to include while leaving
        room for the system prompt, response, and other fixed costs.

        Args:
            overhead_tokens: Tokens to reserve for system prompt, response, etc.

        Returns:
            Maximum tokens available for content.
        """
        return max(0, self.max_tokens - overhead_tokens)
