#!/usr/bin/env python3
"""ContextKit + OpenAI Agents Integration Example

Shows how to use ContextKit with OpenAI's API for context-aware agents
with semantic memory and automatic compression.

Features demonstrated:
- Semantic context retrieval with vector search
- Smart merging of relevant + recent context
- Periodic auto-compression
- Token budget tracking

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/with_openai.py
"""

import os

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    exit(1)

from contextkit import ContextManager


def merge_context(
    relevant: list[dict], recent: list[dict], max_items: int = 30
) -> list[dict]:
    """Merge relevant and recent context, removing duplicates.

    Strategy:
    1. Take all relevant results (from semantic search)
    2. Add recent messages that weren't already found
    3. Cap at max_items to avoid overwhelming the model

    Args:
        relevant: Messages from semantic search.
        recent: Recent messages for conversation continuity.
        max_items: Maximum messages to include.

    Returns:
        Merged, deduplicated list of messages.
    """
    seen = set()
    merged: list[dict] = []

    # Relevant messages first (highest priority)
    for msg in relevant:
        content_key = hash(msg["content"][:200])
        if content_key not in seen:
            seen.add(content_key)
            merged.append(msg)

    # Then recent messages (for conversation continuity)
    for msg in recent:
        content_key = hash(msg["content"][:200])
        if content_key not in seen:
            seen.add(content_key)
            merged.append(msg)

    return merged[:max_items]


def create_openai_agent(
    model: str = "gpt-4o",
    storage_dir: str = "./openai_memory",
    max_context_tokens: int = 100000,
):
    """Create an OpenAI agent with ContextKit-powered memory."""

    client = OpenAI()

    ctx = ContextManager(
        storage=storage_dir,
        max_tokens=max_context_tokens,
        compress_ratio=0.3,
        embedding_model="text-embedding-3-small",
    )

    # System prompt
    ctx.add(
        "system",
        "You are a knowledgeable AI assistant. You remember previous "
        "conversations and can reference them when relevant. You are "
        "concise but thorough.",
    )

    def agent_turn(user_input: str) -> str:
        """Process a user turn with smart context management."""

        ctx.add("user", user_input)

        # Two retrieval strategies in parallel:
        # 1. Semantic search for topically relevant context
        relevant = ctx.get_relevant(user_input, max_tokens=max_context_tokens // 2)

        # 2. Recent messages for conversation continuity
        recent = ctx.get_recent(max_tokens=max_context_tokens // 2)

        # Merge with deduplication
        context_msgs = merge_context(relevant, recent)

        # Build OpenAI message list
        messages = []
        for msg in context_msgs:
            if msg["role"] in ("system", "user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Always include the current user message
        messages.append({"role": "user", "content": user_input})

        # Call OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )

        reply = response.choices[0].message.content or ""
        ctx.add("assistant", reply)

        # Auto-compress periodically
        ctx.auto_compress()

        # Log budget usage
        budget = ctx.token_budget
        token_usage = response.usage
        if token_usage:
            print(
                f"  [tokens: prompt={token_usage.prompt_tokens}, "
                f"completion={token_usage.completion_tokens} | "
                f"budget: {budget['utilization']} used]"
            )

        return reply

    return agent_turn, ctx


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return

    print("=== OpenAI Agents + ContextKit ===")
    print("Type 'quit' to exit, 'budget' to check usage, 'history' to see messages\n")

    agent, ctx = create_openai_agent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "budget":
            print(f"\n📊 {ctx.token_budget}\n")
            continue
        if user_input.lower() == "history":
            print(f"\n📜 Message History ({len(ctx.messages)} messages):")
            for msg in ctx.messages[-10:]:
                print(f"  [{msg['role']}] {msg['content'][:80]}...")
            print()
            continue

        try:
            response = agent(user_input)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
