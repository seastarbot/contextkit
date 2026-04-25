#!/usr/bin/env python3
"""ContextKit + Claude Code Integration Example

Shows how to use ContextKit with the Anthropic API (Claude) for
persistent, context-aware coding sessions.

Features demonstrated:
- Cross-session memory persistence
- Smart context retrieval for Claude's 200K context window
- Automatic compression before hitting limits
- Token budget monitoring

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."  # For embeddings
    python examples/with_claude_code.py
"""

import os

try:
    import anthropic
except ImportError:
    print("Install anthropic: pip install anthropic")
    exit(1)

from contextkit import ContextManager


def create_claude_agent(storage_dir: str = "./claude_memory"):
    """Create a Claude agent with ContextKit-powered memory."""

    ctx = ContextManager(
        storage=storage_dir,
        max_tokens=200000,
        compress_ratio=0.25,  # Compress at 75% utilization
        embedding_model="text-embedding-3-small",
    )

    # Add system prompt
    ctx.add(
        "system",
        "You are an expert Python developer. You help users write clean, "
        "efficient code. When discussing code, always explain your reasoning.",
    )

    client = anthropic.Anthropic()

    def chat(user_input: str) -> str:
        """Send a message and get a response with full context management."""

        # Add user message
        ctx.add("user", user_input)

        # Build context: get recent messages for conversation continuity
        # Claude has a large context window, so we can include more
        history = ctx.get_recent(max_tokens=150000)

        messages = []
        for msg in history:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Get system message if present
        system_prompt = ""
        for msg in ctx.messages:
            if msg["role"] == "system" and "Context Summary" not in msg["content"]:
                system_prompt = msg["content"]
                break

        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )

        reply = response.content[0].text
        ctx.add("assistant", reply)

        # Monitor budget and auto-compress if needed
        budget = ctx.token_budget
        utilization = float(budget["utilization"].rstrip("%"))
        if utilization > 70:
            print(f"\n⚡ Auto-compressing (utilization: {budget['utilization']})")
            ctx.summarize_older_than(hours=1)

        return reply

    return chat, ctx


def main() -> None:
    # Check for required API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY environment variable")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable (for embeddings)")
        return

    print("=== Claude Code + ContextKit ===")
    print("Type 'quit' to exit, 'budget' to check token usage\n")

    chat, ctx = create_claude_agent()

    # Session 1 context will persist to session 2
    print("Session 1 (messages persist to disk):\n")

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

        try:
            response = chat(user_input)
            print(f"\nClaude: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

    # Demonstrate cross-session persistence
    print("\n--- Session 2: Context loaded from disk ---")
    chat2, ctx2 = create_claude_agent("./claude_memory")
    print(f"Loaded {len(ctx2.messages)} messages from previous session")
    print(f"Budget: {ctx2.token_budget}")

    # Ask something related to the previous conversation
    try:
        response = chat2("What were we discussing earlier?")
        print(f"\nClaude: {response}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
