#!/usr/bin/env python3
"""ContextKit Quick Start Example

Demonstrates the core features of ContextKit:
- Adding messages to context
- Retrieving relevant context via semantic search
- Getting recent messages with token budgeting
- Auto-compressing when context is full
- Budget monitoring

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/quickstart.py
"""

from contextkit import ContextManager


def main() -> None:
    # Initialize ContextKit
    ctx = ContextManager(
        storage="./example_memory",
        max_tokens=4096,  # Small budget for demo purposes
        compress_ratio=0.3,
    )

    print("=== ContextKit Quick Start ===\n")

    # Simulate a coding conversation
    conversations = [
        ("system", "You are a Python coding assistant specialized in data structures."),
        ("user", "How do I implement a binary search tree in Python?"),
        (
            "assistant",
            "Here's a BST implementation:\n\n"
            "class Node:\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
            "        self.left = None\n"
            "        self.right = None\n\n"
            "class BST:\n"
            "    def __init__(self):\n"
            "        self.root = None\n"
            "    def insert(self, value): ...\n"
            "    def search(self, value): ...\n"
            "    def delete(self, value): ...",
        ),
        ("user", "Now add an in-order traversal method"),
        (
            "assistant",
            "def inorder(self, node):\n"
            "    if node is None:\n"
            "        return []\n"
            "    return (self.inorder(node.left) + "
            "[node.value] + self.inorder(node.right))",
        ),
        ("user", "What's the time complexity of BST operations?"),
        (
            "assistant",
            "BST operation complexities:\n"
            "- Average case: O(log n) for insert, search, delete\n"
            "- Worst case (degenerate tree): O(n)\n"
            "- To guarantee O(log n), use self-balancing trees like AVL or Red-Black trees",
        ),
        ("user", "Can you show me how to validate if a binary tree is a valid BST?"),
        (
            "assistant",
            "def is_valid_bst(node, min_val=float('-inf'), max_val=float('inf')):\n"
            "    if node is None:\n"
            "        return True\n"
            "    if not (min_val < node.value < max_val):\n"
            "        return False\n"
            "    return (is_valid_bst(node.left, min_val, node.value) and\n"
            "            is_valid_bst(node.right, node.value, max_val))",
        ),
    ]

    # Add all messages
    for role, content in conversations:
        msg_id = ctx.add(role, content)
        print(f"  Added [{role}] (id={msg_id})")

    print(f"\n📊 Budget: {ctx.token_budget}")

    # Semantic search — find context about a specific topic
    print("\n--- Semantic Search: 'time complexity' ---")
    relevant = ctx.get_relevant("What are the time complexities?", max_tokens=1500)
    for msg in relevant:
        score = msg.get("relevance_score", 0)
        print(f"  [{msg['role']}] (score={score:.2f}) {msg['content'][:80]}...")

    # Get recent messages
    print("\n--- Recent Messages (last 1000 tokens) ---")
    recent = ctx.get_recent(max_tokens=1000)
    for msg in recent:
        print(f"  [{msg['role']}] {msg['content'][:60]}...")

    # Auto-compress
    print("\n--- Auto-Compress ---")
    print(f"  Before: {ctx.token_budget}")
    compressed = ctx.auto_compress()
    print(f"  Compressed: {compressed} messages")
    print(f"  After: {ctx.token_budget}")

    # Export context
    ctx.export("./example_export.json")
    print("\n✅ Context exported to ./example_export.json")

    print(f"\nFinal state: {ctx}")


if __name__ == "__main__":
    main()
