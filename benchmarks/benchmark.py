#!/usr/bin/env python3
"""ContextKit Benchmark Suite.

Measures compression ratios, search accuracy, and token savings
across different context sizes. Produces formatted output tables.

Usage:
    python -m benchmarks.benchmark
    contextkit bench
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure imports work when run directly
_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

from contextkit.core import ContextManager
from contextkit.budget import TokenBudget
from contextkit.compressor import ContextCompressor
from contextkit.indexer import VectorIndex


# ─── Test Data ────────────────────────────────────────────────────

SAMPLE_CONVERSATIONS: list[list[tuple[str, str]]] = [
    # Small conversation (~500 tokens)
    [
        ("system", "You are a helpful coding assistant."),
        ("user", "How do I reverse a string in Python?"),
        ("assistant", "You can reverse a string using slicing: `s[::-1]` or `''.join(reversed(s))`. Both are O(n) time complexity."),
        ("user", "What about reversing a linked list?"),
        ("assistant", "For a linked list, use iterative approach with three pointers: prev, current, next. Time: O(n), Space: O(1)."),
        ("user", "Show me the code"),
        ("assistant", "```python\ndef reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev\n```"),
    ],
    # Medium conversation (~2000 tokens)
    [
        ("system", "You are a senior software architect specializing in distributed systems."),
        ("user", "I need to design a real-time chat system. What architecture should I use?"),
        ("assistant", "For a real-time chat system, I recommend:\n1. **WebSocket connections** for real-time bidirectional communication\n2. **Message queues** (Redis/Kafka) for message routing\n3. **Database** (PostgreSQL/MongoDB) for persistence\n4. **Redis** for session management and presence tracking\n5. **Load balancer** for horizontal scaling\n\nKey considerations:\n- Message ordering guarantees\n- Delivery guarantees (at-least-once vs exactly-once)\n- Reconnection handling\n- Rate limiting"),
        ("user", "How do I handle message ordering with multiple servers?"),
        ("assistant", "Message ordering in distributed systems is tricky. Here are the main approaches:\n\n**1. Single-writer per channel**\n- Each chat room has a designated leader node\n- All writes go through the leader\n- Simple but limits write throughput\n\n**2. Logical timestamps (Lamport/Vector clocks)**\n- Attach timestamps to messages\n- Order by timestamp, break ties with server ID\n- Allows concurrent writes but requires conflict resolution\n\n**3. Sequence numbers with consensus**\n- Use Raft/Paxos to agree on message order\n- Strongest guarantees but highest latency\n\nFor most chat apps, approach #1 with sticky sessions is sufficient. Use #2 if you need multi-region support."),
        ("user", "What about the database schema?"),
        ("assistant", "Recommended schema:\n\n```sql\nCREATE TABLE channels (\n    id UUID PRIMARY KEY,\n    name VARCHAR(255),\n    created_at TIMESTAMP DEFAULT NOW()\n);\n\nCREATE TABLE messages (\n    id UUID PRIMARY KEY,\n    channel_id UUID REFERENCES channels(id),\n    sender_id UUID REFERENCES users(id),\n    content TEXT,\n    sequence_num BIGINT,\n    created_at TIMESTAMP DEFAULT NOW()\n);\n\nCREATE INDEX idx_messages_channel ON messages(channel_id, sequence_num);\n```\n\nUse UUID for IDs to avoid hotspots. The sequence_num column ensures ordering within a channel."),
        ("user", "How do I implement presence tracking?"),
        ("assistant", "Presence tracking shows who's online. Use Redis for this:\n\n```python\nimport redis\nimport time\n\nr = redis.Redis()\n\ndef user_online(user_id, channel_id):\n    key = f\"presence:{channel_id}\"\n    r.hset(key, user_id, time.time())\n    r.expire(key, 60)  # TTL for auto-cleanup\n\ndef get_online_users(channel_id):\n    key = f\"presence:{channel_id}\"\n    users = r.hgetall(key)\n    now = time.time()\n    return [uid for uid, ts in users.items() if now - float(ts) < 30]\n```\n\nHeartbeat every 30s keeps the presence entry alive. TTL handles disconnects gracefully."),
        ("user", "What about scaling horizontally?"),
        ("assistant", "Horizontal scaling for chat:\n\n**Stateless API servers** behind a load balancer. Sticky sessions via consistent hashing on channel_id.\n\n**Redis Pub/Sub** for cross-server message delivery:\n- Server A publishes to `channel:{id}`\n- All servers subscribed to that channel receive it\n- For large-scale, use Redis Cluster with shard-level subscriptions\n\n**Connection management:**\n- Each server maintains WebSocket connections\n- Use a connection registry in Redis for routing\n- `server_id → [user_ids]` mapping\n\n**Scaling limits:**\n- Single Redis: ~100K concurrent connections\n- With Redis Cluster: ~1M+ connections\n- Database: partition by channel_id"),
        ("user", "How do I handle reconnection and missed messages?"),
        ("assistant", "Reconnection strategy:\n\n1. **Client stores last received message_id**\n2. On reconnect, client sends `last_message_id` to server\n3. Server queries messages after that ID\n4. Sends missed messages in order\n\n```python\nasync def handle_reconnect(ws, user_id, last_msg_id):\n    # Fetch missed messages\n    missed = await db.fetch(\n        \"SELECT * FROM messages WHERE id > $1 AND channel_id = $2\",\n        last_msg_id, channel_id\n    )\n    for msg in missed:\n        await ws.send(encode(msg))\n    # Resume real-time stream\n```\n\nAlso implement:\n- Exponential backoff for reconnect attempts\n- Message deduplication on client side\n- Offline message queue for sending"),
    ],
    # Large conversation (~5000 tokens)
    [
        ("system", "You are a machine learning expert helping with a recommendation system."),
        ("user", "I'm building a product recommendation system for an e-commerce platform with 10M users and 500K products. What approach should I use?"),
        ("assistant", "For a large-scale e-commerce recommendation system, I recommend a **hybrid approach** combining multiple strategies:\n\n## Architecture Overview\n\n**1. Candidate Generation (Recall)**\n- Collaborative Filtering (user-user, item-item)\n- Content-based filtering (product embeddings)\n- Popularity-based (trending, bestsellers)\n- Deep learning embeddings (Two-Tower model)\n\n**2. Ranking**\n- Learning-to-Rank model (LambdaMART or neural ranker)\n- Features: user history, product attributes, context\n\n**3. Post-processing**\n- Business rules (diversity, freshness, inventory)\n- A/B testing framework\n\nFor 10M users × 500K products, use approximate nearest neighbor (ANN) search with FAISS or ScaNN for the candidate generation step."),
        ("user", "How should I implement the two-tower model?"),
        ("assistant", "The Two-Tower model learns separate embeddings for users and items:\n\n```python\nimport torch\nimport torch.nn as nn\n\nclass TwoTowerModel(nn.Module):\n    def __init__(self, num_users, num_items, embedding_dim=128):\n        super().__init__()\n        self.user_embedding = nn.Embedding(num_users, embedding_dim)\n        self.item_embedding = nn.Embedding(num_items, embedding_dim)\n        self.user_tower = nn.Sequential(\n            nn.Linear(embedding_dim, 256),\n            nn.ReLU(),\n            nn.Linear(256, 128)\n        )\n        self.item_tower = nn.Sequential(\n            nn.Linear(embedding_dim, 256),\n            nn.ReLU(),\n            nn.Linear(256, 128)\n        )\n    \n    def forward(self, user_ids, item_ids):\n        user_emb = self.user_embedding(user_ids)\n        item_emb = self.item_embedding(item_ids)\n        user_vec = self.user_tower(user_emb)\n        item_vec = self.item_tower(item_emb)\n        return torch.cosine_similarity(user_vec, item_vec)\n```\n\n**Training:**\n- Use in-batch negatives for efficiency\n- Binary cross-entropy loss\n- Sample 256 negatives per positive\n- Train on (user, positive_item, negative_items) triples\n\n**Serving:**\n- Pre-compute all item embeddings → FAISS index\n- At serving time, compute user embedding → ANN search → top-K items\n- Refresh FAISS index every few hours"),
        ("user", "How do I handle the cold start problem for new users and products?"),
        ("assistant", "Cold start strategies:\n\n**New Users (no history):**\n1. **Content-based:** Use demographic features (age, location, device)\n2. **Onboarding:** Show popular/trending items during first session\n3. **Bandit approach:** Use Thompson Sampling to explore\n4. **Transfer learning:** If you have other user signals (clicks, views), use those\n\n```python\ndef get_recommendations(user_id):\n    history = get_user_history(user_id)\n    if len(history) < 5:\n        # Cold start: blend popular + diversity\n        popular = get_popular_items(limit=20)\n        diverse = get_diverse_items(limit=10)\n        return blend(popular, diverse, ratio=0.7)\n    else:\n        # Warm: use trained model\n        return model.predict(user_id)\n```\n\n**New Products (no interactions):**\n1. **Content embeddings:** Use product title, description, category for content-based similarity\n2. **Metadata matching:** Find similar existing products by attributes\n3. **Exploration budget:** Reserve 10% of recommendations for new items\n4. **Gradual exposure:** Start with small impressions, scale up based on CTR"),
        ("user", "What metrics should I track?"),
        ("assistant", "Key metrics for recommendation systems:\n\n**Offline Metrics:**\n- Precision@K, Recall@K, NDCG@K\n- MAP (Mean Average Precision)\n- Coverage (% of catalog recommended)\n- Diversity (intra-list similarity)\n\n**Online Metrics (A/B testing):**\n- CTR (Click-Through Rate)\n- Conversion Rate\n- Revenue per user\n- Session duration\n- Return visit rate\n\n**Operational Metrics:**\n- Latency (p50, p95, p99)\n- Throughput (QPS)\n- FAISS index freshness\n- Model staleness\n\n```python\n# Example metric computation\nfrom sklearn.metrics import ndcg_score\n\ndef evaluate(recommended, relevant, k=10):\n    precision = len(set(recommended[:k]) & set(relevant)) / k\n    recall = len(set(recommended[:k]) & set(relevant)) / len(relevant)\n    ndcg = ndcg_score([relevant], [recommended], k=k)\n    return {\"precision\": precision, \"recall\": recall, \"ndcg\": ndcg}\n```\n\n**Target benchmarks:**\n- CTR improvement > 15% vs baseline\n- Conversion improvement > 10%\n- Latency p99 < 100ms"),
        ("user", "How do I deploy this at scale?"),
        ("assistant", "Deployment architecture for 10M users:\n\n```\n┌─────────────┐     ┌──────────────┐     ┌─────────────┐\n│   CDN/Edge   │────▶│  API Gateway  │────▶│  Load Balancer│\n└─────────────┘     └──────────────┘     └──────┬──────┘\n                                                  │\n                    ┌─────────────────────────────┼─────────────┐\n                    │              │               │             │\n              ┌─────▼────┐  ┌─────▼────┐  ┌──────▼─────┐  ┌────▼─────┐\n              │Rec Server│  │Rec Server│  │Rec Server  │  │Rec Server│\n              │  (GPU)   │  │  (GPU)   │  │  (GPU)     │  │  (GPU)   │\n              └─────┬────┘  └─────┬────┘  └──────┬─────┘  └────┬─────┘\n                    │              │               │             │\n              ┌─────▼──────────────▼───────────────▼─────────────▼─────┐\n              │                    Redis Cluster                       │\n              │           (Embeddings + Cache + Pub/Sub)               │\n              └─────────────────────────┬─────────────────────────────┘\n                                        │\n              ┌─────────────────────────▼─────────────────────────────┐\n              │                   PostgreSQL (Primary + Replicas)      │\n              └───────────────────────────────────────────────────────┘\n```\n\n**Infrastructure:**\n- 4-8 GPU servers for model serving (T4 or A10G)\n- Redis Cluster (6 nodes) for caching and ANN\n- PostgreSQL with read replicas for persistence\n- Kubernetes for orchestration\n- Prometheus + Grafana for monitoring\n\n**Cost estimate:** ~$3-5K/month on AWS"),
    ],
]


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def _measure_tokens(messages: list[dict]) -> int:
    """Measure total tokens in a list of messages."""
    total = 0
    for msg in messages:
        total += _count_tokens(msg.get("role", "") + " " + msg.get("content", ""))
    return total


# ─── Benchmark Functions ──────────────────────────────────────────

def bench_compression() -> list[dict[str, Any]]:
    """Test compression ratios across different context sizes."""
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="ctxkit_bench_")

    size_labels = ["Small", "Medium", "Large"]

    for i, conv in enumerate(SAMPLE_CONVERSATIONS):
        label = size_labels[i]
        storage = os.path.join(tmp_dir, f"bench_{i}")
        ctx = ContextManager(
            storage=storage,
            max_tokens=1000000,  # Large budget to prevent auto-compress
            embedding_model=None,
        )

        # Add messages with timestamps spread over time
        base_time = time.time() - 7200  # 2 hours ago
        for j, (role, content) in enumerate(conv):
            ctx.add(role, content, metadata={})
            # Backdate timestamps to simulate old messages
            ctx._messages[-1]["timestamp"] = base_time + j * 300

        # Count original tokens
        original_tokens = _measure_tokens(ctx.messages)
        original_count = len(ctx.messages)

        # Compress (messages are all old enough)
        ctx._compressor = ContextCompressor(model="gpt-4o-mini")

        before = _measure_tokens(ctx.messages)
        # Use fallback summary (no API needed)
        old_msgs = [m for m in ctx.messages if m.get("metadata", {}).get("type") != "summary"]
        if len(old_msgs) >= 3:
            summary_text = ctx._compressor._fallback_summary(old_msgs)
            summary_msg = ctx._compressor.create_summary_message(
                summary=summary_text,
                original_count=len(old_msgs),
                start_time=old_msgs[0].get("timestamp"),
            )
            compressed = [summary_msg]
        else:
            compressed = ctx.messages

        after_tokens = _measure_tokens(compressed)
        ratio = (1 - after_tokens / original_tokens) * 100 if original_tokens > 0 else 0

        results.append({
            "size": label,
            "original_messages": original_count,
            "original_tokens": original_tokens,
            "compressed_messages": len(compressed),
            "compressed_tokens": after_tokens,
            "savings_pct": ratio,
        })

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


def bench_search_accuracy() -> list[dict[str, Any]]:
    """Test search relevance across different query types."""
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="ctxkit_bench_")

    # Build a context with diverse topics
    storage = os.path.join(tmp_dir, "search_test")
    ctx = ContextManager(
        storage=storage,
        max_tokens=1000000,
        embedding_model=None,  # Use keyword fallback
    )

    # Add messages from different topics
    topics = {
        "python_sorting": [
            ("user", "How do I sort a dictionary by value in Python?"),
            ("assistant", "Use `sorted(d.items(), key=lambda x: x[1])` or `dict(sorted(d.items(), key=lambda item: item[1]))`"),
        ],
        "javascript_async": [
            ("user", "What's the difference between Promise.all and Promise.allSettled?"),
            ("assistant", "`Promise.all` rejects on first failure. `Promise.allSettled` waits for all and returns status for each."),
        ],
        "docker_deploy": [
            ("user", "How do I set up a multi-stage Docker build?"),
            ("assistant", "Use multiple FROM statements. First stage builds, second stage copies only the artifact. Reduces image size significantly."),
        ],
        "sql_optimization": [
            ("user", "How can I optimize a slow SQL query with JOINs?"),
            ("assistant", "Add indexes on JOIN columns, use EXPLAIN to analyze the query plan, avoid SELECT *, and consider denormalizing for read-heavy workloads."),
        ],
        "react_state": [
            ("user", "When should I use useState vs useReducer in React?"),
            ("assistant", "Use useState for simple state. Use useReducer when state logic is complex, involves multiple sub-values, or the next state depends on the previous one."),
        ],
    }

    for topic, msgs in topics.items():
        for role, content in msgs:
            ctx.add(role, content)

    # Test queries with expected relevant topics
    test_cases = [
        ("sorting dictionaries python", "python_sorting"),
        ("Promise.all JavaScript", "javascript_async"),
        ("Docker multi-stage build", "docker_deploy"),
        ("SQL query optimization", "sql_optimization"),
        ("React state management", "react_state"),
    ]

    for query, expected_topic in test_cases:
        # Simple keyword search
        query_words = set(query.lower().split())
        scored = []
        for msg in ctx.messages:
            content = msg.get("content", "").lower()
            content_words = set(content.split())
            overlap = len(query_words & content_words)
            score = overlap / len(query_words) if query_words else 0
            if score > 0:
                scored.append((score, msg))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Check if top result is from the expected topic
        found_relevant = False
        if scored:
            top_content = scored[0][1].get("content", "").lower()
            expected_keywords = expected_topic.replace("_", " ").split()
            found_relevant = any(kw in top_content for kw in expected_keywords)

        results.append({
            "query": query,
            "expected_topic": expected_topic,
            "found_relevant": found_relevant,
            "top_score": scored[0][0] if scored else 0,
            "total_matches": len(scored),
        })

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


def bench_token_budget() -> list[dict[str, Any]]:
    """Benchmark token counting accuracy and budget management."""
    results = []

    test_texts = [
        ("Short", "Hello, world!"),
        ("Medium", "The quick brown fox jumps over the lazy dog. " * 10),
        ("Long", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100),
        ("Code", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\n" * 50),
        ("Mixed", "Hello world. def foo(): pass. 你好世界。" * 20),
    ]

    for label, text in test_texts:
        tokens = _count_tokens(text)
        chars = len(text)
        chars_per_token = chars / tokens if tokens > 0 else 0

        budget = TokenBudget(max_tokens=1000)
        msg_tokens = budget.count_message_tokens("user", text)
        status = budget.budget_status(msg_tokens)

        results.append({
            "type": label,
            "chars": chars,
            "tokens": tokens,
            "chars_per_token": round(chars_per_token, 2),
            "message_tokens": msg_tokens,
            "within_budget": not status["is_over_budget"],
        })

    return results


def bench_index_operations() -> list[dict[str, Any]]:
    """Benchmark vector index add/search/remove operations."""
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="ctxkit_bench_")

    sizes = [100, 500, 1000]

    for size in sizes:
        storage = os.path.join(tmp_dir, f"idx_{size}")
        idx = VectorIndex(storage_dir=storage)

        # Benchmark add
        start = time.time()
        for i in range(size):
            idx.add(f"msg_{i}", f"Message number {i} about topic {i % 10}")
        add_time = time.time() - start

        # Benchmark search
        start = time.time()
        for _ in range(10):
            idx.search("topic 5", top_k=10)
        search_time = (time.time() - start) / 10

        # Benchmark save/load
        start = time.time()
        idx.save()
        save_time = time.time() - start

        start = time.time()
        idx.load()
        load_time = time.time() - start

        results.append({
            "size": size,
            "add_total_ms": round(add_time * 1000, 1),
            "add_per_msg_ms": round(add_time / size * 1000, 2),
            "search_ms": round(search_time * 1000, 2),
            "save_ms": round(save_time * 1000, 1),
            "load_ms": round(load_time * 1000, 1),
        })

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


# ─── Output Formatting ────────────────────────────────────────────

def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted table."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print
    print(f"\n{'='*sum(widths + [3*(len(widths)-1) + 4])}")
    print(f"  {title}")
    print(f"{'='*sum(widths + [3*(len(widths)-1) + 4])}")

    # Header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")

    # Rows
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(f"  {line}")

    print()


# ─── Main ─────────────────────────────────────────────────────────

def run_benchmarks() -> None:
    """Run all benchmarks and display results."""
    print("\n" + "="*60)
    print("  ContextKit Benchmark Suite v0.2.0")
    print("  Running on Python " + sys.version.split()[0])
    print("="*60)

    # 1. Token Budget
    print("\n📊 Benchmark 1: Token Counting & Budget")
    budget_results = bench_token_budget()
    _print_table(
        "Token Counting Accuracy",
        ["Type", "Chars", "Tokens", "Chars/Token", "Msg Tokens", "Within Budget"],
        [
            [
                r["type"],
                f'{r["chars"]:,}',
                f'{r["tokens"]:,}',
                str(r["chars_per_token"]),
                f'{r["message_tokens"]:,}',
                "✅" if r["within_budget"] else "❌",
            ]
            for r in budget_results
        ],
    )

    # 2. Compression
    print("📊 Benchmark 2: Context Compression")
    compression_results = bench_compression()
    _print_table(
        "Compression Ratios by Context Size",
        ["Size", "Original", "After Compress", "Token Savings", "Messages"],
        [
            [
                r["size"],
                f'{r["original_tokens"]:,} tok ({r["original_messages"]} msgs)',
                f'{r["compressed_tokens"]:,} tok ({r["compressed_messages"]} msgs)',
                f'{r["savings_pct"]:.1f}%',
                f'{r["original_messages"]} → {r["compressed_messages"]}',
            ]
            for r in compression_results
        ],
    )

    # 3. Search
    print("📊 Benchmark 3: Search Accuracy")
    search_results = bench_search_accuracy()
    _print_table(
        "Keyword Search Accuracy",
        ["Query", "Expected Topic", "Found?", "Top Score", "Matches"],
        [
            [
                r["query"][:35],
                r["expected_topic"],
                "✅" if r["found_relevant"] else "❌",
                f'{r["top_score"]:.3f}',
                str(r["total_matches"]),
            ]
            for r in search_results
        ],
    )
    accuracy = sum(1 for r in search_results if r["found_relevant"]) / len(search_results) * 100
    print(f"  Overall Accuracy: {accuracy:.0f}%\n")

    # 4. Index Operations
    print("📊 Benchmark 4: Index Operations")
    index_results = bench_index_operations()
    _print_table(
        "Vector Index Performance",
        ["Size", "Add (total)", "Add (per msg)", "Search", "Save", "Load"],
        [
            [
                str(r["size"]),
                f'{r["add_total_ms"]:.1f}ms',
                f'{r["add_per_msg_ms"]:.2f}ms',
                f'{r["search_ms"]:.1f}ms',
                f'{r["save_ms"]:.1f}ms',
                f'{r["load_ms"]:.1f}ms',
            ]
            for r in index_results
        ],
    )

    # Summary
    print("="*60)
    print("  Summary")
    print("="*60)
    avg_savings = sum(r["savings_pct"] for r in compression_results) / len(compression_results)
    print(f"  • Average compression savings: {avg_savings:.1f}%")
    print(f"  • Search accuracy: {accuracy:.0f}%")
    print(f"  • Index add throughput: {index_results[-1]['add_per_msg_ms']:.2f}ms/msg")
    print(f"  • Index search latency: {index_results[-1]['search_ms']:.1f}ms")
    print("="*60)
    print()


if __name__ == "__main__":
    run_benchmarks()
