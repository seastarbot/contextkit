#!/usr/bin/env python3
"""ContextKit Benchmark Suite.

Measures compression ratios, search accuracy, index throughput, and token
counting performance across different context sizes.  Produces formatted
tables suitable for README inclusion.

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

# Ensure imports work when run directly from the project root
_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

from contextkit.budget import TokenBudget
from contextkit.compressor import ContextCompressor
from contextkit.core import ContextManager
from contextkit.indexer import VectorIndex


# ─── Sample Conversations ─────────────────────────────────────────────────

SAMPLE_CONVERSATIONS: list[list[tuple[str, str]]] = [
    # Small conversation (~500 tokens)
    [
        ("system", "You are a helpful coding assistant."),
        ("user", "How do I reverse a string in Python?"),
        (
            "assistant",
            "You can reverse a string using slicing: `s[::-1]` or "
            "`''.join(reversed(s))`. Both are O(n) time complexity.",
        ),
        ("user", "What about reversing a linked list?"),
        (
            "assistant",
            "For a linked list, use an iterative approach with three pointers: "
            "prev, current, next. Time: O(n), Space: O(1).",
        ),
        ("user", "Show me the code"),
        (
            "assistant",
            "```python\ndef reverse_list(head):\n    prev = None\n    curr = head\n"
            "    while curr:\n        nxt = curr.next\n        curr.next = prev\n"
            "        prev = curr\n        curr = nxt\n    return prev\n```",
        ),
    ],
    # Medium conversation (~2000 tokens)
    [
        (
            "system",
            "You are a senior software architect specializing in distributed systems.",
        ),
        (
            "user",
            "I need to design a real-time chat system. What architecture should I use?",
        ),
        (
            "assistant",
            "For a real-time chat system, I recommend:\n"
            "1. **WebSocket connections** for real-time bidirectional communication\n"
            "2. **Message queues** (Redis/Kafka) for message routing\n"
            "3. **Database** (PostgreSQL/MongoDB) for persistence\n"
            "4. **Redis** for session management and presence tracking\n"
            "5. **Load balancer** for horizontal scaling\n\n"
            "Key considerations:\n"
            "- Message ordering guarantees\n"
            "- Delivery guarantees (at-least-once vs exactly-once)\n"
            "- Reconnection handling\n"
            "- Rate limiting",
        ),
        ("user", "How do I handle message ordering with multiple servers?"),
        (
            "assistant",
            "Message ordering in distributed systems is tricky. Main approaches:\n\n"
            "**1. Single-writer per channel**\n"
            "- Each chat room has a designated leader node\n"
            "- All writes go through the leader\n"
            "- Simple but limits write throughput\n\n"
            "**2. Logical timestamps (Lamport/Vector clocks)**\n"
            "- Attach timestamps to messages\n"
            "- Order by timestamp, break ties with server ID\n\n"
            "**3. Sequence numbers with consensus**\n"
            "- Use Raft/Paxos to agree on message order\n"
            "- Strongest guarantees but highest latency\n\n"
            "For most chat apps, approach #1 with sticky sessions is sufficient.",
        ),
        ("user", "What about the database schema?"),
        (
            "assistant",
            "Recommended schema:\n\n"
            "```sql\nCREATE TABLE channels (\n"
            "    id UUID PRIMARY KEY,\n"
            "    name VARCHAR(255),\n"
            "    created_at TIMESTAMP DEFAULT NOW()\n"
            ");\n\n"
            "CREATE TABLE messages (\n"
            "    id UUID PRIMARY KEY,\n"
            "    channel_id UUID REFERENCES channels(id),\n"
            "    sender_id UUID REFERENCES users(id),\n"
            "    content TEXT,\n"
            "    sequence_num BIGINT,\n"
            "    created_at TIMESTAMP DEFAULT NOW()\n"
            ");\n\n"
            "CREATE INDEX idx_messages_channel ON messages(channel_id, sequence_num);\n"
            "```\n\n"
            "Use UUID for IDs to avoid hotspots. The sequence_num ensures ordering.",
        ),
        ("user", "How do I implement presence tracking?"),
        (
            "assistant",
            "Presence tracking with Redis:\n\n"
            "```python\nimport redis, time\nr = redis.Redis()\n\n"
            "def user_online(user_id, channel_id):\n"
            "    key = f'presence:{channel_id}'\n"
            "    r.hset(key, user_id, time.time())\n"
            "    r.expire(key, 60)  # TTL for auto-cleanup\n\n"
            "def get_online_users(channel_id):\n"
            "    key = f'presence:{channel_id}'\n"
            "    users = r.hgetall(key)\n"
            "    now = time.time()\n"
            "    return [uid for uid, ts in users.items() if now - float(ts) < 30]\n"
            "```\n\nHeartbeat every 30s keeps presence alive.",
        ),
    ],
    # Large conversation (~5000 tokens)
    [
        (
            "system",
            "You are a machine learning expert helping with a recommendation system.",
        ),
        (
            "user",
            "I'm building a product recommendation system for an e-commerce platform "
            "with 10M users and 500K products. What approach should I use?",
        ),
        (
            "assistant",
            "For a large-scale e-commerce recommendation system, I recommend a "
            "**hybrid approach**:\n\n"
            "## Architecture\n\n"
            "**1. Candidate Generation (Recall)**\n"
            "- Collaborative Filtering (user-user, item-item)\n"
            "- Content-based filtering (product embeddings)\n"
            "- Popularity-based (trending, bestsellers)\n"
            "- Deep learning embeddings (Two-Tower model)\n\n"
            "**2. Ranking**\n"
            "- Learning-to-Rank model (LambdaMART or neural ranker)\n"
            "- Features: user history, product attributes, context\n\n"
            "**3. Post-processing**\n"
            "- Business rules (diversity, freshness, inventory)\n"
            "- A/B testing framework\n\n"
            "For 10M users × 500K products, use FAISS or ScaNN for ANN search.",
        ),
        ("user", "How should I implement the two-tower model?"),
        (
            "assistant",
            "The Two-Tower model learns separate embeddings for users and items:\n\n"
            "```python\nimport torch\nimport torch.nn as nn\n\n"
            "class TwoTowerModel(nn.Module):\n"
            "    def __init__(self, num_users, num_items, embedding_dim=128):\n"
            "        super().__init__()\n"
            "        self.user_embedding = nn.Embedding(num_users, embedding_dim)\n"
            "        self.item_embedding = nn.Embedding(num_items, embedding_dim)\n"
            "        self.user_tower = nn.Sequential(\n"
            "            nn.Linear(embedding_dim, 256), nn.ReLU(),\n"
            "            nn.Linear(256, 128)\n"
            "        )\n"
            "        self.item_tower = nn.Sequential(\n"
            "            nn.Linear(embedding_dim, 256), nn.ReLU(),\n"
            "            nn.Linear(256, 128)\n"
            "        )\n\n"
            "    def forward(self, user_ids, item_ids):\n"
            "        user_vec = self.user_tower(self.user_embedding(user_ids))\n"
            "        item_vec = self.item_tower(self.item_embedding(item_ids))\n"
            "        return torch.cosine_similarity(user_vec, item_vec)\n"
            "```\n\n"
            "Training: use in-batch negatives, binary cross-entropy loss, "
            "256 negatives per positive.\n"
            "Serving: pre-compute item embeddings → FAISS → ANN search at query time.",
        ),
        ("user", "How do I handle cold start for new users and products?"),
        (
            "assistant",
            "Cold start strategies:\n\n"
            "**New Users (no history):**\n"
            "1. Content-based: demographic features (age, location, device)\n"
            "2. Onboarding: show popular/trending during first session\n"
            "3. Bandit approach: Thompson Sampling to explore\n\n"
            "```python\ndef get_recommendations(user_id):\n"
            "    history = get_user_history(user_id)\n"
            "    if len(history) < 5:\n"
            "        popular = get_popular_items(limit=20)\n"
            "        diverse = get_diverse_items(limit=10)\n"
            "        return blend(popular, diverse, ratio=0.7)\n"
            "    return model.predict(user_id)\n"
            "```\n\n"
            "**New Products:**\n"
            "1. Content embeddings from title/description\n"
            "2. Metadata matching with similar existing products\n"
            "3. Reserve 10% of recommendations for new items\n"
            "4. Scale exposure based on CTR",
        ),
        ("user", "What metrics should I track?"),
        (
            "assistant",
            "Key metrics:\n\n"
            "**Offline:** Precision@K, Recall@K, NDCG@K, MAP, Coverage, Diversity\n\n"
            "**Online (A/B):** CTR, Conversion Rate, Revenue per user, Session duration\n\n"
            "**Operational:** Latency (p50/p95/p99), FAISS freshness, Model staleness\n\n"
            "```python\nfrom sklearn.metrics import ndcg_score\n\n"
            "def evaluate(recommended, relevant, k=10):\n"
            "    precision = len(set(recommended[:k]) & set(relevant)) / k\n"
            "    recall = len(set(recommended[:k]) & set(relevant)) / len(relevant)\n"
            "    ndcg = ndcg_score([relevant], [recommended], k=k)\n"
            "    return dict(precision=precision, recall=recall, ndcg=ndcg)\n"
            "```\n\n"
            "Targets: CTR +15%, Conversion +10%, Latency p99 < 100ms",
        ),
        ("user", "How do I deploy this at scale?"),
        (
            "assistant",
            "Deployment for 10M users:\n\n"
            "```\n"
            "CDN → API Gateway → Load Balancer\n"
            "                         │\n"
            "           ┌─────────────┼─────────────┐\n"
            "     Rec Server     Rec Server     Rec Server\n"
            "       (GPU)          (GPU)          (GPU)\n"
            "           └─────────────┼─────────────┘\n"
            "                  Redis Cluster\n"
            "             (Embeddings + Cache)\n"
            "                         │\n"
            "              PostgreSQL (Primary + Replicas)\n"
            "```\n\n"
            "Infrastructure:\n"
            "- 4-8 GPU servers (T4 or A10G) for model serving\n"
            "- Redis Cluster (6 nodes) for caching and ANN\n"
            "- PostgreSQL with read replicas\n"
            "- Kubernetes for orchestration\n"
            "- Prometheus + Grafana for monitoring\n\n"
            "Cost estimate: ~$3-5K/month on AWS",
        ),
    ],
]


# ─── Helpers ──────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (falls back to char-based estimate)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def _measure_tokens(messages: list[dict]) -> int:
    """Measure total tokens in a list of messages."""
    return sum(
        _count_tokens(msg.get("role", "") + " " + msg.get("content", ""))
        for msg in messages
    )


def _elapsed(start: float) -> str:
    """Format elapsed time from a start timestamp."""
    ms = (time.perf_counter() - start) * 1000
    if ms < 1:
        return f"{ms * 1000:.1f}µs"
    if ms < 1000:
        return f"{ms:.2f}ms"
    return f"{ms / 1000:.2f}s"


# ─── Benchmark Functions ──────────────────────────────────────────────────


def bench_token_budget() -> list[dict[str, Any]]:
    """Benchmark token counting accuracy and budget management."""
    results = []

    test_texts = [
        ("Short", "Hello, world!"),
        ("Medium", "The quick brown fox jumps over the lazy dog. " * 10),
        ("Long", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100),
        ("Code", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\n" * 50),
        ("Mixed EN+ZH", "Hello world. def foo(): pass. \u4f60\u597d\u4e16\u754c\u3002" * 20),
    ]

    for label, text in test_texts:
        t0 = time.perf_counter()
        tokens = _count_tokens(text)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        chars = len(text)
        chars_per_token = chars / tokens if tokens > 0 else 0

        budget = TokenBudget(max_tokens=1000)
        msg_tokens = budget.count_message_tokens("user", text)
        status = budget.budget_status(msg_tokens)

        results.append(
            {
                "type": label,
                "chars": chars,
                "tokens": tokens,
                "chars_per_token": round(chars_per_token, 2),
                "message_tokens": msg_tokens,
                "within_budget": not status["is_over_budget"],
                "count_time_us": round(elapsed_us, 1),
            }
        )

    return results


def bench_compression() -> list[dict[str, Any]]:
    """Test compression ratios across different context sizes."""
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="ctxkit_bench_")
    size_labels = ["Small", "Medium", "Large"]

    try:
        for i, conv in enumerate(SAMPLE_CONVERSATIONS):
            label = size_labels[i]
            storage = os.path.join(tmp_dir, f"bench_{i}")
            ctx = ContextManager(
                storage=storage,
                max_tokens=1_000_000,
                embedding_model=None,
            )

            base_time = time.time() - 7200  # 2 hours ago
            for j, (role, content) in enumerate(conv):
                ctx.add(role, content, metadata={})
                ctx._messages[-1]["timestamp"] = base_time + j * 300

            original_tokens = _measure_tokens(ctx.messages)
            original_count = len(ctx.messages)

            # Use fallback summary (no API needed)
            compressor = ContextCompressor(model="gpt-4o-mini")
            old_msgs = [
                m
                for m in ctx.messages
                if m.get("metadata", {}).get("type") != "summary"
            ]

            t0 = time.perf_counter()
            if len(old_msgs) >= 3:
                summary_text = compressor._fallback_summary(old_msgs)
                summary_msg = compressor.create_summary_message(
                    summary=summary_text,
                    original_count=len(old_msgs),
                    start_time=old_msgs[0].get("timestamp"),
                )
                compressed = [summary_msg]
            else:
                compressed = ctx.messages
            compress_ms = (time.perf_counter() - t0) * 1000

            after_tokens = _measure_tokens(compressed)
            savings_pct = (
                (1 - after_tokens / original_tokens) * 100 if original_tokens > 0 else 0
            )

            results.append(
                {
                    "size": label,
                    "original_messages": original_count,
                    "original_tokens": original_tokens,
                    "compressed_messages": len(compressed),
                    "compressed_tokens": after_tokens,
                    "savings_pct": round(savings_pct, 1),
                    "compress_ms": round(compress_ms, 2),
                }
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def bench_search_accuracy() -> list[dict[str, Any]]:
    """Test keyword search relevance across different query types."""
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="ctxkit_bench_")

    try:
        storage = os.path.join(tmp_dir, "search_test")
        ctx = ContextManager(storage=storage, max_tokens=1_000_000, embedding_model=None)

        topics = {
            "python_sorting": [
                ("user", "How do I sort a dictionary by value in Python?"),
                (
                    "assistant",
                    "Use `sorted(d.items(), key=lambda x: x[1])` "
                    "or `dict(sorted(d.items(), key=lambda item: item[1]))`",
                ),
            ],
            "javascript_async": [
                (
                    "user",
                    "What's the difference between Promise.all and Promise.allSettled?",
                ),
                (
                    "assistant",
                    "`Promise.all` rejects on first failure. `Promise.allSettled` "
                    "waits for all and returns status for each.",
                ),
            ],
            "docker_deploy": [
                ("user", "How do I set up a multi-stage Docker build?"),
                (
                    "assistant",
                    "Use multiple FROM statements. First stage builds, second stage "
                    "copies only the artifact. Reduces image size significantly.",
                ),
            ],
            "sql_optimization": [
                ("user", "How can I optimize a slow SQL query with JOINs?"),
                (
                    "assistant",
                    "Add indexes on JOIN columns, use EXPLAIN to analyze the query "
                    "plan, avoid SELECT *, and consider denormalizing for read-heavy workloads.",
                ),
            ],
            "react_state": [
                ("user", "When should I use useState vs useReducer in React?"),
                (
                    "assistant",
                    "Use useState for simple state. Use useReducer when state logic "
                    "is complex, involves multiple sub-values, or the next state depends "
                    "on the previous one.",
                ),
            ],
        }

        for topic, msgs in topics.items():
            for role, content in msgs:
                ctx.add(role, content)

        test_cases = [
            ("sorting dictionaries python", "python_sorting"),
            ("Promise.all JavaScript", "javascript_async"),
            ("Docker multi-stage build", "docker_deploy"),
            ("SQL query optimization", "sql_optimization"),
            ("React state management", "react_state"),
        ]

        for query, expected_topic in test_cases:
            query_words = set(query.lower().split())

            t0 = time.perf_counter()
            scored = []
            for msg in ctx.messages:
                content = msg.get("content", "").lower()
                content_words = set(content.split())
                overlap = len(query_words & content_words)
                score = overlap / len(query_words) if query_words else 0
                if score > 0:
                    scored.append((score, msg))
            scored.sort(key=lambda x: x[0], reverse=True)
            search_ms = (time.perf_counter() - t0) * 1000

            found_relevant = False
            if scored:
                top_content = scored[0][1].get("content", "").lower()
                expected_keywords = expected_topic.replace("_", " ").split()
                found_relevant = any(kw in top_content for kw in expected_keywords)

            results.append(
                {
                    "query": query,
                    "expected_topic": expected_topic,
                    "found_relevant": found_relevant,
                    "top_score": round(scored[0][0], 3) if scored else 0,
                    "total_matches": len(scored),
                    "search_ms": round(search_ms, 3),
                }
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def bench_index_operations() -> list[dict[str, Any]]:
    """Benchmark vector index add / search / save / load at different scales."""
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="ctxkit_bench_")

    try:
        for size in [100, 500, 1000]:
            storage = os.path.join(tmp_dir, f"idx_{size}")
            idx = VectorIndex(storage_dir=storage)

            # Add
            t0 = time.perf_counter()
            for i in range(size):
                idx.add(f"msg_{i}", f"Message number {i} about topic {i % 10}")
            add_total_ms = (time.perf_counter() - t0) * 1000
            add_per_msg_ms = add_total_ms / size

            # Search (average of 10 queries)
            search_times = []
            for _ in range(10):
                t0 = time.perf_counter()
                idx.search("topic 5", top_k=10)
                search_times.append((time.perf_counter() - t0) * 1000)
            search_ms = sum(search_times) / len(search_times)

            # Save
            t0 = time.perf_counter()
            idx.save()
            save_ms = (time.perf_counter() - t0) * 1000

            # Load
            t0 = time.perf_counter()
            idx.load()
            load_ms = (time.perf_counter() - t0) * 1000

            results.append(
                {
                    "size": size,
                    "add_total_ms": round(add_total_ms, 1),
                    "add_per_msg_ms": round(add_per_msg_ms, 3),
                    "search_ms": round(search_ms, 3),
                    "save_ms": round(save_ms, 1),
                    "load_ms": round(load_ms, 1),
                }
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# ─── Output Formatting ────────────────────────────────────────────────────


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a clean, aligned table with title and separator."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    total_width = sum(widths) + 3 * (len(widths) - 1) + 4
    sep = "=" * total_width

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-+-'.join('-' * w for w in widths)}")

    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(f"  {line}")

    print()


# ─── Main ─────────────────────────────────────────────────────────────────


def run_benchmarks() -> None:
    """Run all benchmarks and display formatted result tables."""
    wall_start = time.perf_counter()

    print("\n" + "=" * 62)
    print("  ContextKit Benchmark Suite v0.2.0")
    print(f"  Python {sys.version.split()[0]}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # ── 1. Token Budget ──────────────────────────────────────────────
    print("\n[1/4] Token Counting & Budget")
    t0 = time.perf_counter()
    budget_results = bench_token_budget()
    wall_1 = (time.perf_counter() - t0) * 1000

    _print_table(
        "Token Counting Accuracy",
        ["Type", "Chars", "Tokens", "Chars/Tok", "Msg Tokens", "Budget OK", "Count time"],
        [
            [
                r["type"],
                f'{r["chars"]:,}',
                f'{r["tokens"]:,}',
                str(r["chars_per_token"]),
                f'{r["message_tokens"]:,}',
                "yes" if r["within_budget"] else "NO",
                f'{r["count_time_us"]}µs',
            ]
            for r in budget_results
        ],
    )
    print(f"  Benchmark completed in {wall_1:.1f}ms")

    # ── 2. Compression ───────────────────────────────────────────────
    print("\n[2/4] Context Compression (fallback summarizer — no API needed)")
    t0 = time.perf_counter()
    compression_results = bench_compression()
    wall_2 = (time.perf_counter() - t0) * 1000

    _print_table(
        "Compression Ratios by Context Size",
        ["Size", "Input (tok)", "Output (tok)", "Savings", "Msgs in→out", "Time"],
        [
            [
                r["size"],
                f'{r["original_tokens"]:,}',
                f'{r["compressed_tokens"]:,}',
                f'{r["savings_pct"]}%',
                f'{r["original_messages"]} → {r["compressed_messages"]}',
                f'{r["compress_ms"]:.1f}ms',
            ]
            for r in compression_results
        ],
    )
    avg_savings = sum(r["savings_pct"] for r in compression_results) / len(compression_results)
    print(f"  Average savings: {avg_savings:.1f}%  |  Benchmark: {wall_2:.1f}ms")

    # ── 3. Search Accuracy ───────────────────────────────────────────
    print("\n[3/4] Keyword Search Accuracy")
    t0 = time.perf_counter()
    search_results = bench_search_accuracy()
    wall_3 = (time.perf_counter() - t0) * 1000

    _print_table(
        "Keyword Search Results",
        ["Query", "Expected Topic", "Hit?", "Score", "Matches", "Latency"],
        [
            [
                r["query"][:38],
                r["expected_topic"],
                "yes" if r["found_relevant"] else "NO",
                str(r["top_score"]),
                str(r["total_matches"]),
                f'{r["search_ms"]:.2f}ms',
            ]
            for r in search_results
        ],
    )
    accuracy = (
        sum(1 for r in search_results if r["found_relevant"]) / len(search_results) * 100
    )
    print(f"  Overall accuracy: {accuracy:.0f}%  |  Benchmark: {wall_3:.1f}ms")

    # ── 4. Index Operations ──────────────────────────────────────────
    print("\n[4/4] Vector Index Operations")
    t0 = time.perf_counter()
    index_results = bench_index_operations()
    wall_4 = (time.perf_counter() - t0) * 1000

    _print_table(
        "Index Throughput & Latency",
        ["Messages", "Add total", "Add/msg", "Search (avg)", "Save", "Load"],
        [
            [
                f'{r["size"]:,}',
                f'{r["add_total_ms"]:.1f}ms',
                f'{r["add_per_msg_ms"]:.3f}ms',
                f'{r["search_ms"]:.3f}ms',
                f'{r["save_ms"]:.1f}ms',
                f'{r["load_ms"]:.1f}ms',
            ]
            for r in index_results
        ],
    )
    print(f"  Benchmark completed in {wall_4:.1f}ms")

    # ── Summary ──────────────────────────────────────────────────────
    wall_total = (time.perf_counter() - wall_start) * 1000
    print("\n" + "=" * 62)
    print("  Summary")
    print("=" * 62)
    print(f"  Avg compression savings : {avg_savings:.1f}%")
    print(f"  Search accuracy         : {accuracy:.0f}%")
    print(f"  Index add throughput    : {index_results[-1]['add_per_msg_ms']:.3f}ms / msg")
    print(f"  Index search latency    : {index_results[-1]['search_ms']:.3f}ms")
    print(f"  Total benchmark time    : {wall_total:.0f}ms")
    print("=" * 62)
    print()


if __name__ == "__main__":
    run_benchmarks()
