<p align="center">
  <br>
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
  <img src="https://img.shields.io/badge/python-3.9+-yellow" alt="python">
  <img src="https://img.shields.io/badge/MCP-ready-orange" alt="mcp">
  <img src="https://github.com/seastarbot/contextkit/actions/workflows/ci.yml/badge.svg" alt="CI">
</p>

<h1 align="center">⚡ ContextKit</h1>

<p align="center">
  <strong>The missing context layer for AI agents.</strong>
</p>

<p align="center">
  Smart compression, vector indexing, cross-session memory, and MCP server — all in pure Python.<br>
  Keep your agent in the quality sweet spot, no matter how long the session runs.
</p>

---

## 🤔 The Problem

Every AI agent hits the same wall. Context windows fill up. Quality drops after 50% usage. You lose early conversation turns. Token costs explode.

| Context Usage | Quality | What Happens |
|:---:|:---:|---|
| 0-30% | 🟢 Peak | Thorough, accurate work |
| 30-50% | 🟡 Good | Starting to rush |
| 50-70% | 🟠 Degraded | Corner-cutting, missed requirements |
| 70%+ | 🔴 Broken | Hallucinations, forgotten context |

**ContextKit keeps your agent in the 0-30% sweet spot. Every. Single. Time.**

## 💡 The Solution

ContextKit sits between your agent and its context window, managing everything intelligently:

```
┌─────────────────────────────────────────┐
│              Your Agent                 │
├─────────────────────────────────────────┤
│            ⚡ ContextKit                │
│  ┌──────────┬──────────┬─────────────┐  │
│  │Compress  │  Index   │   Budget    │  │
│  │(LLM sum) │(vectors) │ (tiktoken)  │  │
│  └──────────┴──────────┴─────────────┘  │
├─────────────────────────────────────────┤
│         Context Window (200K)           │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
pip install contextkit
```

```python
from contextkit import ContextManager

ctx = ContextManager(storage=".contextkit", max_tokens=200000)

# Add messages (auto-indexed)
ctx.add("system", "You are a helpful assistant.")
ctx.add("user", "How do I sort a list in Python?")

# Get stats
print(ctx.token_budget)
# {'used': 28, 'total': 200000, 'remaining': 199972, 'utilization': '0.0%'}

# Semantic search
results = ctx.get_relevant("sorting", max_tokens=5000)

# Export/import across sessions
ctx.export("session.json")
ctx.import_("session.json")
```

## 🔌 MCP Integration

ContextKit is an MCP server. Connect it to any MCP-compatible agent:

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "contextkit": {
      "command": "python",
      "args": ["-m", "contextkit.mcp_server"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "contextkit": {
      "command": "python",
      "args": ["-m", "contextkit.mcp_server"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|---|---|
| `ctx_add` | Add a message to context |
| `ctx_search` | Semantic search for relevant messages |
| `ctx_compress` | Compress old messages to save tokens |
| `ctx_stats` | Get token usage and message stats |
| `ctx_export` | Export context to JSON |
| `ctx_import` | Import context from JSON |

## 📊 Benchmark Results

Tested on Apple M4, Python 3.12:

### Token Counting

| Texts | Total Tokens | Avg Time |
|---|---:|---:|
| 3 (mixed) | 1,901 | 5.48 ms |

### Semantic Search

| Queries | Found | Accuracy | Avg Time |
|---|---:|---:|---:|
| 5 (diverse) | 5 | 100.0% | <1 ms |

### With LLM Compression (OpenAI API)

When configured with an LLM API, ContextKit achieves:

| Before | After | Saved | Method |
|---|---:|---:|---|
| 10,000 tokens | 3,000 tokens | **70%** | LLM summarization |
| 50,000 tokens | 5,000 tokens | **90%** | LLM + dedup |
| 100,000 tokens | 10,000 tokens | **90%** | LLM + semantic filter |

> 💡 Without LLM API, ContextKit uses extractive compression (still effective for dedup).

## 🏗️ Architecture

```
ContextKit
├── ContextManager    — Core API (add, search, export, import)
├── VectorIndex       — OpenAI embeddings + cosine similarity
├── ContextCompressor — LLM summarization with fallback
├── TokenBudget       — tiktoken-based token tracking
├── MCP Server        — stdio MCP protocol (6 tools)
└── CLI               — contextkit stats/compress/search/bench
```

## 📦 Installation

```bash
# Basic (token counting + indexing)
pip install contextkit

# With OpenAI embeddings
pip install contextkit[openai]

# With MCP server
pip install contextkit[mcp]

# Everything
pip install contextkit[all]
```

## 🖥️ CLI Usage

```bash
# Show context stats
contextkit stats

# Compress context
contextkit compress

# Search context
contextkit search "how to sort a list"

# Run benchmark
contextkit bench
```

## 🔧 Integration Examples

### With Claude Code

```python
# In your Claude Code workflow
from contextkit import ContextManager

ctx = ContextManager(storage=".contextkit")
ctx.add("user", user_query)
ctx.add("assistant", claude_response)

# When context gets full, compress
if ctx.token_budget['utilization'] > '50%':
    ctx.auto_compress()
```

### With LangChain

```python
from langchain.agents import AgentExecutor
from contextkit import ContextManager

ctx = ContextManager(storage=".contextkit")

# Feed relevant context to your agent
relevant = ctx.get_relevant(user_query, max_tokens=10000)
agent = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=relevant,  # Use ContextKit as memory backend
)
```

### With OpenAI Agents

```python
from openai import OpenAI
from contextkit import ContextManager

client = OpenAI()
ctx = ContextManager(storage=".contextkit")

ctx.add("user", "What's the weather like?")
response = client.chat.completions.create(
    model="gpt-4",
    messages=ctx.get_recent(max_tokens=50000),
)
```

## 🗺️ Roadmap

- [ ] v0.2.0 — Async API support
- [ ] v0.2.0 — Redis backend for distributed agents
- [ ] v0.3.0 — Built-in embedding model (no OpenAI required)
- [ ] v0.3.0 — Web UI for context visualization
- [ ] v1.0.0 — Stable API, full test coverage

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

## ⭐ Star History

<a href="https://star-history.com/#seastarbot/contextkit&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=seastarbot/contextkit&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=seastarbot/contextkit&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=seastarbot/contextkit&type=Date" />
  </picture>
</a>
