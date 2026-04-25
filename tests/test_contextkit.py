"""ContextKit test suite."""

import sys
import os
import json
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from contextkit import ContextManager
from contextkit.indexer import VectorIndex
from contextkit.budget import TokenBudget
from contextkit.compressor import ContextCompressor

TEST_DIR = "/tmp/contextkit_tests"


def clean():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_basic_add():
    clean()
    ctx = ContextManager(storage=f"{TEST_DIR}/t1", max_tokens=10000)
    ctx.add("system", "You are a coding assistant.")
    ctx.add("user", "写一个排序函数")
    ctx.add("assistant", "def sort(arr): return sorted(arr)")
    assert ctx.token_count > 0
    assert len(ctx.messages) == 3
    print("✅ test_basic_add")


def test_export_import():
    clean()
    ctx = ContextManager(storage=f"{TEST_DIR}/t2", max_tokens=10000)
    ctx.add("user", "hello")
    ctx.add("assistant", "hi there")
    ctx.export(f"{TEST_DIR}/export.json")
    ctx2 = ContextManager(storage=f"{TEST_DIR}/t2b", max_tokens=10000)
    n = ctx2.import_(f"{TEST_DIR}/export.json")
    assert len(ctx2.messages) == 2
    assert n == 2
    print("✅ test_export_import")


def test_get_recent():
    clean()
    ctx = ContextManager(storage=f"{TEST_DIR}/t3", max_tokens=10000)
    ctx.add("user", "msg1")
    ctx.add("user", "msg2")
    ctx.add("user", "msg3")
    recent = ctx.get_recent(max_tokens=5000)
    assert len(recent) == 3
    print("✅ test_get_recent")


def test_vector_index_degradation():
    idx = VectorIndex()
    idx.add("msg1", "sort list")
    idx.add("msg2", "cook pasta")
    results = idx.search("sorting", top_k=1)
    assert isinstance(results, list)
    print("✅ test_vector_index_degradation")


def test_token_budget():
    bm = TokenBudget(max_tokens=1000)
    tokens = bm.count_message_tokens("user", "hello " * 50)
    status = bm.budget_status(tokens)
    assert status["used"] > 0
    assert status["remaining"] < 1000
    print("✅ test_token_budget")


def test_compressor_fallback():
    comp = ContextCompressor()
    summary = comp._fallback_summary(
        [{"role": "user", "content": "a" * 500}, {"role": "assistant", "content": "b" * 500}]
    )
    assert len(summary) > 0
    assert len(summary) < 1000  # Should be compressed
    print("✅ test_compressor_fallback")


def test_persistence():
    clean()
    ctx = ContextManager(storage=f"{TEST_DIR}/t6", max_tokens=10000)
    ctx.add("user", "persistent message")
    ctx._save()
    ctx2 = ContextManager(storage=f"{TEST_DIR}/t6", max_tokens=10000)
    assert len(ctx2.messages) == 1
    assert ctx2.messages[0]["content"] == "persistent message"
    print("✅ test_persistence")


def test_auto_save():
    clean()
    ctx = ContextManager(storage=f"{TEST_DIR}/t7", max_tokens=10000)
    for i in range(20):
        ctx.add("user", f"message {i}")
    assert os.path.exists(f"{TEST_DIR}/t7/messages.json")
    print("✅ test_auto_save")


def test_token_budget_class():
    bm = TokenBudget(max_tokens=1000)
    assert bm.headroom(500) == 500
    assert not bm.is_near_limit(500)
    assert bm.is_near_limit(900)
    print("✅ test_token_budget_class")


if __name__ == "__main__":
    test_basic_add()
    test_export_import()
    test_get_recent()
    test_vector_index_degradation()
    test_token_budget()
    test_compressor_fallback()
    test_persistence()
    test_auto_save()
    test_token_budget_class()
    clean()
    print("\n🎉 All tests passed!")
