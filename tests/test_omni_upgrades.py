"""
PyTest suite for Omni-inspired upgrades:
- L1 Reflex Intent Router (sub-millisecond rule-based dispatching)
- Dynamic Tool Routing (domain-filtered Gemini Function Calling)
- L2 SQLite Memory WAL Mode & FTS5 Hybrid Search (BM25 + Semantic RRF)
- L3 Memory Consolidator (deduplication, rule supersession, chat pruning)

Pure-Python implementation compatible across Python 3.11 - 3.13 without
external runtime dependencies (no telegram, pydantic, or aiohttp required).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from core.agent_tools import (
    get_docker_containers,
    get_tools_for_route,
    get_vps_status,
    restart_docker_container,
)
from core.memory import MLTB_SYSTEM_KNOWLEDGE, MemoryEngine
from core.reflex_router import ReflexRoute, route_prompt


# ============================================================================
# 1. L1 Reflex Router Tests
# ============================================================================


def test_reflex_router_domains():
    """Verify sub-millisecond reflex routing across intent domains and boundary conditions."""
    # 1. Casual prompt -> no tools, no mltb
    casual_route = route_prompt("Hôm nay trời đẹp quá")
    assert casual_route.include_mltb is False
    assert casual_route.tool_domains == ()
    assert casual_route.use_all_tools is False

    # 2. MLTB prompt with link -> include_mltb=True
    mltb_route = route_prompt("tải video này https://youtube.com/watch?v=123 giúp tôi")
    assert mltb_route.include_mltb is True
    assert "mltb" in mltb_route.matched_intents

    # 3. VPS / Docker prompt -> vps_docker in tool_domains
    vps_route = route_prompt("Kiểm tra RAM và container Docker trên VPS")
    assert "vps_docker" in vps_route.tool_domains

    # 4. Notes / Reminders prompt -> notes_reminders in tool_domains
    notes_route = route_prompt("Nhắc tôi lúc 18h họp và lưu ghi chú")
    assert "notes_reminders" in notes_route.tool_domains

    # 5. 3D Printer prompt -> printer_3d in tool_domains
    printer_route = route_prompt("Kiểm tra nhiệt độ nozzle máy in Anycubic Kobra")
    assert "printer_3d" in printer_route.tool_domains

    # 6. Ambiguous / generic action prompt -> use_all_tools=True
    ambiguous_route = route_prompt("Kiểm tra giúp tôi")
    assert ambiguous_route.use_all_tools is True

    # 7. Empty and whitespace-only prompts -> empty ReflexRoute
    assert route_prompt("") == ReflexRoute()
    assert route_prompt("   \n\t  ") == ReflexRoute()

    # 8. Non-string inputs -> raises TypeError
    with pytest.raises(TypeError):
        route_prompt(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        route_prompt(12345)  # type: ignore[arg-type]


# ============================================================================
# 2. Agent Tools Routing Tests
# ============================================================================


def test_agent_tools_routing():
    """Verify that agent function tools are selectively filtered based on the ReflexRoute."""
    # Casual route: no tools attached
    casual_tools = get_tools_for_route(route_prompt("chào bạn"))
    assert casual_tools == []

    # Domain-specific route: only VPS/Docker tools attached
    vps_tools = get_tools_for_route(route_prompt("xem ram vps"))
    expected_vps_tools = [get_vps_status, get_docker_containers, restart_docker_container]
    assert vps_tools == expected_vps_tools

    # Generic action route: returns all 7 agent tools
    all_tools = get_tools_for_route(route_prompt("kiểm tra giúp tôi"))
    assert len(all_tools) == 7
    assert get_vps_status in all_tools
    assert get_docker_containers in all_tools
    assert restart_docker_container in all_tools


# ============================================================================
# 3. Memory WAL Mode and FTS5 Hybrid Search Tests
# ============================================================================


def test_memory_wal_and_fts5_hybrid(tmp_path: Path):
    """
    Test SQLite WAL mode, automatic FTS5 backfill from legacy un-indexed databases,
    safe token sanitization under special characters, and prompt synthesis filtering.
    """
    db_file = tmp_path / "test_memory.db"

    # Step 1: Create a legacy database without FTS5 tables to simulate migration
    legacy_conn = sqlite3.connect(db_file)
    legacy_conn.execute("""
        CREATE TABLE user_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            embedding TEXT,
            created_at TEXT NOT NULL
        )
    """)
    legacy_conn.execute("""
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            embedding TEXT,
            created_at TEXT NOT NULL
        )
    """)
    legacy_conn.execute("""
        CREATE TABLE learned_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            rule TEXT NOT NULL,
            trigger_context TEXT,
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
    """)
    legacy_conn.execute("""
        CREATE TABLE chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            media_type TEXT,
            created_at TEXT NOT NULL
        )
    """)
    legacy_conn.execute("""
        CREATE TABLE reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            task TEXT NOT NULL,
            remind_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_completed INTEGER NOT NULL DEFAULT 0
        )
    """)
    # Insert legacy rows
    legacy_conn.execute(
        "INSERT INTO user_facts (user_id, fact, created_at) VALUES ('u1', 'User owns Anycubic Kobra 2 Neo 3D printer', '2025-01-01T00:00:00Z')"
    )
    legacy_conn.execute(
        "INSERT INTO notes (user_id, text, tags, created_at) VALUES ('u1', 'Cấu hình Docker Compose cho MLTB service', 'docker,mltb', '2025-01-01T00:00:00Z')"
    )
    legacy_conn.commit()
    legacy_conn.close()

    # Step 2: Initialize MemoryEngine on legacy database
    engine = MemoryEngine(db_file)

    # Verify WAL journal mode
    with engine._connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.lower() == "wal"

    # Verify legacy rows were backfilled into FTS5 tables and are searchable
    fts_facts = engine.search_facts_hybrid("u1", query_text="Anycubic")
    assert any("Anycubic Kobra" in f for f in fts_facts)

    fts_notes = engine.search_notes("u1", query="Docker")
    assert any("Docker Compose" in n.text for n in fts_notes)

    # Step 3: Test search_facts_hybrid with special punctuation that usually breaks raw FTS5
    engine.add_fact("u1", "VPS/Oracle (RAM: 24GB) - CPU! High performance config")
    punct_results = engine.search_facts_hybrid("u1", query_text="VPS/Oracle? (RAM: 24GB) - CPU!")
    assert len(punct_results) > 0
    assert any("Oracle" in f and "24GB" in f for f in punct_results)

    # Step 4: Verify build_memory_system_prompt behavior
    prompt_no_mltb = engine.build_memory_system_prompt("u1", query_text="Kobra", include_mltb=False)
    assert "[MLTB KNOWLEDGE BASE]" not in prompt_no_mltb
    assert MLTB_SYSTEM_KNOWLEDGE.strip() not in prompt_no_mltb
    assert "Anycubic Kobra" in prompt_no_mltb

    prompt_with_mltb = engine.build_memory_system_prompt("u1", query_text="Kobra", include_mltb=True)
    assert "[MLTB KNOWLEDGE BASE]" in prompt_with_mltb


# ============================================================================
# 4. L3 Background Memory Consolidation Tests
# ============================================================================


def test_l3_consolidate_memory(tmp_path: Path):
    """
    Test L3 background memory consolidation:
    - Rule deduplication and contextual supersession
    - Fact deduplication preserving embeddings
    - Chat history pruning maintaining newest turns
    """
    db_file = tmp_path / "test_consolidate.db"
    engine = MemoryEngine(db_file)
    user_id = "test_user_consolidation"

    # --- 1. Rules Consolidation ---
    # Case A: Exact normalized text duplicates
    r1_id = engine.add_rule(user_id, "  Always reply concisely! ", category="tone")
    r2_id = engine.add_rule(user_id, "always reply concisely", category="tone")

    # Case B: Different text but overlapping specific context
    r3_id = engine.add_rule(
        user_id, "Log trace into file when error occurs", category="behavior", trigger_context="on error"
    )
    r4_id = engine.add_rule(
        user_id, "Notify admin immediately on failure", category="behavior", trigger_context="on error"
    )

    # --- 2. Facts Consolidation ---
    # Duplicate facts: one without embedding, one with embedding
    f1_id = engine.add_fact(user_id, "User prefers dark mode in all editors.")
    sample_embedding = [0.12, 0.45, 0.78]
    f2_id = engine.add_fact(user_id, "user prefers dark mode in all editors!", embedding=sample_embedding)

    # --- 3. Chat History Pruning ---
    # Add 10 chat messages
    for i in range(10):
        engine.add_history(
            user_id=user_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message turn #{i}",
        )

    # --- Execute Consolidation ---
    stats = engine.consolidate_memory(user_id=user_id, max_history_per_user=4)

    # Validate consolidation statistics
    assert stats["rules_superseded"] >= 2
    assert stats["facts_deduped"] >= 1
    assert stats["history_pruned"] == 6

    # Verify Rules State in Database
    with engine._connection() as conn:
        cursor = conn.cursor()

        # Older duplicate rule r1 must be superseded by r2
        cursor.execute("SELECT is_active, superseded_by FROM learned_rules WHERE id = ?", (r1_id,))
        row_r1 = cursor.fetchone()
        assert row_r1["is_active"] == 0
        assert row_r1["superseded_by"] == r2_id

        # Newer rule r2 remains active
        cursor.execute("SELECT is_active, superseded_by FROM learned_rules WHERE id = ?", (r2_id,))
        row_r2 = cursor.fetchone()
        assert row_r2["is_active"] == 1
        assert row_r2["superseded_by"] is None

        # Older context rule r3 must be superseded by newest rule r4
        cursor.execute("SELECT is_active, superseded_by FROM learned_rules WHERE id = ?", (r3_id,))
        row_r3 = cursor.fetchone()
        assert row_r3["is_active"] == 0
        assert row_r3["superseded_by"] == r4_id

        # Newer context rule r4 remains active
        cursor.execute("SELECT is_active, superseded_by FROM learned_rules WHERE id = ?", (r4_id,))
        row_r4 = cursor.fetchone()
        assert row_r4["is_active"] == 1

        # Verify Facts State: survivor is f2_id with embedding retained
        cursor.execute("SELECT id, fact, embedding FROM user_facts WHERE user_id = ?", (user_id,))
        surviving_facts = cursor.fetchall()
        assert len(surviving_facts) == 1
        survivor = surviving_facts[0]
        assert survivor["id"] == f2_id
        assert survivor["embedding"] is not None
        assert json.loads(survivor["embedding"]) == sample_embedding

        # Verify FTS5 facts table was updated accordingly
        cursor.execute("SELECT fact_id FROM user_facts_fts WHERE user_id = ?", (user_id,))
        fts_fact_ids = [r["fact_id"] for r in cursor.fetchall()]
        assert str(f2_id) in [str(fid) for fid in fts_fact_ids]
        assert str(f1_id) not in [str(fid) for fid in fts_fact_ids]

    # Verify History Pruning: only 4 newest entries remain in chronological order
    remaining_history = engine.get_recent_history(user_id=user_id, limit=20)
    assert len(remaining_history) == 4
    expected_contents = [f"Message turn #{i}" for i in range(6, 10)]
    assert [m.content for m in remaining_history] == expected_contents