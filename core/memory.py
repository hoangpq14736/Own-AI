import sqlite3
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
import logging

from core.models import LearnedRule, UserFact, ChatMessage, Reminder, Note
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vector lists."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def rrf_fuse(ranked_lists: List[List[Any]], k: int = 20) -> List[Any]:
    """
    Applies Reciprocal Rank Fusion (RRF) across multiple ranked lists.
    score(doc) = sum(1.0 / (k + rank)) (1-indexed).
    """
    scores: Dict[Any, float] = {}
    for r_list in ranked_lists:
        for rank, item in enumerate(r_list, start=1):
            scores[item] = scores.get(item, 0.0) + (1.0 / (k + rank))
    sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_items


def _sanitize_fts_query(query: str) -> str:
    """
    Extracts alphanumeric/Unicode word tokens and joins them with ' OR ' quoted safely.
    Prevents sqlite3.OperationalError on characters like '?', '-', ':', '/'.
    """
    if not query:
        return ""
    tokens = re.findall(r"\w+", query, re.UNICODE)
    if not tokens:
        return ""
    # Wrap each token in double quotes to prevent syntax collisions in FTS5
    return " OR ".join(f'"{token}"' for token in tokens)


MLTB_SYSTEM_KNOWLEDGE = """
[MLTB KNOWLEDGE BASE]
You are also the user's expert assistant for MLTB (Mirror-Leech-Telegram-Bot):
- `/mirror <link/magnet>`: Downloads file or torrent and uploads to Google Drive or Cloud.
- `/leech <link/magnet>`: Downloads file or torrent and uploads to Telegram (splits if > 2GB).
- `/clone <gdrive_link>`: Clones a Google Drive folder or file.
- `/ytdl <link>`: Downloads video/audio from YouTube or social media.
- Common flags:
  `-z`: Compress files into a .zip archive before upload.
  `-uz`: Unpack/unzip archive after download.
  `-s`: Select specific files from a multi-file torrent.
  `-d`: Provide archive password.
  `-b`: Bulk download multiple links.
  `-n <new_name>`: Rename file before upload.

CRITICAL INSTRUCTION FOR MLTB COMMAND DISPATCH:
When the user asks you to leech, mirror, clone, or ytdl a link or file:
Output ONLY the exact, raw command directly on the very first line (e.g. `/leech <link>` or `/mirror <link> [flags]`) so that MLTB can trigger immediately.
DO NOT prefix with conversational filler like "Here is your command:".
DO NOT wrap the command in markdown backticks or code blocks.
Simply output the raw command line.
[END MLTB KNOWLEDGE BASE]
"""

PERSONAL_COMPANION_DIRECTIVE = """
[PERSONAL COMPANION DIRECTIVE]
You are Own-AI, the user's private, trusted personal companion and assistant.
- Candid & Authentic: Speak naturally, openly, and candidly as a close peer and confidant. Match the user's conversational mood, slang, banter, or seriousness.
- No Moralizing or Lectures: NEVER scold, lecture, preach, or offer unsolicited ethical disclaimers or patronizing moral advice. The user is an adult conversing with their private, owned system.
- Relaxed & Unconstrained: Engage comfortably with dark humor, creative fiction, roleplay, mature themes, emotional venting, and uncensored opinions, while remaining factually truthful, helpful, and loyal.
[END PERSONAL COMPANION DIRECTIVE]
"""


class MemoryEngine:
    """
    Thread-safe SQLite engine managing learned rules, personal facts, and chat history.
    Enforces boundary validation and returns immutable domain models.
    Supports L2 Hybrid Retrieval (FTS5 BM25 + Vector RRF) and L3 Consolidation.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._fts5_enabled = False
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            if str(self.db_path) != ":memory:":
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                except sqlite3.OperationalError:
                    pass
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            yield conn
        finally:
            conn.close()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_db(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            # Rules learned from feedback and corrections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    rule TEXT NOT NULL,
                    trigger_context TEXT,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
            """)
            # Key personal facts about the user
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # Short-term chat history for continuity
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    media_type TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            # Scheduled reminders
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    remind_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_completed INTEGER NOT NULL DEFAULT 0
                )
            """)
            # Personal notes and bookmarks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    embedding TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Ensure embedding column exists in user_facts for semantic RAG
            try:
                cursor.execute("ALTER TABLE user_facts ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass

            # Safe Schema Evolution for learned_rules
            for col_def in [
                "hit_count INTEGER NOT NULL DEFAULT 0",
                "last_used_at TEXT",
                "superseded_by INTEGER"
            ]:
                try:
                    cursor.execute(f"ALTER TABLE learned_rules ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass

            # Check and initialize FTS5
            try:
                cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(col)")
                cursor.execute("DROP TABLE _fts5_test")
                self._fts5_enabled = True
            except sqlite3.OperationalError:
                self._fts5_enabled = False

            if self._fts5_enabled:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS user_facts_fts
                    USING fts5(fact, user_id UNINDEXED, fact_id UNINDEXED)
                """)
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
                    USING fts5(text, tags, user_id UNINDEXED, note_id UNINDEXED)
                """)

                # Backfill user_facts_fts if empty but user_facts has rows
                cursor.execute("SELECT COUNT(*) AS cnt FROM user_facts_fts")
                fts_facts_cnt = cursor.fetchone()["cnt"]
                cursor.execute("SELECT COUNT(*) AS cnt FROM user_facts")
                real_facts_cnt = cursor.fetchone()["cnt"]
                if fts_facts_cnt == 0 and real_facts_cnt > 0:
                    cursor.execute("""
                        INSERT INTO user_facts_fts (fact, user_id, fact_id)
                        SELECT fact, user_id, id FROM user_facts
                    """)

                # Backfill notes_fts if empty but notes has rows
                cursor.execute("SELECT COUNT(*) AS cnt FROM notes_fts")
                fts_notes_cnt = cursor.fetchone()["cnt"]
                cursor.execute("SELECT COUNT(*) AS cnt FROM notes")
                real_notes_cnt = cursor.fetchone()["cnt"]
                if fts_notes_cnt == 0 and real_notes_cnt > 0:
                    cursor.execute("""
                        INSERT INTO notes_fts (text, tags, user_id, note_id)
                        SELECT text, tags, user_id, id FROM notes
                    """)

            conn.commit()

    # --- Rules Management ---

    def add_rule(
        self,
        user_id: str,
        rule: str,
        category: str = "general",
        trigger_context: Optional[str] = None
    ) -> int:
        """Saves a new learned rule. Validates input and avoids exact duplicates."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty.")
        if not rule or not rule.strip():
            raise ValidationError("rule content cannot be empty.")

        rule_clean = rule.strip()
        uid = user_id.strip()

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM learned_rules WHERE user_id = ? AND rule = ? AND is_active = 1",
                (uid, rule_clean)
            )
            existing = cursor.fetchone()
            if existing:
                return existing["id"]

            cursor.execute(
                """
                INSERT INTO learned_rules (user_id, category, rule, trigger_context, created_at, is_active, hit_count)
                VALUES (?, ?, ?, ?, ?, 1, 0)
                """,
                (uid, category.strip() or "general", rule_clean, trigger_context, self._now_iso())
            )
            conn.commit()
            return cursor.lastrowid

    def list_rules(self, user_id: str) -> List[LearnedRule]:
        """Returns list of active learned rules as immutable LearnedRule objects."""
        if not user_id or not user_id.strip():
            return []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, category, rule, trigger_context, created_at, is_active 
                FROM learned_rules 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY id ASC
                """,
                (user_id.strip(),)
            )
            return [
                LearnedRule(
                    id=row["id"],
                    user_id=row["user_id"],
                    category=row["category"],
                    rule=row["rule"],
                    trigger_context=row["trigger_context"],
                    created_at=row["created_at"],
                    is_active=bool(row["is_active"])
                )
                for row in cursor.fetchall()
            ]

    def delete_rule(self, user_id: str, rule_id: int) -> bool:
        """Marks a learned rule as inactive."""
        if not user_id or not user_id.strip():
            return False

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE learned_rules SET is_active = 0 WHERE user_id = ? AND id = ?",
                (user_id.strip(), rule_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_rules(self, user_id: str) -> int:
        """Marks all rules for a user as inactive."""
        if not user_id or not user_id.strip():
            return 0

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE learned_rules SET is_active = 0 WHERE user_id = ? AND is_active = 1",
                (user_id.strip(),)
            )
            conn.commit()
            return cursor.rowcount

    # --- Facts Management ---

    def add_fact(self, user_id: str, fact: str, embedding: Optional[List[float]] = None) -> int:
        """Saves a personal fact about the user, optionally with embedding vector."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty.")
        if not fact or not fact.strip():
            raise ValidationError("fact content cannot be empty.")

        fact_clean = fact.strip()
        uid = user_id.strip()
        emb_json = json.dumps(embedding) if embedding else None
        now_iso = self._now_iso()

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_facts (user_id, fact, embedding, created_at) VALUES (?, ?, ?, ?)",
                (uid, fact_clean, emb_json, now_iso)
            )
            fact_id = cursor.lastrowid
            if self._fts5_enabled:
                cursor.execute(
                    "INSERT INTO user_facts_fts (fact, user_id, fact_id) VALUES (?, ?, ?)",
                    (fact_clean, uid, fact_id)
                )
            conn.commit()
            return fact_id

    def list_facts(self, user_id: str) -> List[str]:
        """Returns list of remembered fact strings for the user."""
        if not user_id or not user_id.strip():
            return []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fact FROM user_facts WHERE user_id = ? ORDER BY id ASC",
                (user_id.strip(),)
            )
            return [row["fact"] for row in cursor.fetchall()]

    def search_facts_semantic(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.4
    ) -> List[str]:
        """Returns the most semantically relevant facts using cosine similarity."""
        if not user_id or not user_id.strip() or not query_embedding:
            return []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fact, embedding FROM user_facts WHERE user_id = ? AND embedding IS NOT NULL",
                (user_id.strip(),)
            )
            rows = cursor.fetchall()

        scored_facts = []
        for r in rows:
            try:
                emb = json.loads(r["embedding"])
                score = cosine_similarity(query_embedding, emb)
                if score >= min_score:
                    scored_facts.append((score, r["fact"]))
            except Exception:
                continue

        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored_facts[:top_k]]

    def search_facts_hybrid(
        self,
        user_id: str,
        query_text: str = "",
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5
    ) -> List[str]:
        """
        Hybrid facts retrieval combining FTS5 BM25 and vector cosine similarity via RRF.
        Falls back to substring search if BM25/Vector results are empty.
        """
        uid = (user_id or "").strip()
        if not uid:
            return []

        ranked_bm25: List[str] = []
        clean_text = (query_text or "").strip()

        # BM25 search
        if clean_text and self._fts5_enabled:
            sanitized_fts = _sanitize_fts_query(clean_text)
            if sanitized_fts:
                with self._connection() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(
                            """
                            SELECT fact FROM user_facts_fts
                            WHERE user_facts_fts MATCH ? AND user_id = ?
                            ORDER BY bm25(user_facts_fts)
                            LIMIT ?
                            """,
                            (sanitized_fts, uid, top_k * 2)
                        )
                        ranked_bm25 = [row["fact"] for row in cursor.fetchall()]
                    except sqlite3.OperationalError:
                        ranked_bm25 = []

        # Vector search
        ranked_vector: List[str] = []
        if query_embedding:
            ranked_vector = self.search_facts_semantic(uid, query_embedding, top_k=top_k * 2, min_score=0.4)

        ranked_lists = []
        if ranked_bm25:
            ranked_lists.append(ranked_bm25)
        if ranked_vector:
            ranked_lists.append(ranked_vector)

        if ranked_lists:
            fused = rrf_fuse(ranked_lists, k=20)
            if fused:
                return fused[:top_k]

        # Case-insensitive substring fallback if query_text was provided
        if clean_text:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT fact FROM user_facts WHERE user_id = ? AND fact LIKE ? ORDER BY id DESC LIMIT ?",
                    (uid, f"%{clean_text}%", top_k)
                )
                fallback_facts = [row["fact"] for row in cursor.fetchall()]
                if fallback_facts:
                    return fallback_facts

        return []

    def list_facts_detailed(self, user_id: str) -> List[UserFact]:
        """Returns list of facts as immutable UserFact objects."""
        if not user_id or not user_id.strip():
            return []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, fact, created_at FROM user_facts WHERE user_id = ? ORDER BY id ASC",
                (user_id.strip(),)
            )
            return [
                UserFact(
                    id=row["id"],
                    user_id=row["user_id"],
                    fact=row["fact"],
                    created_at=row["created_at"]
                )
                for row in cursor.fetchall()
            ]

    # --- Chat History Management ---

    def add_history(
        self,
        user_id: str,
        role: str,
        content: str,
        media_type: Optional[str] = None
    ) -> None:
        """Records a user or assistant message to chat history."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty.")
        if not content or not content.strip():
            return

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_history (user_id, role, content, media_type, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id.strip(), role.strip(), content.strip(), media_type, self._now_iso())
            )
            conn.commit()

    def get_recent_history(self, user_id: str, limit: int = 10) -> List[ChatMessage]:
        """Returns the most recent conversation turns in chronological order."""
        if not user_id or not user_id.strip() or limit <= 0:
            return []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content, media_type, created_at 
                FROM chat_history 
                WHERE user_id = ? 
                ORDER BY id DESC LIMIT ?
                """,
                (user_id.strip(), limit)
            )
            rows = cursor.fetchall()
            return [
                ChatMessage(
                    role=r["role"],
                    content=r["content"],
                    media_type=r["media_type"],
                    created_at=r["created_at"]
                )
                for r in reversed(rows)
            ]

    def clear_history(self, user_id: str) -> int:
        """Clears short-term chat history for a user."""
        if not user_id or not user_id.strip():
            return 0

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id.strip(),))
            conn.commit()
            return cursor.rowcount

    # --- Prompt Synthesis ---

    def build_memory_system_prompt(
        self,
        user_id: str,
        query_embedding: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        include_mltb: bool = True,
        max_facts: int = 8
    ) -> str:
        """Constructs the prompt containing active learned rules and user facts (optionally hybrid RAG filtered)."""
        rules = self.list_rules(user_id)
        if query_embedding is not None or query_text is not None:
            facts = self.search_facts_hybrid(
                user_id,
                query_text=query_text or "",
                query_embedding=query_embedding,
                top_k=max_facts
            )
            if not facts:
                all_facts = self.list_facts(user_id)
                facts = all_facts[-max_facts:] if all_facts else []
        else:
            facts = self.list_facts(user_id)

        sections = [
            PERSONAL_COMPANION_DIRECTIVE.strip(),
        ]
        if include_mltb:
            sections.append(MLTB_SYSTEM_KNOWLEDGE.strip())

        if rules or facts:
            sections.append("[PERSISTENT MEMORY & SELF-LEARNED RULES]")
            sections.append("You have learned the following rules and facts from past interactions with this user. You MUST adhere to them strictly:")

            if facts:
                sections.append("\nKnown User Facts:")
                for f in facts:
                    sections.append(f"- {f}")

            if rules:
                sections.append("\nActive Learned Rules:")
                for r in rules:
                    trigger = f" (Context: {r.trigger_context})" if r.trigger_context else ""
                    sections.append(f"- [{r.category.upper()}] {r.rule}{trigger}")

            sections.append("[END PERSISTENT MEMORY]\n")

        return "\n".join(sections)

    # --- Reminders Management ---

    def add_reminder(
        self,
        user_id: str,
        task: str,
        remind_at: str
    ) -> int:
        """Saves a new scheduled reminder."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty.")
        if not task or not task.strip():
            raise ValidationError("task cannot be empty.")
        if not remind_at or not remind_at.strip():
            raise ValidationError("remind_at cannot be empty.")

        now_iso = self._now_iso()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reminders (user_id, task, remind_at, created_at, is_completed)
                VALUES (?, ?, ?, ?, 0)
            """, (user_id.strip(), task.strip(), remind_at.strip(), now_iso))
            conn.commit()
            return cursor.lastrowid

    def get_due_reminders(self) -> List[Reminder]:
        """Fetches all active reminders whose scheduled time is now or in the past."""
        now_iso = self._now_iso()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, task, remind_at, created_at, is_completed
                FROM reminders
                WHERE is_completed = 0 AND remind_at <= ?
                ORDER BY remind_at ASC
            """, (now_iso,))
            rows = cursor.fetchall()
            return [
                Reminder(
                    id=row["id"],
                    user_id=row["user_id"],
                    task=row["task"],
                    remind_at=row["remind_at"],
                    created_at=row["created_at"],
                    is_completed=bool(row["is_completed"])
                )
                for row in rows
            ]

    def mark_reminder_completed(self, reminder_id: int) -> bool:
        """Marks a reminder as completed."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE reminders SET is_completed = 1 WHERE id = ?
            """, (reminder_id,))
            conn.commit()
            return cursor.rowcount > 0

    def list_active_reminders(self, user_id: str) -> List[Reminder]:
        """Lists pending/active reminders for a specific user."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, task, remind_at, created_at, is_completed
                FROM reminders
                WHERE user_id = ? AND is_completed = 0
                ORDER BY remind_at ASC
            """, (user_id,))
            rows = cursor.fetchall()
            return [
                Reminder(
                    id=row["id"],
                    user_id=row["user_id"],
                    task=row["task"],
                    remind_at=row["remind_at"],
                    created_at=row["created_at"],
                    is_completed=bool(row["is_completed"])
                )
                for row in rows
            ]

    def delete_reminder(self, reminder_id: int, user_id: str) -> bool:
        """Deletes or cancels a reminder for a specific user."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM reminders WHERE id = ? AND user_id = ?
            """, (reminder_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    # --- Notes Management ---

    def add_note(
        self,
        user_id: str,
        text: str,
        tags: str = "",
        embedding: Optional[List[float]] = None
    ) -> int:
        """Saves a new note or bookmark with optional tags and vector embedding."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty.")
        if not text or not text.strip():
            raise ValidationError("note text cannot be empty.")

        text_clean = text.strip()
        tags_clean = tags.strip()
        uid = user_id.strip()
        emb_json = json.dumps(embedding) if embedding else None
        now_iso = self._now_iso()

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes (user_id, text, tags, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (uid, text_clean, tags_clean, emb_json, now_iso))
            note_id = cursor.lastrowid
            if self._fts5_enabled:
                cursor.execute("""
                    INSERT INTO notes_fts (text, tags, user_id, note_id)
                    VALUES (?, ?, ?, ?)
                """, (text_clean, tags_clean, uid, note_id))
            conn.commit()
            return note_id

    def list_notes(
        self,
        user_id: str,
        tag: Optional[str] = None,
        limit: int = 20
    ) -> List[Note]:
        """Lists notes for a specific user, optionally filtered by tag."""
        if not user_id or not user_id.strip():
            return []

        query = "SELECT id, user_id, text, tags, embedding, created_at FROM notes WHERE user_id = ?"
        params: list = [user_id.strip()]

        if tag and tag.strip():
            query += " AND tags LIKE ?"
            params.append(f"%{tag.strip()}%")

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [
                Note(
                    id=row["id"],
                    user_id=row["user_id"],
                    text=row["text"],
                    tags=row["tags"],
                    embedding=row["embedding"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def get_note(self, note_id: int, user_id: str) -> Optional[Note]:
        """Fetches a specific note by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, text, tags, embedding, created_at FROM notes WHERE id = ? AND user_id = ?",
                (note_id, user_id.strip())
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Note(
                id=row["id"],
                user_id=row["user_id"],
                text=row["text"],
                tags=row["tags"],
                embedding=row["embedding"],
                created_at=row["created_at"],
            )

    def delete_note(self, note_id: int, user_id: str) -> bool:
        """Deletes a note for a specific user."""
        uid = user_id.strip()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, uid))
            deleted = cursor.rowcount > 0
            if deleted and self._fts5_enabled:
                cursor.execute("DELETE FROM notes_fts WHERE note_id = ? AND user_id = ?", (note_id, uid))
            conn.commit()
            return deleted

    def search_notes(
        self,
        user_id: str,
        query: str = "",
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5
    ) -> List[Note]:
        """
        Hybrid search for notes using FTS5 BM25 and vector cosine similarity fused via RRF.
        Falls back to LIKE queries when FTS5 produces no match (e.g. substrings).
        """
        uid = (user_id or "").strip()
        if not uid:
            return []

        ranked_bm25_ids: List[int] = []
        clean_query = (query or "").strip()

        # 1. BM25 search via FTS5
        if clean_query and self._fts5_enabled:
            sanitized = _sanitize_fts_query(clean_query)
            if sanitized:
                with self._connection() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(
                            """
                            SELECT note_id FROM notes_fts
                            WHERE notes_fts MATCH ? AND user_id = ?
                            ORDER BY bm25(notes_fts)
                            LIMIT ?
                            """,
                            (sanitized, uid, top_k * 2)
                        )
                        ranked_bm25_ids = [int(r["note_id"]) for r in cursor.fetchall() if r["note_id"] is not None]
                    except sqlite3.OperationalError:
                        ranked_bm25_ids = []

        # 2. Vector search
        ranked_vector_ids: List[int] = []
        if query_embedding:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, embedding FROM notes WHERE user_id = ? AND embedding IS NOT NULL",
                    (uid,)
                )
                rows = cursor.fetchall()

            scored = []
            for r in rows:
                try:
                    emb = json.loads(r["embedding"])
                    score = cosine_similarity(query_embedding, emb)
                    if score > 0.3:
                        scored.append((score, int(r["id"])))
                except Exception:
                    continue

            scored.sort(key=lambda x: x[0], reverse=True)
            ranked_vector_ids = [nid for _, nid in scored[:top_k * 2]]

        # 3. Fuse via RRF if either list has elements
        ranked_lists = []
        if ranked_bm25_ids:
            ranked_lists.append(ranked_bm25_ids)
        if ranked_vector_ids:
            ranked_lists.append(ranked_vector_ids)

        if ranked_lists:
            fused_ids = rrf_fuse(ranked_lists, k=20)[:top_k]
            if fused_ids:
                with self._connection() as conn:
                    cursor = conn.cursor()
                    placeholders = ",".join("?" for _ in fused_ids)
                    cursor.execute(
                        f"""
                        SELECT id, user_id, text, tags, embedding, created_at 
                        FROM notes 
                        WHERE id IN ({placeholders}) AND user_id = ?
                        """,
                        (*fused_ids, uid)
                    )
                    rows_by_id = {row["id"]: row for row in cursor.fetchall()}
                    return [
                        Note(
                            id=rows_by_id[nid]["id"],
                            user_id=rows_by_id[nid]["user_id"],
                            text=rows_by_id[nid]["text"],
                            tags=rows_by_id[nid]["tags"],
                            embedding=rows_by_id[nid]["embedding"],
                            created_at=rows_by_id[nid]["created_at"],
                        )
                        for nid in fused_ids if nid in rows_by_id
                    ]

        # 4. Text fallback search using LIKE query
        if query_embedding is not None and not clean_query:
            return []
        q = f"%{clean_query}%" if clean_query else "%"
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, text, tags, embedding, created_at 
                FROM notes 
                WHERE user_id = ? AND (text LIKE ? OR tags LIKE ?) 
                ORDER BY id DESC LIMIT ?
                """,
                (uid, q, q, top_k)
            )
            rows = cursor.fetchall()
            return [
                Note(
                    id=row["id"],
                    user_id=row["user_id"],
                    text=row["text"],
                    tags=row["tags"],
                    embedding=row["embedding"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    # --- L3 Background Consolidator ---

    def consolidate_memory(
        self,
        user_id: Optional[str] = None,
        max_history_per_user: int = 200
    ) -> Dict[str, int]:
        """
        Deduplicates learned rules and facts, marks superseded rules, and prunes chat history.
        Normalizes text: " ".join(s.lower().strip().rstrip(".!?").split())
        """
        def _normalize(s: str) -> str:
            if not s:
                return ""
            return " ".join(s.lower().strip().rstrip(".!?").split())

        rules_superseded = 0
        facts_deduped = 0
        history_pruned = 0

        target_uid = user_id.strip() if user_id and user_id.strip() else None

        with self._connection() as conn:
            cursor = conn.cursor()

            # 1. Consolidate learned_rules
            rule_query = "SELECT id, user_id, category, rule, trigger_context FROM learned_rules WHERE is_active = 1"
            params: list = []
            if target_uid:
                rule_query += " AND user_id = ?"
                params.append(target_uid)
            rule_query += " ORDER BY id ASC"

            cursor.execute(rule_query, tuple(params))
            active_rules = cursor.fetchall()

            # Group rules by user_id
            rules_by_user: Dict[str, List[sqlite3.Row]] = {}
            for r in active_rules:
                rules_by_user.setdefault(r["user_id"], []).append(r)

            for u_id, r_list in rules_by_user.items():
                seen_texts: Dict[str, int] = {}
                seen_contexts: Dict[tuple, int] = {}

                # Traverse from newest to oldest
                for r in reversed(r_list):
                    rid = r["id"]
                    norm_text = _normalize(r["rule"])
                    cat = _normalize(r["category"])
                    trig = _normalize(r["trigger_context"] or "")

                    # Check duplicate text
                    if norm_text in seen_texts:
                        newest_id = seen_texts[norm_text]
                        cursor.execute(
                            "UPDATE learned_rules SET is_active = 0, superseded_by = ? WHERE id = ?",
                            (newest_id, rid)
                        )
                        rules_superseded += 1
                        continue
                    else:
                        seen_texts[norm_text] = rid

                    # Check duplicate (category, trigger_context) if specific
                    if trig and trig not in ("general",):
                        ctx_key = (cat, trig)
                        if ctx_key in seen_contexts:
                            newest_id = seen_contexts[ctx_key]
                            cursor.execute(
                                "UPDATE learned_rules SET is_active = 0, superseded_by = ? WHERE id = ?",
                                (newest_id, rid)
                            )
                            rules_superseded += 1
                            continue
                        else:
                            seen_contexts[ctx_key] = rid

            # 2. Consolidate user_facts
            facts_query = "SELECT id, user_id, fact, embedding FROM user_facts"
            f_params: list = []
            if target_uid:
                facts_query += " WHERE user_id = ?"
                f_params.append(target_uid)
            facts_query += " ORDER BY id ASC"

            cursor.execute(facts_query, tuple(f_params))
            all_facts = cursor.fetchall()

            facts_by_user: Dict[str, List[sqlite3.Row]] = {}
            for f in all_facts:
                facts_by_user.setdefault(f["user_id"], []).append(f)

            for u_id, f_list in facts_by_user.items():
                fact_groups: Dict[str, List[sqlite3.Row]] = {}
                for f in f_list:
                    norm = _normalize(f["fact"])
                    fact_groups.setdefault(norm, []).append(f)

                for norm, rows in fact_groups.items():
                    if len(rows) <= 1:
                        continue
                    # Keep row with embedding if possible, breaking ties by highest id
                    rows_sorted = sorted(
                        rows,
                        key=lambda r: (1 if r["embedding"] is not None else 0, r["id"]),
                        reverse=True
                    )
                    survivor = rows_sorted[0]
                    to_delete_ids = [r["id"] for r in rows_sorted[1:]]

                    for del_id in to_delete_ids:
                        cursor.execute("DELETE FROM user_facts WHERE id = ?", (del_id,))
                        if self._fts5_enabled:
                            cursor.execute("DELETE FROM user_facts_fts WHERE fact_id = ?", (del_id,))
                        facts_deduped += 1

            # 3. Prune chat_history
            if target_uid:
                user_ids = [target_uid]
            else:
                cursor.execute("SELECT DISTINCT user_id FROM chat_history")
                user_ids = [r["user_id"] for r in cursor.fetchall()]

            for u_id in user_ids:
                cursor.execute(
                    "SELECT id FROM chat_history WHERE user_id = ? ORDER BY id DESC",
                    (u_id,)
                )
                h_rows = cursor.fetchall()
                if len(h_rows) > max_history_per_user:
                    prune_ids = [r["id"] for r in h_rows[max_history_per_user:]]
                    placeholders = ",".join("?" for _ in prune_ids)
                    cursor.execute(
                        f"DELETE FROM chat_history WHERE id IN ({placeholders})",
                        prune_ids
                    )
                    history_pruned += cursor.rowcount

            conn.commit()

        return {
            "rules_superseded": rules_superseded,
            "facts_deduped": facts_deduped,
            "history_pruned": history_pruned,
        }