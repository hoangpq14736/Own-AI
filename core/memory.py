import sqlite3
import json
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
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
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
                pass  # column already exists
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
                INSERT INTO learned_rules (user_id, category, rule, trigger_context, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
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

        emb_json = json.dumps(embedding) if embedding else None
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_facts (user_id, fact, embedding, created_at) VALUES (?, ?, ?, ?)",
                (user_id.strip(), fact.strip(), emb_json, self._now_iso())
            )
            conn.commit()
            return cursor.lastrowid

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
        query_embedding: Optional[List[float]] = None
    ) -> str:
        """Constructs the prompt containing active learned rules and user facts (optionally RAG filtered)."""
        rules = self.list_rules(user_id)
        if query_embedding:
            facts = self.search_facts_semantic(user_id, query_embedding, top_k=5)
            if not facts:
                facts = self.list_facts(user_id)[:5]
        else:
            facts = self.list_facts(user_id)

        sections = [
            PERSONAL_COMPANION_DIRECTIVE.strip(),
            MLTB_SYSTEM_KNOWLEDGE.strip(),
        ]

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

        emb_json = json.dumps(embedding) if embedding else None
        now_iso = self._now_iso()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes (user_id, text, tags, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id.strip(), text.strip(), tags.strip(), emb_json, now_iso))
            conn.commit()
            return cursor.lastrowid

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
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id.strip()))
            conn.commit()
            return cursor.rowcount > 0

    def search_notes(
        self,
        user_id: str,
        query: str = "",
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5
    ) -> List[Note]:
        """Searches notes by vector similarity if query_embedding is provided, or text substring."""
        if not user_id or not user_id.strip():
            return []

        # Vector semantic search
        if query_embedding:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, user_id, text, tags, embedding, created_at FROM notes WHERE user_id = ? AND embedding IS NOT NULL",
                    (user_id.strip(),)
                )
                rows = cursor.fetchall()

            scored = []
            for r in rows:
                try:
                    emb = json.loads(r["embedding"])
                    score = cosine_similarity(query_embedding, emb)
                    scored.append((score, r))
                except Exception:
                    continue

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                Note(
                    id=r["id"],
                    user_id=r["user_id"],
                    text=r["text"],
                    tags=r["tags"],
                    embedding=r["embedding"],
                    created_at=r["created_at"],
                )
                for score, r in scored[:top_k] if score > 0.3
            ]

        # Text fallback search
        q = f"%{query.strip()}%" if query.strip() else "%"
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, text, tags, embedding, created_at 
                FROM notes 
                WHERE user_id = ? AND (text LIKE ? OR tags LIKE ?) 
                ORDER BY id DESC LIMIT ?
                """,
                (user_id.strip(), q, q, top_k)
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

