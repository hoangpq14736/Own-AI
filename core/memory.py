import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
import logging

from core.models import LearnedRule, UserFact, ChatMessage
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)

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

    def add_fact(self, user_id: str, fact: str) -> int:
        """Saves a personal fact about the user."""
        if not user_id or not user_id.strip():
            raise ValidationError("user_id cannot be empty.")
        if not fact or not fact.strip():
            raise ValidationError("fact content cannot be empty.")

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_facts (user_id, fact, created_at) VALUES (?, ?, ?)",
                (user_id.strip(), fact.strip(), self._now_iso())
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

    def build_memory_system_prompt(self, user_id: str) -> str:
        """Constructs the prompt containing active learned rules and user facts."""
        rules = self.list_rules(user_id)
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
