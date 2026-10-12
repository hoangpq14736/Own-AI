"""
Telegram Utilities for Group Chat and Control Room Management.
Provides target filtering, mention stripping, and group authorization checks per ECC.
"""

import re
from typing import Optional, Tuple


def strip_bot_mention(text: str, bot_username: Optional[str]) -> str:
    """
    Strips the bot's @username mention from prompt text or captions.
    E.g. '@MyAssistantBot prepare a leech link' -> 'prepare a leech link'
    """
    if not text:
        return ""
    if not bot_username:
        return text.strip()

    clean_username = bot_username.lstrip("@").strip()
    pattern = rf"@\b{re.escape(clean_username)}\b"
    stripped = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Clean up excess spaces
    return re.sub(r"\s+", " ", stripped).strip()


def is_message_targeted_at_bot(
    chat_type: str,
    text: Optional[str] = None,
    caption: Optional[str] = None,
    bot_username: Optional[str] = None,
    is_reply_to_bot: bool = False
) -> bool:
    """
    Determines if a message is intended for the bot:
    - In private chats: ALWAYS True.
    - In group / supergroup chats: True ONLY IF:
      1. It is a direct reply to one of the bot's messages, OR
      2. The message text or caption explicitly mentions @bot_username, OR
      3. It starts with a slash command directed at the bot.
    """
    # Private 1-on-1 chats are always targeted
    if chat_type in ("private", "sender"):
        return True

    # In groups/channels:
    if is_reply_to_bot:
        return True

    content = (text or "") + " " + (caption or "")
    content = content.strip()

    if not content:
        return False

    if bot_username:
        clean_user = bot_username.lstrip("@").lower()
        if f"@{clean_user}" in content.lower():
            return True

    return False


def can_user_execute_command(
    user_id: int,
    allowed_user_ids: Tuple[int, ...]
) -> bool:
    """
    Checks if a user is authorized to execute commands.
    If allowed_user_ids is empty, authorization is open.
    """
    if not allowed_user_ids:
        return True
    return user_id in allowed_user_ids
