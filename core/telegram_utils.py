"""
Telegram Utilities for Group Chat and Control Room Management.
Provides target filtering, mention stripping, and group authorization checks per ECC.
"""

import logging
import re
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


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
    is_reply_to_bot: bool = False,
    is_admin_chat: bool = False,
    enable_free_chat: bool = False,
) -> bool:
    """
    Determines if a message is intended for the bot:
    - In private chats: ALWAYS True.
    - In Admin / Control Room chat (is_admin_chat): ALWAYS True.
    - If free chat is enabled (enable_free_chat): ALWAYS True.
    - In group / supergroup chats: True IF:
      1. It is a direct reply to one of the bot's messages, OR
      2. The message text or caption explicitly mentions @bot_username.
    """
    # Private 1-on-1 chats are always targeted
    if chat_type in ("private", "sender"):
        return True

    # In Control Room group or when free chat mode is enabled
    if is_admin_chat or enable_free_chat:
        return True

    # In groups/channels when replying directly to the bot
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


def extract_mltb_command(
    text: Optional[str] = None,
    user_prompt: Optional[str] = None,
    has_reply_media: bool = False,
) -> Optional[str]:
    """
    Extracts or formats a raw MLTB command (/leech, /mirror, /clone, /ytdl)
    from AI response or natural language user prompt.
    Returns the exact command (e.g. '/leech https://...' or '/leech')
    without markdown code blocks or introductory conversational filler.
    """
    # 1. Check if AI response contains an MLTB command
    if text:
        match = re.search(r"/(leech|mirror|clone|ytdl)\b[^\n\r`]*", text, re.IGNORECASE)
        if match:
            cmd = match.group(0).strip()
            # Clean trailing backticks, quotes, or markdown
            cmd = re.sub(r"[`\"']+$", "", cmd).strip()
            return cmd

    # 2. Check if user prompt is asking to leech a replied media
    if user_prompt and has_reply_media:
        prompt_lower = user_prompt.lower()
        if re.search(r"\b(leech|tải|tai|down)\b", prompt_lower):
            return "/leech"
        if re.search(r"\b(mirror)\b", prompt_lower):
            return "/mirror"

    # 3. Check if user prompt contains a URL or magnet link with leech/mirror intent
    if user_prompt:
        url_match = re.search(r"(https?://\S+|magnet:\?\S+)", user_prompt)
        if url_match:
            url = url_match.group(1).rstrip(">)].,\"'")
            prompt_lower = user_prompt.lower()

            action = "/leech"
            if re.search(r"\b(mirror)\b", prompt_lower):
                action = "/mirror"
            elif re.search(r"\b(clone)\b", prompt_lower):
                action = "/clone"
            elif re.search(r"\b(ytdl|youtube)\b", prompt_lower):
                action = "/ytdl"

            # Check common flags
            flags = []
            if "-z" in user_prompt or re.search(r"\bzip\b", prompt_lower):
                flags.append("-z")
            if "-uz" in user_prompt or re.search(r"\bunzip\b", prompt_lower):
                flags.append("-uz")
            if "-s" in user_prompt or re.search(r"\bselect\b", prompt_lower):
                flags.append("-s")

            cmd = f"{action} {url}"
            if flags:
                cmd += " " + " ".join(flags)
            return cmd

    return None


async def send_safe_response(message: Any, text: str, parse_mode: Optional[str] = "Markdown") -> None:
    """
    Safely delivers text to Telegram:
    1. Splits messages longer than 4000 characters to prevent Telegram API crashes.
    2. Gracefully falls back to plain text if Markdown entities fail to parse.
    """
    if not text:
        text = "I received an empty response."

    chunk_size = 4000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        try:
            if parse_mode:
                await message.reply_text(chunk, parse_mode=parse_mode)
            else:
                await message.reply_text(chunk)
        except Exception:
            await message.reply_text(chunk)


async def safe_edit_text(message: Any, text: str, parse_mode: Optional[str] = "Markdown") -> Any:
    """
    Safely edits an existing Telegram message:
    1. Attempts to edit with the specified parse_mode.
    2. Gracefully falls back to plain text if Markdown entity parsing fails.
    """
    try:
        if parse_mode:
            return await message.edit_text(text, parse_mode=parse_mode)
        else:
            return await message.edit_text(text)
    except Exception as e:
        logger.warning(f"edit_text with parse_mode={parse_mode} failed ({e}); falling back to plain text.")
        try:
            return await message.edit_text(text)
        except Exception as inner_e:
            logger.error(f"Failed to edit message in plain text fallback: {inner_e}")
            raise inner_e

