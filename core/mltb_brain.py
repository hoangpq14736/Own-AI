"""
core/mltb_brain.py - 90%+ Feature MLTB Natural Language Synthesizer
Compatible with anasty17/mirror-leech-telegram-bot
"""
import json
import logging
from typing import Optional, Tuple
from google.genai import types

logger = logging.getLogger(__name__)

MLTB_SYSTEM_PROMPT = """
You are the MLTB AI Brain for anasty17/mirror-leech-telegram-bot.
Convert the user's conversational message (English, Vietnamese, or shorthand) into an exact, production-grade MLTB slash command with all required flags.

90%+ MLTB Capabilities:
1. Cloud Mirror:
   - /mirror (or /m): Direct link / torrent to Google Drive or Rclone
   - /zipmirror: Compress to .zip before cloud upload
   - /unzipmirror (or /mirror -uz): Extract archive (zip, rar, 7z, tar, iso) before upload
   - /qbmirror: Force qBittorrent engine for torrents
   - /qbzipmirror / /qbunzipmirror: qBit engine with zip / extract

2. Telegram Leech:
   - /leech (or /l): Direct link / torrent to Telegram chat/channel
   - /zipleech: Pack into .zip before Telegram upload
   - /unzipleech: Extract archives before Telegram upload
   - /qbleech: Force qBittorrent engine to Telegram
   - /qbzipleech / /qbunzipleech: qBit engine with zip / extract

3. Media Streams:
   - /ytdl: YouTube, TikTok, Twitter/X, SoundCloud to Cloud
   - /ytdlleech: Media download directly to Telegram
   - /ytdlzip / /ytdlzipleech: Zip media playlist before upload

4. Cloud Clones:
   - /clone: Google Drive clone
   - /count: Count Google Drive folder items & total size
   - /list: Search Google Drive or rclone remote
   - /delete: Delete Google Drive link

5. Task Controls:
   - /status: Active downloads, speeds, ETA
   - /cancel: Abort active task
   - /cancelall: Abort all tasks
   - /pause / /resume: Pause or resume torrent tasks

6. Utilities:
   - /speedtest: Ookla server bandwidth test
   - /stats: Server CPU, RAM, Disk usage
   - /search: Search torrent engines (1337x, Nyaa, YTS)

7. MLTB Flags:
   - -n <name>: Rename downloaded file or folder
   - -z: Zip compression
   - -uz or -e: Archive extraction
   - -s: Selective file downloading in torrents
   - -sd <ratio:time>: Seeding ratio and duration
   - -up <remote:path>: Custom rclone remote destination
   - -rcf <flags>: Custom rclone flags
   - -c: Upload as raw document instead of streaming video
   - -ss <time>: Screenshot / sample video generation
   - -b: Bulk download links
   - -au <user:pass>: HTTP/FTP authentication
   - -h <header:value>: Custom HTTP headers

RULES:
- If the user's message is general chat, programming questions, or non-MLTB conversation, set "is_mltb_intent" to false.
- Respect any user memory rules provided (e.g. default upload remote, preferred engines, naming rules).
- Return JSON ONLY with keys:
  "is_mltb_intent": bool,
  "command": base command (e.g. "/qbunzipmirror"),
  "fullCommand": complete command with URL and flags (e.g. "/qbunzipmirror magnet:... -n Film.mkv -up gdrive:Movies"),
  "explanation": concise 1-line reason in the user's language
"""

async def translate_natural_language_to_mltb(
    gemini_client,
    user_prompt: str,
    system_rules: str = "",
    has_reply_media: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Synthesizes an exact MLTB command using Gemini 3.8 Flash.
    Returns (full_command, explanation) or (None, None) if not an MLTB intent.
    """
    if not user_prompt:
        return None, None

    # Fast-path for replying to media with "leech" or "tải"
    if has_reply_media and any(k in user_prompt.lower() for k in ["leech", "tải", "down"]):
        flags = []
        if any(u in user_prompt.lower() for u in ["unzip", "giải nén", "extract"]):
            flags.append("-uz")
        elif any(z in user_prompt.lower() for z in ["zip", "nén"]):
            flags.append("-z")
        cmd = "/leech"
        if flags:
            cmd += " " + " ".join(flags)
        return cmd, "Leech replied media to Telegram"

    # Assemble system instruction including Own-AI's learned user rules
    instruction = MLTB_SYSTEM_PROMPT
    if system_rules:
        instruction += f"\n\nUSER LEARNED PREFERENCES & RULES:\n{system_rules}"

    try:
        response = await gemini_client.client.aio.models.generate_content(
            model="gemini-3.8-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                response_mime_type="application/json",
            )
        )
        data = json.loads(response.text)
        if data.get("is_mltb_intent") and data.get("fullCommand"):
            return data["fullCommand"], data.get("explanation", "")
    except Exception as e:
        logger.error(f"[MLTB Brain] Gemini parsing error: {e}")

    return None, None
