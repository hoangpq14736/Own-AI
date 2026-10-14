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

8. YTDL Options (via opt: or -opt):
   anasty17 MLTB accepts yt-dlp arguments formatted as "opt: key=val,key2=val2" or "opt: format=...".
   - Best Audio / Music:
     opt: format=bestaudio/best
   - Best Video / Highest Quality:
     opt: format=bv*+ba/b
   - 4K Resolution (2160p):
     opt: format=bv*[height<=2160]+ba/b
   - 2K Resolution (1440p):
     opt: format=bv*[height<=1440]+ba/b
   - 1080p Full HD:
     opt: format=bv*[height<=1080]+ba/b
   - 720p HD:
     opt: format=bv*[height<=720]+ba/b
   - 480p SD:
     opt: format=bv*[height<=480]+ba/b
   - MP3 Extraction:
     opt: format=bestaudio/best,postprocessors=[{'key':'FFmpegExtractAudio','preferredcodec':'mp3','preferredquality':'320'}]
   - Subtitles (Vietnamese / English):
     opt: writesubtitles=True,subtitleslangs=['vi','en'],embedsubtitles=True
   - Playlist Selection:
     opt: playlist_items=1-5
   - Single Video Only (ignore playlist):
     opt: noplaylist=True

RULES:
- If the user's message is general chat, programming questions, or non-MLTB conversation, set "is_mltb_intent" to false.
- Respect any user memory rules provided (e.g. default upload remote, preferred engines, naming rules).
- Destination: Default to /ytdlleech (Telegram) unless user specifically mentions cloud, drive, rclone, or mirror (/ytdl).
- Format & Quality Mapping:
  * "âm thanh tốt nhất", "tải nhạc", "nhạc tốt nhất", "audio tốt nhất", "best audio" -> use "opt: format=bestaudio/best"
  * "video tốt nhất", "định dạng video tốt nhất", "chất lượng cao nhất", "best video" -> use "opt: format=bv*+ba/b"
  * "định dạng 2k", "chất lượng 2k", "2k", "1440p" -> use "opt: format=bv*[height<=1440]+ba/b"
  * "định dạng 4k", "chất lượng 4k", "4k", "2160p" -> use "opt: format=bv*[height<=2160]+ba/b"
  * "1080p", "full hd" -> use "opt: format=bv*[height<=1080]+ba/b"
  * "720p", "hd" -> use "opt: format=bv*[height<=720]+ba/b"
- Return JSON ONLY with keys:
  "is_mltb_intent": bool,
  "command": base command (e.g. "/ytdlleech"),
  "fullCommand": complete command with URL and flags (e.g. "/ytdlleech https://... opt: format=bestaudio/best"),
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
