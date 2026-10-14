import os
import io
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from telegram import (
    Update,
    constants,
    BotCommand,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeAllGroupChats,
)
from telegram.error import TimedOut, NetworkError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import config
from core.memory import MemoryEngine
from core.gemini_client import GeminiClient
from core.reflector import Reflector
from core.media_generator import generate_google_image, generate_google_video, text_to_speech
from core.vps_manager import (
    get_vps_health,
    list_docker_containers,
    get_mltb_logs,
    restart_mltb_container,
)
from core.scheduler import SchedulerService, create_daily_health_report, create_memory_backup
from core.telegram_utils import (
    strip_bot_mention,
    is_message_targeted_at_bot,
    extract_mltb_command,
    send_safe_response,
    safe_edit_text,
    resolve_mltb_target,
)
from core.updater import (
    check_for_updates,
    apply_update,
    restart_bot_process,
    check_and_apply_update_on_startup,
    get_current_commit_info,
)
from core.userbot import UserbotClient
from core.mltb_brain import translate_natural_language_to_mltb

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize components
memory = MemoryEngine(config.DB_PATH)
gemini = GeminiClient(api_key=config.GEMINI_API_KEY, model_name=config.MODEL_NAME)
reflector = Reflector(gemini_client=gemini, memory_engine=memory)
userbot = UserbotClient(
    api_id=config.TELEGRAM_API_ID,
    api_hash=config.TELEGRAM_API_HASH,
    session_string=config.TELEGRAM_USER_SESSION,
)

def is_authorized(user_id: int) -> bool:
    if not config.ALLOWED_USER_IDS:
        return True
    return user_id in config.ALLOWED_USER_IDS

# --- Error Handler ---

async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handles network timeouts gracefully without dumping raw crash traces."""
    if isinstance(context.error, TimedOut):
        logger.warning("Telegram request timed out during upload, but media was likely delivered.")
    elif isinstance(context.error, NetworkError):
        logger.warning(f"Telegram network warning: {context.error}")
    else:
        logger.error(f"Telegram error: {context.error}", exc_info=context.error)

# --- Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
        return

    msg = (
        "⚡ **Welcome to your Multimodal AI Assistant & VPS Manager!**\n\n"
        "**What I Can Understand & Do:**\n"
        "• 💬 **Text** - High-speed chat & coding\n"
        "• 📸 **Photos** - Instant visual analysis (< 1.5s)\n"
        "• 🎙️ **Voice Notes** - I listen & reply with **voice**!\n"
        "• 🎬 **Videos & Docs** - Video comprehension & multi-page PDFs\n"
        "• 👥 **Group Chat Mode** - In Control Room, mention me (`@bot`) or reply to me!\n\n"
        "**Automated Reports & Backups:**\n"
        "• 📊 `/report` - Instant daily executive briefing (VPS + Docker + Memory)\n"
        "• 📦 `/backup` - Download snapshot of `memory.db`\n\n"
        "**Media Generation:**\n"
        "• 🎨 `/draw <prompt>` - Generate image with **Google Gemini**\n"
        "• 🎬 `/video <prompt>` - Generate video with **Google Veo 3.1**\n"
        "• 🗣️ `/speak <text>` - Speak in **Neural Voice**\n\n"
        "**VPS & MLTB Management:**\n"
        "• 📊 `/vps` - Real-time CPU, RAM (24GB), Disk, & Uptime\n"
        "• 🐳 `/docker` - List active containers on your VPS\n"
        "• 📋 `/mltb_logs` - Read last 25 lines of MLTB download logs\n"
        "• 🔄 `/mltb_restart` - Restart MLTB if downloads freeze\n\n"
        "**Auto-Updates & Maintenance:**\n"
        "• 🔍 `/check_update` - Check for new commits on GitHub\n"
        "• 🚀 `/update` - Pull latest updates & restart bot\n\n"
        "**Self-Learning Memory:**\n"
        "• `/rules` - View learned rules\n"
        "• `/delete_rule <id>` - Remove a rule\n"
        "• `/facts` - View remembered facts about you\n"
        "• `/clear` - Reset short-term conversation context"
    )
    await send_safe_response(update.message, msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start_command(update, context)

# --- Reporting & Backup Commands ---

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Triggers an on-demand daily health & supervisor briefing."""
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    report = create_daily_health_report(
        memory_engine=memory,
        user_id=str(update.effective_user.id)
    )
    await send_safe_response(update.message, report.text)

async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Creates an atomic SQLite memory snapshot and transmits it as a document."""
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    status_msg = await update.message.reply_text("📦 Creating database backup snapshot...")
    backup_res = create_memory_backup(config.DB_PATH)

    if not backup_res.success:
        await safe_edit_text(status_msg, f"❌ Backup failed: {backup_res.error}")
        return

    backup_path = Path(backup_res.db_path)
    size_kb = backup_res.size_bytes / 1024
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        with open(backup_path, "rb") as doc_file:
            await update.message.reply_document(
                document=doc_file,
                filename=backup_path.name,
                caption=f"📦 **Memory Database Backup Snapshot**\n📅 Date: `{now_str}`\n📊 Size: `{size_kb:.1f} KB`",
                read_timeout=60,
                write_timeout=60
            )
        await status_msg.delete()
    except Exception as e:
        logger.error(f"Failed to transmit backup file: {e}")
        await safe_edit_text(status_msg, f"❌ Failed to send backup document: {e}")

# --- VPS & MLTB Management Commands ---

async def vps_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    stats = get_vps_health()
    await send_safe_response(update.message, stats.text)

async def docker_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    text = list_docker_containers()
    await send_safe_response(update.message, text)

async def mltb_logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    lines = 25
    if context.args and context.args[0].isdigit():
        lines = min(int(context.args[0]), 100)

    text = get_mltb_logs(container_name="mirrorbot-app-1", lines=lines)
    await send_safe_response(update.message, text)

async def mltb_restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    status_msg = await update.message.reply_text("🔄 Sending restart signal to `mirrorbot-app-1`...")
    text = restart_mltb_container(container_name="mirrorbot-app-1")
    await safe_edit_text(status_msg, text)

# --- Repository Update & Maintenance Commands ---

async def check_update_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Checks for new commits on the configured GitHub repository branch."""
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    status_msg = await update.message.reply_text("🔍 Checking GitHub repository for updates...")
    info = check_for_updates(branch=config.GIT_BRANCH)

    if info.has_update:
        commits_preview = "\n".join(f"• `{c}`" for c in info.commits_behind[:5])
        if len(info.commits_behind) > 5:
            commits_preview += f"\n... and {len(info.commits_behind) - 5} more commit(s)"
        text = (
            f"🔄 *Update Available!*\n\n"
            f"• *Current Version:* `{info.current_commit}`\n"
            f"• *Latest Version:* `{info.latest_commit}`\n"
            f"• *Commit:* {info.latest_commit_message}\n"
            f"• *Commits Behind:* {info.behind_count}\n\n"
            f"*Recent Commits:*\n{commits_preview}\n\n"
            f"👉 Send `/update` to pull changes and restart the bot."
        )
    elif info.error:
        text = (
            f"⚠️ *Could not check for updates:*\n"
            f"{info.error}\n\n"
            f"• *Current version:* `{info.current_commit}`\n"
            f"• *Commit message:* {info.current_commit_message}"
        )
    else:
        text = (
            f"✅ *Bot is up to date!*\n\n"
            f"• *Running Commit:* `{info.current_commit}`\n"
            f"• *Message:* {info.current_commit_message}\n"
            f"• *Branch:* `{config.GIT_BRANCH}`"
        )

    await safe_edit_text(status_msg, text)

async def update_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pulls latest updates from GitHub and restarts the bot process."""
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    status_msg = await update.message.reply_text("🔍 Checking repository before updating...")
    info = check_for_updates(branch=config.GIT_BRANCH)

    if not info.has_update and not info.error:
        await safe_edit_text(
            status_msg,
            f"✅ Bot is already running the latest commit (`{info.current_commit}`). No update needed.",
        )
        return

    await safe_edit_text(status_msg, "📥 Pulling latest code and verifying dependencies...")
    success, msg = apply_update(branch=config.GIT_BRANCH)

    if not success:
        await safe_edit_text(
            status_msg,
            f"❌ *Update Failed:*\n{msg}\n\nPlease check git status or VPS permissions.",
        )
        return

    await safe_edit_text(
        status_msg,
        f"✅ *{msg}*\n\n🚀 Restarting bot now... I will be back online in a few seconds!",
    )
    # Pause briefly to ensure Telegram sends the reply before process replacement
    await asyncio.sleep(1.0)
    restart_bot_process()

# --- Userbot Status Command ---

async def userbot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reports Userbot helper status or explains how to configure it."""
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if userbot.is_connected:
        user_handle = f"@{userbot.username}" if userbot.username else f"User ID: `{userbot.user_id}`"
        target_info = f"`{config.settings.mltb_target_chat}`" if config.settings.mltb_target_chat else "Control Room group"
        text = (
            f"⚡ *Userbot is Connected & Active!*\n\n"
            f"• *Account:* {user_handle}\n"
            f"• *Status:* Online & ready to auto-dispatch MLTB commands\n"
            f"• *Dispatch Destination:* {target_info}\n\n"
            f"💡 *Tip:* You can chat with me in private, and I will automatically dispatch `/leech` or `/ytdl` to MLTB without double messages in the group!"
        )
    elif userbot.is_configured:
        text = (
            f"⚠️ *Userbot is configured but not connected.*\n\n"
            f"Please check your VPS logs or restart the bot: `/update`."
        )
    else:
        text = (
            f"ℹ️ *Userbot is not configured yet.*\n\n"
            f"To enable 100% hands-free MLTB dispatching:\n"
            f"1. Run `python scripts/generate_session.py` on your PC or VPS.\n"
            f"2. Add `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `TELEGRAM_USER_SESSION` to `.env`.\n"
            f"3. Optionally set `MLTB_TARGET_CHAT=@YourMLTBBot` to send directly to MLTB's inbox!\n"
            f"4. Restart the bot (`docker compose restart ai-supervisor`)!\n\n"
            f"*(In the meantime, I will give you 1-tap copy boxes for all leech commands).* "
        )
    await send_safe_response(update.message, text)

# --- Media Generation Commands ---

async def draw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if not context.args:
        await update.message.reply_text("Usage: `/draw <description of image>`\nExample: `/draw a golden origami phoenix rising from ashes, cinematic`")
        return

    prompt = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🎨 Generating image with Google Gemini:\n_\"{prompt}\"_...", parse_mode=constants.ParseMode.MARKDOWN)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.UPLOAD_PHOTO)

    img_bytes = await generate_google_image(prompt)
    if img_bytes:
        try:
            await status_msg.delete()
        except Exception:
            pass
        bio = io.BytesIO(img_bytes)
        bio.name = "google_image.jpg"
        try:
            await update.message.reply_photo(
                photo=bio,
                caption=f"✨ **Google Image:** {prompt}",
                read_timeout=60,
                write_timeout=60
            )
        except TimedOut:
            logger.info("Upload timed out on client acknowledgment, photo delivered successfully.")
    else:
        await safe_edit_text(status_msg, "❌ Failed to generate image with Google. Please try again.")

async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if not context.args:
        await update.message.reply_text("Usage: `/video <description of video>`\nExample: `/video a glowing jellyfish swimming in deep purple ocean, 4k`")
        return

    prompt = " ".join(context.args)
    status_msg = await update.message.reply_text(
        f"🎬 **Google Veo 3.1** is rendering your video...\n_\"{prompt}\"_\n\n⏳ This usually takes ~20–30 seconds, please hold on!",
        parse_mode=constants.ParseMode.MARKDOWN
    )
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.UPLOAD_VIDEO)

    video_path = await generate_google_video(prompt)
    if video_path and video_path.exists():
        try:
            await status_msg.delete()
        except Exception:
            pass
        try:
            with open(video_path, "rb") as f:
                await update.message.reply_video(
                    video=f,
                    caption=f"🎬 **Google Veo 3.1:** {prompt}",
                    read_timeout=120,
                    write_timeout=120
                )
        except TimedOut:
            logger.info("Video upload completed on Telegram server.")
        finally:
            try:
                video_path.unlink()
            except Exception:
                pass
    else:
        await safe_edit_text(status_msg, "❌ Failed to generate video with Google Veo. Please check your prompt or try again.")

async def speak_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        if update.effective_chat.type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    if not context.args:
        await update.message.reply_text("Usage: `/speak <text to speak>`")
        return

    text = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.RECORD_VOICE)

    audio_path = await text_to_speech(text)
    if audio_path and audio_path.exists():
        with open(audio_path, "rb") as f:
            await update.message.reply_voice(voice=f, read_timeout=60, write_timeout=60)
        try:
            audio_path.unlink()
        except Exception:
            pass
    else:
        await update.message.reply_text("❌ Could not generate voice audio.")

# --- Rules & Facts Management ---

async def list_rules_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    rules = memory.list_rules(user_id)

    if not rules:
        await update.message.reply_text(
            "📝 I haven't learned any rules from you yet.\n"
            "Tell me your preferences (e.g. *'Always use bullet points'*), and I will learn it automatically!"
        )
        return

    text = "🧠 **Learned Rules:**\n\n"
    for r in rules:
        trigger = f" *(Context: {r.trigger_context})*" if r.trigger_context else ""
        text += f"• `#{r.id}` [{r.category.upper()}] {r.rule}{trigger}\n"

    text += "\nTo remove a rule, type: `/delete_rule <id>`"
    await send_safe_response(update.message, text)

async def delete_rule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return

    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: `/delete_rule <rule_id>` (e.g., `/delete_rule 1`)")
        return

    rule_id = int(context.args[0])
    success = memory.delete_rule(user_id, rule_id)
    if success:
        await update.message.reply_text(f"✅ Rule `#{rule_id}` removed.")
    else:
        await update.message.reply_text(f"❌ Rule `#{rule_id}` not found.")

async def clear_rules_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return

    count = memory.clear_rules(user_id)
    await update.message.reply_text(f"🧹 Cleared {count} learned rules.")

async def list_facts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    facts = memory.list_facts(user_id)

    if not facts:
        await update.message.reply_text("📋 I don't have any saved facts about you yet.")
        return

    text = "📋 **Remembered Facts About You:**\n\n"
    for f in facts:
        text += f"• {f}\n"
    await send_safe_response(update.message, text)

async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return

    memory.clear_history(user_id)
    await update.message.reply_text("🔄 Short-term conversation history cleared. (Learned rules are still preserved!)")

# --- Background Self-Learning Worker (Smart 1-Request Filter) ---

async def background_reflection_task(bot, chat_id: int, user_id: str, prompt: str, assistant_response: str):
    try:
        learned_item = await reflector.reflect_and_learn(
            user_id=user_id,
            user_message=prompt,
            last_assistant_response=assistant_response
        )
        if learned_item and learned_item.should_learn:
            if learned_item.type == "rule":
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"💡 *Learned Rule #{learned_item.id}:* {learned_item.content}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
            elif learned_item.type == "fact":
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"💡 *Remembered Fact:* {learned_item.content}",
                    parse_mode=constants.ParseMode.MARKDOWN
                )
    except Exception as e:
        logger.debug(f"Background reflection error: {e}")

# --- Core Interaction Handler ---

async def handle_multimodal_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    media_path: Path = None,
    media_type: str = "text",
    prompt_text: str = "",
    reply_with_voice: bool = False
):
    user_id = str(update.effective_user.id)
    chat_type = update.effective_chat.type

    # 1. Authorization check
    if not is_authorized(update.effective_user.id):
        if chat_type == constants.ChatType.PRIVATE:
            await update.message.reply_text("⛔ Unauthorized.")
        return

    # 2. Control Room / Group Mentions Filter
    is_reply = bool(
        update.message.reply_to_message and
        update.message.reply_to_message.from_user and
        update.message.reply_to_message.from_user.id == context.bot.id
    )
    bot_username = context.bot.username
    is_admin_chat = bool(
        config.ADMIN_CHAT_ID is not None and
        update.effective_chat.id == config.ADMIN_CHAT_ID
    )

    # In groups: ignore noise unless message is targeted to the bot
    if not is_message_targeted_at_bot(
        chat_type=chat_type,
        text=prompt_text,
        caption=update.message.caption,
        bot_username=bot_username,
        is_reply_to_bot=is_reply,
        is_admin_chat=is_admin_chat,
        enable_free_chat=config.settings.enable_group_free_chat,
    ):
        return

    # Clean mention handle from user's prompt so Gemini receives clean text
    clean_prompt = strip_bot_mention(prompt_text, bot_username)

    # 2b. Direct MLTB Leech / Mirror Command Dispatch
    has_reply_media = bool(
        update.message.reply_to_message and
        (
            update.message.reply_to_message.photo or
            update.message.reply_to_message.document or
            update.message.reply_to_message.video or
            update.message.reply_to_message.audio or
            update.message.reply_to_message.voice
        )
    )

    # Lấy các quy tắc và sở thích đã học từ SQLite memory
    system_rules = memory.build_memory_system_prompt(user_id)

    # Gọi bộ não AI dịch ngôn ngữ tự nhiên sang lệnh MLTB
    direct_mltb, mltb_explanation = await translate_natural_language_to_mltb(
        gemini_client=gemini,
        user_prompt=clean_prompt,
        system_rules=system_rules,
        has_reply_media=has_reply_media,
    )

    if direct_mltb:
        logger.info(f"MLTB command synthesized via Brain: {direct_mltb} ({mltb_explanation})")
        memory.add_history(user_id=user_id, role="user", content=clean_prompt, media_type=media_type)
        memory.add_history(user_id=user_id, role="assistant", content=direct_mltb)

        target_chat = resolve_mltb_target(
            current_chat_id=update.effective_chat.id,
            chat_type=chat_type,
            mltb_target=config.settings.mltb_target_chat,
            admin_chat_id=config.settings.get_admin_chat_id(),
        )

        if userbot.is_connected and target_chat:
            target_reply_id = (
                update.message.reply_to_message.message_id
                if (has_reply_media and target_chat == update.effective_chat.id)
                else None
            )
            dispatched = await userbot.send_command(
                chat_id=target_chat,
                command=direct_mltb,
                reply_to_msg_id=target_reply_id,
            )
            if dispatched:
                dest_name = (
                    f"@{str(target_chat).lstrip('@')}"
                    if str(target_chat).startswith("@")
                    else "Control Room"
                )
                await update.message.reply_text(
                    f"⚡ *Auto-dispatched to MLTB ({dest_name}):*\n`{direct_mltb}`\n\n"
                    f"_{mltb_explanation}_\n*(Sent via your user account)*",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                return

        # Fallback nếu userbot tạm thời chưa kết nối
        await update.message.reply_text(
            f"📋 *Tap to copy for MLTB:*\n`{direct_mltb}`\n\n"
            f"_{mltb_explanation}_\n*(Telegram blocks bot-to-bot commands; tap above to copy & send, or check `/userbot`!)*",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
        return

    # Typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)

    # 3. Fetch learned rules & history
    system_rules = memory.build_memory_system_prompt(user_id)
    recent_history = memory.get_recent_history(user_id, limit=4)

    # 4. Get previous response for reflection
    last_assistant_msg = ""
    for turn in reversed(recent_history):
        if turn.role == "assistant":
            last_assistant_msg = turn.content
            break

    # 5. Fast Multimodal Call (Primary Request)
    effective_prompt = clean_prompt if clean_prompt else "Please analyze this file."
    response = await gemini.generate_multimodal(
        prompt=effective_prompt,
        media_path=media_path,
        system_instruction=system_rules,
        chat_history=recent_history
    )

    # 6. Save to chat history
    memory.add_history(user_id=user_id, role="user", content=effective_prompt, media_type=media_type)
    memory.add_history(user_id=user_id, role="assistant", content=response)

    # 7. Clean up downloaded local file
    if media_path and media_path.exists():
        try:
            media_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete local file {media_path}: {e}")

    # 8. Deliver conversational response
    await send_safe_response(update.message, response)

    # 9. Voice Reply if triggered by audio input
    if reply_with_voice:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.RECORD_VOICE)
            audio_reply_path = await text_to_speech(response)
            if audio_reply_path and audio_reply_path.exists():
                with open(audio_reply_path, "rb") as f:
                    await update.message.reply_voice(voice=f, read_timeout=60, write_timeout=60)
                audio_reply_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to send voice reply: {e}")

    # 10. Background Reflection (Pre-filtered: Only fires if learning signals detected)
    if reflector.should_trigger_reflection(effective_prompt):
        asyncio.create_task(
            background_reflection_task(
                bot=context.bot,
                chat_id=update.effective_chat.id,
                user_id=user_id,
                prompt=effective_prompt,
                assistant_response=last_assistant_msg
            )
        )

# --- Media Handlers ---

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    await handle_multimodal_message(update, context, prompt_text=text, media_type="text")

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    caption = update.message.caption or "What is in this picture?"
    file = await photo.get_file()
    dest = config.MEDIA_DIR / f"{file.file_unique_id}.jpg"
    await file.download_to_drive(dest)
    await handle_multimodal_message(update, context, media_path=dest, media_type="image", prompt_text=caption)

async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice or update.message.audio
    caption = update.message.caption or "Listen to this audio and respond appropriately."
    file = await voice.get_file()
    extension = ".oga" if update.message.voice else ".mp3"
    dest = config.MEDIA_DIR / f"{file.file_unique_id}{extension}"
    await file.download_to_drive(dest)
    await handle_multimodal_message(update, context, media_path=dest, media_type="audio", prompt_text=caption, reply_with_voice=True)

async def video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    video = update.message.video
    caption = update.message.caption or "Analyze this video clip and describe what is happening."
    file = await video.get_file()
    dest = config.MEDIA_DIR / f"{file.file_unique_id}.mp4"
    await file.download_to_drive(dest)
    await handle_multimodal_message(update, context, media_path=dest, media_type="video", prompt_text=caption)

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    caption = update.message.caption or "Analyze this document and summarize its key insights."
    file = await doc.get_file()
    suffix = Path(doc.file_name).suffix if doc.file_name else ".pdf"
    dest = config.MEDIA_DIR / f"{file.file_unique_id}{suffix}"
    await file.download_to_drive(dest)
    await handle_multimodal_message(update, context, media_path=dest, media_type="document", prompt_text=caption)

BOT_COMMANDS = [
    BotCommand("start", "Welcome message and overview"),
    BotCommand("help", "Show all available features"),
    BotCommand("report", "Daily executive briefing (VPS, Docker & Memory)"),
    BotCommand("backup", "Download snapshot backup of memory database"),
    BotCommand("vps", "Live CPU, 24GB RAM, Disk, & Uptime metrics"),
    BotCommand("docker", "List all running Docker containers"),
    BotCommand("mltb_logs", "Read recent download logs from MLTB"),
    BotCommand("mltb_restart", "Restart MLTB container if downloads freeze"),
    BotCommand("draw", "Generate image with Google Gemini"),
    BotCommand("video", "Generate cinematic video with Google Veo 3.1"),
    BotCommand("speak", "Speak text in bilingual neural voice"),
    BotCommand("rules", "View active learned rules and preferences"),
    BotCommand("delete_rule", "Remove a learned rule by ID"),
    BotCommand("clear_rules", "Clear all learned rules"),
    BotCommand("facts", "View remembered personal facts"),
    BotCommand("clear", "Reset short-term conversation context"),
    BotCommand("check_update", "Check for updates from GitHub repository"),
    BotCommand("update", "Pull latest updates from GitHub and restart"),
    BotCommand("userbot", "Check Userbot auto-dispatch status"),
]

async def post_init_hook(application):
    """Registers slash commands for both Private Chats and Group Chats, starts scheduler and userbot, and notifies on restart."""
    try:
        await application.bot.set_my_commands(BOT_COMMANDS, scope=BotCommandScopeAllPrivateChats())
        await application.bot.set_my_commands(BOT_COMMANDS, scope=BotCommandScopeAllGroupChats())
        logger.info("Successfully registered slash commands for both private and group chats.")
    except Exception as e:
        logger.warning(f"Failed to auto-register bot commands: {e}")

    logger.info("Initializing SchedulerService in event loop...")
    scheduler = SchedulerService(
        bot=application.bot,
        settings=config.settings,
        memory_engine=memory
    )
    scheduler.start()
    application.bot_data["scheduler"] = scheduler

    if userbot.is_configured:
        logger.info("Initializing Userbot client in background...")
        started = await userbot.start()
        if started:
            user_handle = f"@{userbot.username}" if userbot.username else f"User {userbot.user_id}"
            logger.info(f"Userbot auto-dispatch active as {user_handle}")

    # Notify admin if this startup was triggered by an applied update
    if os.environ.get("_ALREADY_UPDATED") == "1":
        os.environ.pop("_ALREADY_UPDATED", None)
        admin_id = config.settings.get_admin_chat_id()
        if admin_id:
            try:
                commit_hash, commit_msg = get_current_commit_info()
                await application.bot.send_message(
                    chat_id=admin_id,
                    text=(
                        f"🚀 **Bot restarted with latest updates!**\n\n"
                        f"📌 **Commit:** `{commit_hash}`\n"
                        f"📝 **Message:** _{commit_msg}_\n\n"
                        f"All systems are online and operational."
                    ),
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                logger.info("Sent update restart notification to admin chat.")
            except Exception as e:
                logger.warning(f"Failed to send update restart notification: {e}")

async def post_shutdown_hook(application):
    """Disconnects Userbot upon bot shutdown."""
    if userbot.is_connected:
        await userbot.stop()

def main():
    try:
        config.settings.validate_for_bot()
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return

    # Check for repository updates upon startup / restart
    if check_and_apply_update_on_startup(config.settings):
        # Process was replaced in-place via restart_bot_process()
        return

    print("🚀 Initializing Ultra-Fast Telegram Multimodal Assistant & VPS Manager...")
    app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .read_timeout(60.0)      # Generous timeout for uploading large media
        .write_timeout(60.0)     # Generous upload timeout
        .connect_timeout(30.0)
        .pool_timeout(30.0)
        .post_init(post_init_hook)
        .post_shutdown(post_shutdown_hook)
        .build()
    )

    # Register error handler
    app.add_error_handler(global_error_handler)

    # System & VPS commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("vps", vps_command))
    app.add_handler(CommandHandler("docker", docker_command))
    app.add_handler(CommandHandler("mltb_logs", mltb_logs_command))
    app.add_handler(CommandHandler("mltb_restart", mltb_restart_command))

    # Maintenance & Update commands
    app.add_handler(CommandHandler("check_update", check_update_command))
    app.add_handler(CommandHandler("update", update_command))
    app.add_handler(CommandHandler("userbot", userbot_command))

    # Reporting & Backup commands
    app.add_handler(CommandHandler("report", report_command))
    app.add_handler(CommandHandler("backup", backup_command))

    # Media generation commands
    app.add_handler(CommandHandler("draw", draw_command))
    app.add_handler(CommandHandler("image", draw_command))
    app.add_handler(CommandHandler("video", video_command))
    app.add_handler(CommandHandler("speak", speak_command))

    # Memory & Rules commands
    app.add_handler(CommandHandler("rules", list_rules_command))
    app.add_handler(CommandHandler("delete_rule", delete_rule_command))
    app.add_handler(CommandHandler("clear_rules", clear_rules_command))
    app.add_handler(CommandHandler("facts", list_facts_command))
    app.add_handler(CommandHandler("clear", clear_history_command))

    # Media and message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))
    app.add_handler(MessageHandler(filters.VIDEO, video_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))

    print(f"✅ Bot is listening! Active model: '{config.MODEL_NAME}'")
    print("📊 Automated daily reports (/report) & memory backups (/backup) enabled!")
    print("👥 Group Control Room mode active: Bot listens when mentioned or replied to.")
    print("🔄 Auto-updater active on startup / restart and commands /check_update & /update.")
    app.run_polling()

if __name__ == "__main__":
    main()
