"""
Automated Scheduler and Background Watchdog.
Handles daily VPS health briefings, SQLite memory backups, and proactive alerts.
"""

import asyncio
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Callable
import logging

import config
from core.models import BackupResult, ScheduledReport, VPSMetrics
from core.vps_manager import (
    get_vps_health,
    list_docker_containers,
    get_docker_events_generator,
)
from core.memory import MemoryEngine

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    TELEGRAM_UI_AVAILABLE = True
except ImportError:
    TELEGRAM_UI_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_memory_backup(
    db_path: Path,
    backup_dir: Optional[Path] = None
) -> BackupResult:
    """
    Creates an atomic SQLite database backup snapshot using SQLite's backup API.
    Returns an immutable BackupResult instance.
    """
    if not db_path.exists():
        return BackupResult(
            success=False,
            db_path="",
            size_bytes=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            error=f"Source database does not exist: {db_path}"
        )

    target_dir = backup_dir or (db_path.parent / "backups")
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = target_dir / f"memory_backup_{timestamp_str}.db"

    try:
        # Atomic SQLite backup
        src_conn = sqlite3.connect(db_path)
        dst_conn = sqlite3.connect(backup_file)
        try:
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            src_conn.close()
            dst_conn.close()

        size = backup_file.stat().st_size
        logger.info(f"Database backup created successfully: {backup_file} ({size} bytes)")

        # Prune older local backups, keeping the last 7
        existing_backups = sorted(target_dir.glob("memory_backup_*.db"), key=lambda p: p.stat().st_mtime)
        while len(existing_backups) > 7:
            oldest = existing_backups.pop(0)
            try:
                oldest.unlink()
            except Exception as e:
                logger.warning(f"Could not remove old backup {oldest}: {e}")

        return BackupResult(
            success=True,
            db_path=str(backup_file),
            size_bytes=size,
            created_at=datetime.now(timezone.utc).isoformat(),
            error=None
        )

    except Exception as e:
        logger.error(f"Failed to create database backup: {e}", exc_info=True)
        return BackupResult(
            success=False,
            db_path="",
            size_bytes=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            error=str(e)
        )


def create_daily_health_report(
    vps_func: Callable[[], VPSMetrics] = get_vps_health,
    docker_func: Callable[[], str] = list_docker_containers,
    memory_engine: Optional[MemoryEngine] = None,
    user_id: Optional[str] = None
) -> ScheduledReport:
    """
    Synthesizes a daily morning briefing combining VPS hardware health,
    Docker container status, and memory statistics.
    """
    metrics = vps_func()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    text_parts = [
        f"☀️ **Good Morning! Daily VPS & Assistant Briefing**",
        f"📅 Date: `{now_str}`\n",
        f"⚙️ **System Resources:**",
        f"• CPU: `{metrics.cpu_percent}%` ({metrics.cpu_count} Cores)",
        f"• RAM: `{metrics.ram_used_gb:.1f} GB` / `{metrics.ram_total_gb:.1f} GB` ({metrics.ram_percent}%)",
        f"• Disk: `{metrics.disk_free_gb:.1f} GB free` ({metrics.disk_percent}% used)",
        f"• Uptime: `{metrics.uptime}`"
    ]

    if metrics.warning:
        text_parts.append(f"\n{metrics.warning}")

    # Docker status
    docker_status = docker_func()
    if "Active Docker Containers" in docker_status:
        text_parts.append(f"\n🐳 **Containers:**")
        # Extract container summary lines
        lines = [line.strip() for line in docker_status.split("\n") if line.strip().startswith(("🟢", "🔴"))]
        if lines:
            text_parts.extend(lines)
        else:
            text_parts.append("• All containers healthy.")
    else:
        text_parts.append("\n🐳 **Containers:** Docker monitoring active.")

    # Memory statistics if memory engine provided
    if memory_engine and user_id:
        rules_count = len(memory_engine.list_rules(user_id))
        facts_count = len(memory_engine.list_facts(user_id))
        text_parts.append(f"\n🧠 **Self-Learning Engine:**")
        text_parts.append(f"• Active Learned Rules: `{rules_count}`")
        text_parts.append(f"• Remembered Facts: `{facts_count}`")

    text_parts.append("\n_All systems functioning normally on Oracle Cloud VPS._")

    full_text = "\n".join(text_parts)
    return ScheduledReport(
        report_type="daily_health",
        text=full_text,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


class SchedulerService:
    """
    Background worker service that schedules recurring reports,
    performs periodic disk/watchdog checks, and transmits database backups.
    """
    def __init__(
        self,
        bot,
        settings: config.Settings,
        memory_engine: MemoryEngine,
        sleep_interval: int = 60
    ):
        self.bot = bot
        self.settings = settings
        self.memory = memory_engine
        self.sleep_interval = sleep_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._events_task: Optional[asyncio.Task] = None
        self._last_report_date: Optional[str] = None
        self._last_alert_time: Optional[datetime] = None
        self._container_alert_throttle: dict[str, datetime] = {}

    async def send_report(self, chat_id: int) -> bool:
        """Generates and delivers the daily executive briefing to a chat."""
        try:
            report = create_daily_health_report(
                memory_engine=self.memory,
                user_id=str(chat_id)
            )
            await self.bot.send_message(
                chat_id=chat_id,
                text=report.text,
                parse_mode="Markdown"
            )
            logger.info(f"Daily health report sent to chat {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}", exc_info=True)
            return False

    async def send_backup(self, chat_id: int) -> bool:
        """Creates a snapshot of memory.db and delivers it as an attachment."""
        try:
            backup_res = create_memory_backup(self.settings.db_path)
            if not backup_res.success:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Failed to create memory backup: `{backup_res.error}`"
                )
                return False

            backup_path = Path(backup_res.db_path)
            size_kb = backup_res.size_bytes / 1024
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")

            with open(backup_path, "rb") as doc_file:
                await self.bot.send_document(
                    chat_id=chat_id,
                    document=doc_file,
                    filename=backup_path.name,
                    caption=f"📦 **Automated Memory Database Backup**\n📅 Date: `{timestamp_str}`\n📊 Size: `{size_kb:.1f} KB`"
                )

            logger.info(f"Database backup document sent to chat {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send backup document: {e}", exc_info=True)
            return False

    async def _check_watchdog(self, chat_id: int):
        """Watches for critical issues (e.g. disk > 85%) and alerts proactively."""
        try:
            metrics = get_vps_health()
            if metrics.disk_percent > 85.0:
                now = datetime.now()
                # Throttle alerts to once every 6 hours
                if self._last_alert_time is None or (now - self._last_alert_time) > timedelta(hours=6):
                    self._last_alert_time = now
                    alert_msg = (
                        f"🚨 **CRITICAL VPS ALERT: DISK STORAGE NEAR CAPACITY**\n\n"
                        f"• Disk Usage: `{metrics.disk_percent}%` ({metrics.disk_free_gb:.1f} GB free)\n"
                        f"• Host RAM: `{metrics.ram_used_gb:.1f} GB` / `{metrics.ram_total_gb:.1f} GB`\n\n"
                        f"Please consider cleaning old downloads with `/mltb_logs` or prune unused docker images!"
                    )
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=alert_msg,
                        parse_mode="Markdown"
                    )
        except Exception as e:
            logger.debug(f"Watchdog check failed: {e}")

    async def _check_reminders(self):
        """Checks for due reminders and delivers notifications to users."""
        try:
            due_reminders = self.memory.get_due_reminders()
            for r in due_reminders:
                try:
                    chat_id = int(r.user_id)
                    alert_msg = (
                        f"⏰ **NHẮC VIỆC HẸN GIỜ:**\n\n"
                        f"📌 **Nhiệm vụ:** {r.task}\n"
                        f"⏱️ **Thời gian đặt:** `{r.remind_at}`\n\n"
                        f"_Chúc bạn hoàn thành tốt công việc!_"
                    )
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=alert_msg,
                        parse_mode="Markdown"
                    )
                    self.memory.mark_reminder_completed(r.id)
                    logger.info(f"Delivered reminder #{r.id} to user {chat_id}")
                except Exception as e:
                    logger.warning(f"Failed to deliver reminder #{r.id}: {e}")
        except Exception as e:
            logger.debug(f"Error checking due reminders: {e}")

    async def _loop(self):
        """Main background scheduling loop."""
        logger.info("Scheduler service loop started.")
        while self._running:
            try:
                chat_id = self.settings.get_admin_chat_id()
                if chat_id:
                    now = datetime.now()
                    today_str = now.strftime("%Y-%m-%d")

                    # 1. Daily Report Check (Fires at configured hour once per day)
                    if (
                        self.settings.enable_daily_report and
                        now.hour == self.settings.daily_report_hour and
                        self._last_report_date != today_str
                    ):
                        self._last_report_date = today_str
                        await self.send_report(chat_id)
                        if self.settings.enable_auto_backup:
                            await self.send_backup(chat_id)

                    # 2. Watchdog Check
                    await self._check_watchdog(chat_id)

                    # 3. Scheduled Reminders Check
                    await self._check_reminders()

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)

            await asyncio.sleep(self.sleep_interval)

    async def _listen_docker_events(self, chat_id: int):
        """Streams Docker container die/oom events and alerts Telegram in real-time."""
        events_gen = get_docker_events_generator()
        if not events_gen:
            return

        def _get_next_event():
            try:
                return next(events_gen)
            except Exception:
                return None

        while self._running:
            event = await asyncio.to_thread(_get_next_event)
            if not event or not self._running:
                await asyncio.sleep(2)
                continue

            try:
                actor = event.get("Actor", {})
                attrs = actor.get("Attributes", {})
                container_name = attrs.get("name") or "unknown"
                exit_code = attrs.get("exitCode", "unknown")
                action = event.get("Action", "die")

                # If container exited abnormally or OOM killed
                if exit_code != "0" or action == "oom":
                    now = datetime.now()
                    last_time = self._container_alert_throttle.get(container_name)
                    # Throttle duplicate alerts per container to 5 minutes
                    if last_time and (now - last_time) < timedelta(minutes=5):
                        continue

                    self._container_alert_throttle[container_name] = now
                    alert_text = (
                        f"🚨 **DOCKER CONTAINER CRASH ALERT!**\n\n"
                        f"• 📦 **Container:** `{container_name}`\n"
                        f"• ⚠️ **Action:** `{action}` | **Exit Code:** `{exit_code}`\n"
                        f"• ⏱️ **Thời gian:** `{now.strftime('%H:%M:%S %d/%m/%Y')}`\n\n"
                        f"Container đã bị dừng bất thường. Bấm nút bên dưới để khởi động lại ngay."
                    )

                    keyboard = None
                    if TELEGRAM_UI_AVAILABLE:
                        keyboard = InlineKeyboardMarkup([
                            [
                                InlineKeyboardButton(
                                    f"🔄 Restart {container_name[:16]}",
                                    callback_data=f"dk:r:{container_name}"
                                ),
                                InlineKeyboardButton(
                                    "📋 Logs",
                                    callback_data=f"dk:l:{container_name}"
                                ),
                            ]
                        ])

                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=alert_text,
                        reply_markup=keyboard,
                        parse_mode="Markdown"
                    )
            except Exception as e:
                logger.debug(f"Error handling docker event: {e}")

    def start(self) -> asyncio.Task:
        """Starts the scheduler in the current asyncio event loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop())
            chat_id = self.settings.get_admin_chat_id()
            if chat_id:
                self._events_task = asyncio.create_task(self._listen_docker_events(chat_id))
        return self._task

    def stop(self):
        """Stops the scheduler gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        if self._events_task and not self._events_task.done():
            self._events_task.cancel()
