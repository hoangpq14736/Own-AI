"""
Smart Reminder Time & Task Parser for Telegram.
Parses natural language relative time, clock time, and calendar schedules per ECC standards.
"""

from __future__ import annotations
import re
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional

from core.exceptions import ValidationError


def parse_reminder_text(text: str, current_time: Optional[datetime] = None) -> Tuple[str, str]:
    """
    Parses a reminder string into (remind_at_iso, task_description).
    Examples:
      - "30m Kiểm tra VPS"
      - "1h30m Đi họp"
      - "2 tiếng Đón con"
      - "18:30 Ăn tối"
      - "19h Gặp đối tác"
      - "2026-10-29 08:00 Gửi báo cáo"
    """
    if not text or not text.strip():
        raise ValidationError("Nội dung nhắc việc không được để trống.")

    cleaned = text.strip()
    now = current_time or datetime.now()

    # 1. Check for specific date + time: YYYY-MM-DD HH:MM
    date_time_match = re.match(
        r"^(\d{4}-\d{2}-\d{2})\s+(\d{1,2})[:h](\d{2})\s+(.+)$",
        cleaned,
        re.IGNORECASE
    )
    if date_time_match:
        date_str, hour_str, min_str, task = date_time_match.groups()
        try:
            target_dt = datetime.strptime(
                f"{date_str} {int(hour_str):02d}:{int(min_str):02d}",
                "%Y-%m-%d %H:%M"
            )
            return target_dt.isoformat(), task.strip()
        except ValueError as e:
            raise ValidationError(f"Định dạng ngày giờ không hợp lệ: {e}")

    # 2. Check for combined hours and minutes relative: e.g. 1h30m Task or 1h30p Task
    hm_match = re.match(
        r"^(\d+)\s*(?:h|giờ|tiếng)\s*(\d+)\s*(?:m|min|p|phút)\s+(.+)$",
        cleaned,
        re.IGNORECASE
    )
    if hm_match:
        h, m, task = hm_match.groups()
        target_dt = now + timedelta(hours=int(h), minutes=int(m))
        return target_dt.isoformat(), task.strip()

    # 3. Check for single relative units: minutes, hours, seconds, days
    rel_match = re.match(
        r"^(\d+)\s*(m|min|mins|phút|p|h|giờ|tiếng|s|giây|sec|d|ngày|day|days)\s+(.+)$",
        cleaned,
        re.IGNORECASE
    )
    if rel_match:
        val_str, unit, task = rel_match.groups()
        val = int(val_str)
        unit_lower = unit.lower()

        if unit_lower in ("m", "min", "mins", "phút", "p"):
            delta = timedelta(minutes=val)
        elif unit_lower in ("h", "giờ", "tiếng"):
            delta = timedelta(hours=val)
        elif unit_lower in ("s", "giây", "sec"):
            delta = timedelta(seconds=val)
        elif unit_lower in ("d", "ngày", "day", "days"):
            delta = timedelta(days=val)
        else:
            delta = timedelta(minutes=val)

        target_dt = now + delta
        return target_dt.isoformat(), task.strip()

    # 4. Check for clock time today: HH:MM or HHhMM (e.g. 18:30 or 18h30)
    clock_match = re.match(r"^(\d{1,2})[:h](\d{2})\s+(.+)$", cleaned, re.IGNORECASE)
    if clock_match:
        hour, minute, task = clock_match.groups()
        h, m = int(hour), int(minute)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValidationError("Giờ hoặc phút không hợp lệ (0-23h, 0-59m).")

        target_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        # If the time has already passed today, schedule for tomorrow
        if target_dt <= now:
            target_dt += timedelta(days=1)
        return target_dt.isoformat(), task.strip()

    # 5. Check for exact hour: e.g. 19h Task
    hour_match = re.match(r"^(\d{1,2})\s*h\s+(.+)$", cleaned, re.IGNORECASE)
    if hour_match:
        hour, task = hour_match.groups()
        h = int(hour)
        if not (0 <= h <= 23):
            raise ValidationError("Giờ không hợp lệ (0-23h).")

        target_dt = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if target_dt <= now:
            target_dt += timedelta(days=1)
        return target_dt.isoformat(), task.strip()

    raise ValidationError(
        "Không nhận diện được thời gian. Cú pháp hỗ trợ:\n"
        "• `/remind 30m Kiểm tra tải file`\n"
        "• `/remind 2h Đi họp`\n"
        "• `/remind 18:30 Nấu cơm`\n"
        "• `/remind 2026-10-29 09:00 Gặp đối tác`"
    )
