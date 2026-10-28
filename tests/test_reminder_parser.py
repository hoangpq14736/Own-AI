import pytest
from datetime import datetime, timedelta
from core.reminder_parser import parse_reminder_text
from core.exceptions import ValidationError

def test_parse_relative_minutes():
    base_time = datetime(2026, 10, 28, 12, 0, 0)
    remind_at, task = parse_reminder_text("30m Kiểm tra tải file", current_time=base_time)
    assert task == "Kiểm tra tải file"
    assert remind_at == (base_time + timedelta(minutes=30)).isoformat()

def test_parse_relative_hours_and_mins():
    base_time = datetime(2026, 10, 28, 12, 0, 0)
    remind_at, task = parse_reminder_text("1h30m Đi họp dự án", current_time=base_time)
    assert task == "Đi họp dự án"
    assert remind_at == (base_time + timedelta(hours=1, minutes=30)).isoformat()

def test_parse_clock_time_today():
    base_time = datetime(2026, 10, 28, 12, 0, 0)
    remind_at, task = parse_reminder_text("18:30 Nấu cơm tối", current_time=base_time)
    assert task == "Nấu cơm tối"
    assert remind_at == datetime(2026, 10, 28, 18, 30, 0).isoformat()

def test_parse_clock_time_tomorrow_if_passed():
    base_time = datetime(2026, 10, 28, 20, 0, 0)
    remind_at, task = parse_reminder_text("08:00 Đi làm", current_time=base_time)
    assert task == "Đi làm"
    assert remind_at == datetime(2026, 10, 29, 8, 0, 0).isoformat()

def test_parse_specific_date():
    base_time = datetime(2026, 10, 28, 12, 0, 0)
    remind_at, task = parse_reminder_text("2026-11-05 14:00 Gặp đối tác", current_time=base_time)
    assert task == "Gặp đối tác"
    assert remind_at == datetime(2026, 11, 5, 14, 0, 0).isoformat()

def test_parse_invalid_text():
    with pytest.raises(ValidationError):
        parse_reminder_text("")

    with pytest.raises(ValidationError):
        parse_reminder_text("xin chào nhé")
