import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bot import (
    remind_command,
    list_reminders_command,
    delete_reminder_command,
    read_url_command,
    speedtest_command,
)
from core.models import Reminder
from core.web_reader import WebPageContent
from core.speedtest_manager import SpeedtestResult

@pytest.mark.asyncio
async def test_remind_command_unauthorized():
    mock_update = MagicMock()
    mock_update.effective_user.id = 999999
    mock_update.effective_chat.type = "private"
    mock_update.message.reply_text = AsyncMock()

    with patch("bot.is_authorized", return_value=False):
        await remind_command(mock_update, MagicMock())

    mock_update.message.reply_text.assert_called_once_with("⛔ Unauthorized.")

@pytest.mark.asyncio
async def test_remind_command_success():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_update.message.reply_text = AsyncMock()
    mock_context = MagicMock()
    mock_context.args = ["30m", "Kiểm", "tra", "tải", "file"]

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.memory.add_reminder", return_value=101):
        await remind_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    reply = mock_update.message.reply_text.call_args[0][0]
    assert "Đã đặt nhắc việc thành công (#101)" in reply
    assert "Kiểm tra tải file" in reply

@pytest.mark.asyncio
async def test_list_reminders_empty():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_update.message.reply_text = AsyncMock()

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.memory.list_active_reminders", return_value=[]):
        await list_reminders_command(mock_update, MagicMock())

    assert "không có lịch nhắc việc nào" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_list_reminders_with_items():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_update.message.reply_text = AsyncMock()
    fake_reminders = [
        Reminder(id=1, user_id="12345", task="Họp team", remind_at="2026-10-28T18:00:00")
    ]

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.memory.list_active_reminders", return_value=fake_reminders):
        await list_reminders_command(mock_update, MagicMock())

    reply = mock_update.message.reply_text.call_args[0][0]
    assert "#1" in reply
    assert "Họp team" in reply

@pytest.mark.asyncio
async def test_delete_reminder_command():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_update.message.reply_text = AsyncMock()
    mock_context = MagicMock()
    mock_context.args = ["1"]

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.memory.delete_reminder", return_value=True):
        await delete_reminder_command(mock_update, mock_context)

    assert "Đã hủy nhắc việc `#1` thành công" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_read_url_command():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_status_msg = AsyncMock()
    mock_update.message.reply_text = AsyncMock(return_value=mock_status_msg)
    mock_context = MagicMock()
    mock_context.args = ["https://example.com/article"]

    fake_page = WebPageContent(
        url="https://example.com/article",
        title="AI News",
        text="Sample content",
        length=14,
        error=None
    )

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.fetch_web_page", return_value=fake_page), \
         patch("bot.gemini.generate_multimodal", return_value="Summary of AI News"):
        await read_url_command(mock_update, mock_context)

    mock_status_msg.edit_text.assert_called()
    assert "Summary of AI News" in mock_status_msg.edit_text.call_args[0][0]

@pytest.mark.asyncio
async def test_speedtest_command():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_status_msg = AsyncMock()
    mock_update.message.reply_text = AsyncMock(return_value=mock_status_msg)

    fake_speed = SpeedtestResult(
        download_mbps=100.0,
        upload_mbps=50.0,
        ping_ms=10.0,
        server_name="HCM",
        server_country="VN",
        sponsor="FPT",
        text="Speedtest 100 Mbps",
        error=None
    )

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.run_vps_speedtest", return_value=fake_speed):
        await speedtest_command(mock_update, MagicMock())

    mock_status_msg.edit_text.assert_called_once_with("Speedtest 100 Mbps", parse_mode="Markdown")
