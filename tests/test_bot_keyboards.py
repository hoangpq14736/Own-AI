import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bot import (
    get_vps_keyboard,
    get_docker_keyboard,
    vps_docker_callback_handler,
    exec_command,
)

def test_get_vps_keyboard():
    markup = get_vps_keyboard()
    assert len(markup.inline_keyboard) == 1
    row = markup.inline_keyboard[0]
    assert len(row) == 2
    assert row[0].callback_data == "vps:ref"
    assert row[1].callback_data == "vps:clean"

def test_get_docker_keyboard_empty():
    markup = get_docker_keyboard([])
    assert len(markup.inline_keyboard) == 1
    assert markup.inline_keyboard[0][0].callback_data == "dk:ref"

def test_get_docker_keyboard_with_containers():
    containers = [
        {"name": "mirrorbot-app-1", "status": "running", "image": "mirror:latest"},
        {"name": "portainer", "status": "exited", "image": "portainer:latest"},
    ]
    markup = get_docker_keyboard(containers)
    assert len(markup.inline_keyboard) == 3  # 2 container rows + 1 refresh row
    # Row 1: mirrorbot-app-1
    assert markup.inline_keyboard[0][0].callback_data == "dk:r:mirrorbot-app-1"
    assert markup.inline_keyboard[0][1].callback_data == "dk:l:mirrorbot-app-1"
    # Row 2: portainer
    assert markup.inline_keyboard[1][0].callback_data == "dk:r:portainer"
    assert markup.inline_keyboard[1][1].callback_data == "dk:l:portainer"
    # Row 3: refresh
    assert markup.inline_keyboard[2][0].callback_data == "dk:ref"

@pytest.mark.asyncio
async def test_callback_unauthorized():
    mock_query = AsyncMock()
    mock_query.from_user.id = 999999
    mock_query.data = "vps:ref"

    mock_update = MagicMock()
    mock_update.callback_query = mock_query

    with patch("bot.is_authorized", return_value=False):
        await vps_docker_callback_handler(mock_update, MagicMock())

    mock_query.answer.assert_called_once_with("⛔ Bạn không có quyền thao tác.", show_alert=True)
    mock_query.edit_message_text.assert_not_called()

@pytest.mark.asyncio
async def test_callback_vps_refresh():
    mock_query = AsyncMock()
    mock_query.from_user.id = 12345
    mock_query.data = "vps:ref"

    mock_update = MagicMock()
    mock_update.callback_query = mock_query

    with patch("bot.is_authorized", return_value=True):
        await vps_docker_callback_handler(mock_update, MagicMock())

    mock_query.answer.assert_called_once_with("🔄 Đã cập nhật chỉ số VPS!")
    mock_query.edit_message_text.assert_called_once()

@pytest.mark.asyncio
async def test_callback_vps_clean():
    mock_query = AsyncMock()
    mock_query.from_user.id = 12345
    mock_query.data = "vps:clean"

    mock_update = MagicMock()
    mock_update.callback_query = mock_query

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.clean_vps_cache", return_value="Test clean ok"):
        await vps_docker_callback_handler(mock_update, MagicMock())

    mock_query.answer.assert_called_once_with("Test clean ok", show_alert=True)
    mock_query.edit_message_text.assert_called_once()

@pytest.mark.asyncio
async def test_callback_docker_restart():
    mock_query = AsyncMock()
    mock_query.from_user.id = 12345
    mock_query.data = "dk:r:test_bot"

    mock_update = MagicMock()
    mock_update.callback_query = mock_query

    with patch("bot.is_authorized", return_value=True), \
         patch("bot.restart_mltb_container", return_value="Restarted test_bot"):
        await vps_docker_callback_handler(mock_update, MagicMock())

    mock_query.edit_message_text.assert_called_once()

@pytest.mark.asyncio
async def test_exec_command_unauthorized():
    mock_update = MagicMock()
    mock_update.effective_user.id = 999999
    mock_update.effective_chat.type = "private"
    mock_update.message.reply_text = AsyncMock()

    with patch("bot.is_authorized", return_value=False):
        await exec_command(mock_update, MagicMock())

    mock_update.message.reply_text.assert_called_once_with("⛔ Unauthorized.")

@pytest.mark.asyncio
async def test_exec_command_empty():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_update.message.reply_text = AsyncMock()
    mock_context = MagicMock()
    mock_context.args = []

    with patch("bot.is_authorized", return_value=True):
        await exec_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    assert "Cách dùng:" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_exec_command_success():
    mock_update = MagicMock()
    mock_update.effective_user.id = 12345
    mock_status_msg = AsyncMock()
    mock_update.message.reply_text = AsyncMock(return_value=mock_status_msg)
    mock_context = MagicMock()
    mock_context.args = ["echo", "test_ok"]

    with patch("bot.is_authorized", return_value=True):
        await exec_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once()
    mock_status_msg.edit_text.assert_called_once()
    assert "test_ok" in mock_status_msg.edit_text.call_args[0][0]
