import pytest
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes

import bot
from bot import (
    note_command,
    list_notes_command,
    delete_note_command,
    find_note_command,
)


@pytest.fixture
def mock_telegram_objects(monkeypatch):
    monkeypatch.setattr(bot, "is_authorized", lambda uid: True)

    user = MagicMock(spec=User)
    user.id = 12345
    user.username = "testuser"

    chat = MagicMock(spec=Chat)
    chat.id = 12345
    chat.type = "private"

    message = MagicMock(spec=Message)
    message.from_user = user
    message.chat = chat
    message.reply_to_message = None
    message.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.effective_user = user
    update.effective_chat = chat
    update.message = message

    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.args = []
    context.bot = MagicMock()

    return update, context, message


@pytest.mark.asyncio
async def test_note_command_standalone(mock_telegram_objects, monkeypatch, tmp_path):
    update, context, message = mock_telegram_objects
    context.args = ["#devops", "Oracle", "swap", "4GB"]

    # Mock get_embedding
    monkeypatch.setattr(bot.gemini, "get_embedding", AsyncMock(return_value=[0.1, 0.2]))

    await note_command(update, context)
    message.reply_text.assert_called_once()
    reply_args = message.reply_text.call_args[0][0]
    assert "Đã lưu ghi chú" in reply_args
    assert "[#devops]" in reply_args


@pytest.mark.asyncio
async def test_note_command_reply(mock_telegram_objects, monkeypatch):
    update, context, message = mock_telegram_objects
    context.args = ["#quote"]

    replied_msg = MagicMock(spec=Message)
    replied_msg.text = "This is a great thought to save."
    replied_msg.caption = None
    message.reply_to_message = replied_msg

    monkeypatch.setattr(bot.gemini, "get_embedding", AsyncMock(return_value=[0.3, 0.4]))

    await note_command(update, context)
    message.reply_text.assert_called_once()
    reply_args = message.reply_text.call_args[0][0]
    assert "Đã lưu ghi chú" in reply_args


@pytest.mark.asyncio
async def test_list_notes_command(mock_telegram_objects, monkeypatch):
    update, context, message = mock_telegram_objects
    context.args = []

    await list_notes_command(update, context)
    assert message.reply_text.called or message.chat.send_message.called



@pytest.mark.asyncio
async def test_find_note_command(mock_telegram_objects, monkeypatch):
    update, context, message = mock_telegram_objects
    context.args = ["swap"]

    monkeypatch.setattr(bot.gemini, "get_embedding", AsyncMock(return_value=[0.1, 0.2]))

    await find_note_command(update, context)
    assert message.reply_text.called or update.effective_chat.send_message.called


@pytest.mark.asyncio
async def test_delete_note_command(mock_telegram_objects, monkeypatch):
    update, context, message = mock_telegram_objects
    # Invalid arg
    context.args = ["abc"]
    await delete_note_command(update, context)
    message.reply_text.assert_called_with("Cách dùng: `/del_note <id>` (Ví dụ: `/del_note 1`)", parse_mode="Markdown")

    # Valid arg
    context.args = ["1"]
    await delete_note_command(update, context)
    assert message.reply_text.call_count == 2
