import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, Chat, User, Document
from telegram.ext import ContextTypes

import bot
from bot import (
    dfam_command,
    gcode_command,
    kobra_command,
    document_handler,
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
async def test_kobra_command(mock_telegram_objects):
    update, context, message = mock_telegram_objects
    await kobra_command(update, context)
    message.reply_text.assert_called_once()
    reply = message.reply_text.call_args[0][0]
    assert "Anycubic Kobra" in reply


@pytest.mark.asyncio
async def test_dfam_command_standalone(mock_telegram_objects):
    update, context, message = mock_telegram_objects
    await dfam_command(update, context)
    message.reply_text.assert_called_once()
    reply = message.reply_text.call_args[0][0]
    assert "Kiểm Tra Khả Dụng In 3D (DfAM)" in reply
    assert "Anycubic Kobra" in reply


@pytest.mark.asyncio
async def test_gcode_command_standalone(mock_telegram_objects):
    update, context, message = mock_telegram_objects
    await gcode_command(update, context)
    message.reply_text.assert_called_once()
    reply = message.reply_text.call_args[0][0]
    assert "Kiểm Tra Thông Số File G-Code" in reply


@pytest.mark.asyncio
async def test_document_handler_routing(mock_telegram_objects, monkeypatch, tmp_path):
    update, context, message = mock_telegram_objects

    # 1. Test STL routing
    doc = MagicMock(spec=Document)
    doc.file_name = "sample.stl"
    doc.file_unique_id = "test_unique_stl"
    file_mock = MagicMock()
    file_mock.download_to_drive = AsyncMock()
    doc.get_file = AsyncMock(return_value=file_mock)
    message.document = doc

    with patch("bot.analyze_stl_file") as mock_dfam:
        mock_rep = MagicMock()
        mock_rep.text = "DfAM Report for Anycubic Kobra"
        mock_dfam.return_value = mock_rep

        await document_handler(update, context)
        message.reply_text.assert_called_with("DfAM Report for Anycubic Kobra", parse_mode="Markdown")
        mock_dfam.assert_called_once()
