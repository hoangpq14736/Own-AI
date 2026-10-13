"""
Unit Tests for UserbotClient (Telethon Integration).
Tests configuration validation, connection lifecycles, and command dispatching.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.userbot import UserbotClient


@pytest.mark.anyio
async def test_userbot_unconfigured():
    client = UserbotClient(api_id=None, api_hash="", session_string="")
    assert client.is_configured is False
    assert client.is_connected is False
    assert client.username is None
    assert client.user_id is None

    # Starting unconfigured client returns False
    res = await client.start()
    assert res is False
    assert client.is_connected is False

    # Sending command without connection returns False
    dispatched = await client.send_command(chat_id=12345, command="/leech http://example.com")
    assert dispatched is False


@pytest.mark.anyio
async def test_userbot_custom_injected_client():
    mock_me = MagicMock()
    mock_me.id = 987654321
    mock_me.username = "test_user"

    mock_raw_client = MagicMock()
    mock_raw_client.get_me = AsyncMock(return_value=mock_me)
    mock_raw_client.send_message = AsyncMock(return_value=MagicMock())
    mock_raw_client.disconnect = AsyncMock()

    client = UserbotClient(
        api_id=12345,
        api_hash="abcdef",
        session_string="test_session",
        client=mock_raw_client,
    )

    assert client.is_configured is True
    started = await client.start()
    assert started is True
    assert client.is_connected is True
    assert client.username == "test_user"
    assert client.user_id == 987654321

    # Dispatch command with reply
    sent = await client.send_command(
        chat_id=-100123456789,
        command="/leech https://example.com/file.zip",
        reply_to_msg_id=555,
    )
    assert sent is True
    mock_raw_client.send_message.assert_awaited_once_with(
        entity=-100123456789,
        message="/leech https://example.com/file.zip",
        reply_to=555,
    )

    # Disconnect
    await client.stop()
    assert client.is_connected is False
    mock_raw_client.disconnect.assert_awaited_once()


@pytest.mark.anyio
async def test_userbot_send_command_error_handling():
    mock_me = MagicMock()
    mock_me.id = 111
    mock_me.username = "error_user"

    mock_raw_client = MagicMock()
    mock_raw_client.get_me = AsyncMock(return_value=mock_me)
    mock_raw_client.send_message = AsyncMock(side_effect=RuntimeError("FloodWait error"))
    mock_raw_client.disconnect = AsyncMock()

    client = UserbotClient(
        api_id=12345,
        api_hash="abcdef",
        session_string="test_session",
        client=mock_raw_client,
    )

    await client.start()
    dispatched = await client.send_command(chat_id=123, command="/mirror magnet:?xt=urn")
    assert dispatched is False


@pytest.mark.anyio
async def test_userbot_telethon_start_failure():
    with patch("core.userbot.StringSession") as mock_ss, patch("core.userbot.TelegramClient") as mock_tg:
        mock_instance = MagicMock()
        mock_instance.start = AsyncMock(side_effect=ValueError("Invalid session string"))
        mock_tg.return_value = mock_instance

        client = UserbotClient(
            api_id=12345,
            api_hash="abcdef",
            session_string="bad_session_string",
        )
        assert client.is_configured is True
        started = await client.start()
        assert started is False
        assert client.is_connected is False


@pytest.mark.anyio
async def test_userbot_telethon_start_success():
    with patch("core.userbot.StringSession") as mock_ss, patch("core.userbot.TelegramClient") as mock_tg:
        mock_me = MagicMock()
        mock_me.id = 444
        mock_me.username = "real_telethon_user"

        mock_instance = MagicMock()
        mock_instance.start = AsyncMock()
        mock_instance.get_me = AsyncMock(return_value=mock_me)
        mock_instance.disconnect = AsyncMock()
        mock_tg.return_value = mock_instance

        client = UserbotClient(
            api_id=12345,
            api_hash="abcdef",
            session_string="good_session_string",
        )
        started = await client.start()
        assert started is True
        assert client.is_connected is True
        assert client.username == "real_telethon_user"
        assert client.user_id == 444

        await client.stop()
        assert client.is_connected is False
