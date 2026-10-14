"""
Unit Tests for core/mltb_brain.py (MLTB Natural Language Synthesizer).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.mltb_brain import translate_natural_language_to_mltb


@pytest.mark.anyio
async def test_translate_empty_prompt():
    mock_gemini = MagicMock()
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, "")
    assert cmd is None
    assert exp is None


@pytest.mark.anyio
async def test_translate_reply_media_fast_path():
    mock_gemini = MagicMock()

    # Normal leech
    cmd, exp = await translate_natural_language_to_mltb(
        mock_gemini,
        "tải file này về telegram",
        has_reply_media=True
    )
    assert cmd == "/leech"
    assert "Telegram" in exp

    # Leech with unzip
    cmd, exp = await translate_natural_language_to_mltb(
        mock_gemini,
        "giải nén rồi leech hộ",
        has_reply_media=True
    )
    assert cmd == "/leech -uz"

    # Leech with zip
    cmd, exp = await translate_natural_language_to_mltb(
        mock_gemini,
        "nén zip rồi tải",
        has_reply_media=True
    )
    assert cmd == "/leech -z"


@pytest.mark.anyio
async def test_translate_gemini_mltb_intent_success():
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "is_mltb_intent": True,
        "command": "/ytdl",
        "fullCommand": "/ytdl https://youtu.be/xyz -s",
        "explanation": "Tải video YouTube lên đám mây"
    })

    mock_gemini = MagicMock()
    mock_gemini.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    cmd, exp = await translate_natural_language_to_mltb(
        mock_gemini,
        "tải video youtube https://youtu.be/xyz lên drive",
        system_rules="Default cloud: gdrive"
    )

    assert cmd == "/ytdl https://youtu.be/xyz -s"
    assert exp == "Tải video YouTube lên đám mây"


@pytest.mark.anyio
async def test_translate_gemini_not_mltb_intent():
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "is_mltb_intent": False,
        "command": None,
        "fullCommand": None,
        "explanation": "General conversation"
    })

    mock_gemini = MagicMock()
    mock_gemini.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    cmd, exp = await translate_natural_language_to_mltb(
        mock_gemini,
        "Xin chào bạn khỏe không?"
    )

    assert cmd is None
    assert exp is None


@pytest.mark.anyio
async def test_translate_gemini_exception_handled():
    mock_gemini = MagicMock()
    mock_gemini.client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API error"))

    cmd, exp = await translate_natural_language_to_mltb(
        mock_gemini,
        "download torrent magnet:?xt=urn:btih:..."
    )

    assert cmd is None
    assert exp is None
