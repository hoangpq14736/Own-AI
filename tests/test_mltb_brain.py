import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from core.mltb_brain import translate_natural_language_to_mltb

@pytest.fixture
def mock_gemini():
    client_mock = MagicMock()
    aio_mock = MagicMock()
    models_mock = MagicMock()
    
    client_mock.aio = aio_mock
    aio_mock.models = models_mock
    
    gemini_obj = MagicMock()
    gemini_obj.client = client_mock
    return gemini_obj

@pytest.mark.asyncio
async def test_vietnamese_torrent_unzip(mock_gemini):
    # Mock Gemini response
    response_mock = MagicMock()
    response_mock.text = json.dumps({
        "is_mltb_intent": True,
        "command": "/unzipleech",
        "fullCommand": "/unzipleech magnet:?xt=urn:btih:3b245504cf5f",
        "explanation": "Leech torrent and extract archives to Telegram"
    })
    mock_gemini.client.aio.models.generate_content = AsyncMock(return_value=response_mock)

    prompt = "Kéo torrent này về telegram giải nén hết ra cho tao: magnet:?xt=urn:btih:3b245504cf5f"
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, prompt)

    assert cmd is not None
    assert "/unzipleech" in cmd
    assert "magnet:?xt=urn:btih:3b245504cf5f" in cmd

@pytest.mark.asyncio
async def test_rename_and_custom_rclone_remote(mock_gemini):
    response_mock = MagicMock()
    response_mock.text = json.dumps({
        "is_mltb_intent": True,
        "command": "/mirror",
        "fullCommand": "/mirror https://example.com/test.iso -n ubuntu.iso -up backup:ISOs",
        "explanation": "Mirror link with rename and custom remote"
    })
    mock_gemini.client.aio.models.generate_content = AsyncMock(return_value=response_mock)

    prompt = "Tải file này lên remote backup:ISOs đổi tên thành ubuntu.iso: https://example.com/test.iso"
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, prompt, system_rules="Default remote: backup")

    assert "-n ubuntu.iso" in cmd
    assert "-up backup:ISOs" in cmd

@pytest.mark.asyncio
async def test_ytdl_best_audio_format(mock_gemini):
    response_mock = MagicMock()
    response_mock.text = json.dumps({
        "is_mltb_intent": True,
        "command": "/ytdlleech",
        "fullCommand": "/ytdlleech https://youtu.be/music opt: format=bestaudio/best",
        "explanation": "Leech audio with best sound format"
    })
    mock_gemini.client.aio.models.generate_content = AsyncMock(return_value=response_mock)

    prompt = "tải nhạc này về với định dạng âm thanh tốt nhất: https://youtu.be/music"
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, prompt)

    assert "/ytdlleech" in cmd
    assert "opt: format=bestaudio/best" in cmd

@pytest.mark.asyncio
async def test_ytdl_best_video_and_2k_formats(mock_gemini):
    # Test 2K
    response_mock = MagicMock()
    response_mock.text = json.dumps({
        "is_mltb_intent": True,
        "command": "/ytdlleech",
        "fullCommand": "/ytdlleech https://youtu.be/video opt: format=bv*[height<=1440]+ba/b",
        "explanation": "Leech 2K video"
    })
    mock_gemini.client.aio.models.generate_content = AsyncMock(return_value=response_mock)

    prompt = "tải video này với định dạng 2k: https://youtu.be/video"
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, prompt)

    assert "height<=1440" in cmd

    # Test Best Video
    response_mock.text = json.dumps({
        "is_mltb_intent": True,
        "command": "/ytdlleech",
        "fullCommand": "/ytdlleech https://youtu.be/video opt: format=bv*+ba/b",
        "explanation": "Leech best video"
    })
    prompt = "tải video này với định dạng video tốt nhất: https://youtu.be/video"
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, prompt)

    assert "format=bv*+ba/b" in cmd

@pytest.mark.asyncio
async def test_general_chat_not_triggered(mock_gemini):
    response_mock = MagicMock()
    response_mock.text = json.dumps({
        "is_mltb_intent": False,
        "command": "",
        "fullCommand": "",
        "explanation": "General conversational query"
    })
    mock_gemini.client.aio.models.generate_content = AsyncMock(return_value=response_mock)

    prompt = "Hôm nay thời tiết Hà Nội thế nào?"
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, prompt)

    assert cmd is None

@pytest.mark.asyncio
async def test_empty_prompt_and_reply_media_fast_path(mock_gemini):
    # Empty prompt
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, "")
    assert cmd is None

    # Reply media fast-path
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, "giải nén rồi leech", has_reply_media=True)
    assert cmd == "/leech -uz"

    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, "nén zip tải về", has_reply_media=True)
    assert cmd == "/leech -z"

@pytest.mark.asyncio
async def test_gemini_exception_handled(mock_gemini):
    mock_gemini.client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API Network Error"))
    cmd, exp = await translate_natural_language_to_mltb(mock_gemini, "tải video https://youtu.be/test")
    assert cmd is None
