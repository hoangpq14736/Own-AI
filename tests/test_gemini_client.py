import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from core.gemini_client import GeminiClient
from core.models import ChatMessage

@pytest.fixture
def mock_genai_client():
    client = MagicMock()
    # Mock models.generate_content
    mock_response = MagicMock()
    mock_response.text = "Hello from mock Gemini!"
    client.models.generate_content.return_value = mock_response

    # Mock files.upload and files.delete
    mock_uploaded = MagicMock()
    mock_uploaded.name = "files/test_file_id"
    client.files.upload.return_value = mock_uploaded
    client.files.delete.return_value = None

    return client

def test_gemini_client_init_and_empty_key():
    client = GeminiClient(api_key="", model_name="gemini-3.8-flash")
    assert client.api_key == ""
    assert client.model_name == "gemini-3.8-flash"
    assert "gemini-3.5-flash-lite" in client.fallback_models

@pytest.mark.anyio
async def test_generate_multimodal_missing_key():
    client = GeminiClient(api_key="")
    res = await client.generate_multimodal(prompt="Hello")
    assert "GEMINI_API_KEY is not configured" in res

@pytest.mark.anyio
async def test_generate_multimodal_text_success(mock_genai_client):
    client = GeminiClient(api_key="test-api-key", client=mock_genai_client)

    chat_history = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello! How are you?"),
        {"role": "user", "content": "Tell me a joke."}
    ]

    result = await client.generate_multimodal(
        prompt="Why did the chicken cross the road?",
        chat_history=chat_history
    )

    assert result == "Hello from mock Gemini!"
    assert mock_genai_client.models.generate_content.called

@pytest.mark.anyio
async def test_generate_multimodal_with_inline_file(mock_genai_client, tmp_path):
    client = GeminiClient(api_key="test-api-key", client=mock_genai_client)

    # Create small test file (<15MB)
    sample_file = tmp_path / "test_photo.jpg"
    sample_file.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 1024)

    result = await client.generate_multimodal(
        prompt="Analyze this photo",
        media_path=sample_file
    )

    assert result == "Hello from mock Gemini!"
    assert mock_genai_client.models.generate_content.called

@pytest.mark.anyio
async def test_generate_multimodal_with_large_file(mock_genai_client, tmp_path):
    client = GeminiClient(api_key="test-api-key", client=mock_genai_client)

    # Mock large file > 15MB
    sample_file = tmp_path / "large_video.mp4"
    sample_file.write_bytes(b"\x00" * 100)

    # Mock stat size to be 20MB
    with patch.object(Path, "stat") as mock_stat:
        mock_stat_res = MagicMock()
        mock_stat_res.st_size = 20 * 1024 * 1024
        mock_stat.return_value = mock_stat_res

        result = await client.generate_multimodal(
            prompt="Analyze this long video",
            media_path=sample_file
        )

        assert result == "Hello from mock Gemini!"
        mock_genai_client.files.upload.assert_called_once()
        mock_genai_client.files.delete.assert_called_once_with(name="files/test_file_id")

@pytest.mark.anyio
async def test_generate_multimodal_503_fallback(mock_genai_client):
    client = GeminiClient(api_key="test-api-key", client=mock_genai_client)

    # First call fails with 503 high demand, second succeeds
    fail_res = Exception("503 Model is temporarily overloaded. Please try again later.")
    success_res = MagicMock()
    success_res.text = "Fallback model response succeeded!"

    mock_genai_client.models.generate_content.side_effect = [fail_res, success_res]

    result = await client.generate_multimodal(prompt="Hello during surge")
    assert result == "Fallback model response succeeded!"
    assert mock_genai_client.models.generate_content.call_count == 2

@pytest.mark.anyio
async def test_generate_multimodal_unrecoverable_error(mock_genai_client):
    client = GeminiClient(api_key="test-api-key", client=mock_genai_client)
    mock_genai_client.models.generate_content.side_effect = Exception("403 Forbidden: Invalid API credentials")

    result = await client.generate_multimodal(prompt="Hello")
    assert "Error communicating with Gemini API" in result
    assert "403 Forbidden" in result
