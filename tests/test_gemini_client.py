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

def test_safety_settings_block_none():
    from google.genai import types
    client = GeminiClient(api_key="test-key")
    settings = client._get_safety_settings()
    assert len(settings) == 5
    for s in settings:
        assert s.threshold == types.HarmBlockThreshold.BLOCK_NONE

@pytest.mark.anyio
async def test_generate_multimodal_safety_toggle(mock_genai_client):
    # With safety filters disabled (default)
    client_disabled = GeminiClient(api_key="test-key", client=mock_genai_client, disable_safety_filters=True)
    await client_disabled.generate_multimodal(prompt="Test prompt")
    call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].safety_settings is not None
    assert len(call_kwargs["config"].safety_settings) == 5

    # With safety filters enabled
    client_enabled = GeminiClient(api_key="test-key", client=mock_genai_client, disable_safety_filters=False)
    await client_enabled.generate_multimodal(prompt="Test prompt")
    call_kwargs2 = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs2["config"].safety_settings is None

@pytest.mark.anyio
async def test_generate_multimodal_search_grounding(mock_genai_client):
    # Setup mock response with grounding chunks
    chunk1 = MagicMock(web=MagicMock(title="VnExpress", uri="https://vnexpress.net/gold"))
    chunk2 = MagicMock(web=MagicMock(title="Tuoi Tre", uri="https://tuoitre.vn/gold"))
    candidate_mock = MagicMock()
    candidate_mock.grounding_metadata.grounding_chunks = [chunk1, chunk2]
    
    mock_response = MagicMock()
    mock_response.text = "Giá vàng hôm nay là 85 triệu đồng/lượng."
    mock_response.candidates = [candidate_mock]
    mock_genai_client.models.generate_content.return_value = mock_response

    client = GeminiClient(api_key="test-key", client=mock_genai_client)
    res = await client.generate_multimodal(prompt="Giá vàng hôm nay", enable_search=True)

    # Verify search tools were attached
    call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].tools is not None
    assert call_kwargs["config"].tools == [{"google_search": {}}]

    # Verify citations appended
    assert "Giá vàng hôm nay là 85 triệu đồng/lượng." in res
    assert "Nguồn tra cứu:" in res
    assert "[VnExpress](https://vnexpress.net/gold)" in res
    assert "[Tuoi Tre](https://tuoitre.vn/gold)" in res

    # Without search
    await client.generate_multimodal(prompt="Tâm sự", enable_search=False)
    call_kwargs_no_search = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs_no_search["config"].tools is None

@pytest.mark.anyio
async def test_generate_multimodal_empty_response_fallback(mock_genai_client):
    client = GeminiClient(api_key="test-key", client=mock_genai_client)

    # First call returns response with empty text, second returns valid text
    empty_res = MagicMock()
    empty_res.text = ""
    empty_res.candidates = []

    valid_res = MagicMock()
    valid_res.text = "Second model succeeded!"
    valid_res.candidates = [MagicMock()]

    mock_genai_client.models.generate_content.side_effect = [empty_res, valid_res]

    res = await client.generate_multimodal(prompt="Test empty")
    assert res == "Second model succeeded!"
    assert mock_genai_client.models.generate_content.call_count == 2

@pytest.mark.anyio
async def test_generate_multimodal_parts_text_extraction(mock_genai_client):
    client = GeminiClient(api_key="test-key", client=mock_genai_client)

    # Response where response.text is None, but parts have text
    resp = MagicMock()
    resp.text = None
    part1 = MagicMock(text="Part 1 ")
    part2 = MagicMock(text="Part 2")
    cand = MagicMock()
    cand.content.parts = [part1, part2]
    cand.finish_reason = "STOP"
    resp.candidates = [cand]

    mock_genai_client.models.generate_content.return_value = resp

    res = await client.generate_multimodal(prompt="Test parts")
    assert res == "Part 1 Part 2"

@pytest.mark.anyio
async def test_generate_multimodal_prompt_blocked_retry_without_history(mock_genai_client):
    client = GeminiClient(api_key="test-key", client=mock_genai_client)

    # First call with history gets blocked
    class RealBlockReason:
        name = "SAFETY"
    class RealPromptFeedback:
        block_reason = RealBlockReason()

    blocked_res = MagicMock()
    blocked_res.prompt_feedback = RealPromptFeedback()
    blocked_res.text = None
    blocked_res.candidates = []

    # Second call (retry with clean prompt) succeeds
    success_res = MagicMock()
    success_res.prompt_feedback = None
    success_res.text = "Clean prompt succeeded!"
    success_res.candidates = [MagicMock()]

    mock_genai_client.models.generate_content.side_effect = [blocked_res, success_res]

    res = await client.generate_multimodal(
        prompt="Tell me about dogs",
        chat_history=[{"role": "user", "content": "edgy question"}]
    )
    assert res == "Clean prompt succeeded!"
    assert mock_genai_client.models.generate_content.call_count == 2


