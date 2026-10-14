import pytest
import base64
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from core.media_generator import (
    validate_prompt,
    generate_google_image,
    generate_google_video,
    text_to_speech,
    get_google_client,
)
from core.exceptions import ValidationError

def test_get_google_client():
    assert get_google_client("") is None
    client = get_google_client("test_api_key")
    assert client is not None

def test_validate_prompt():
    assert validate_prompt("A glowing phoenix") == "A glowing phoenix"
    assert validate_prompt("  padded prompt  ") == "padded prompt"

    with pytest.raises(ValidationError):
        validate_prompt("")

    with pytest.raises(ValidationError):
        validate_prompt("   ")

    with pytest.raises(ValidationError):
        validate_prompt("a" * 1001, max_len=1000)

@pytest.mark.anyio
async def test_generate_google_image_no_client():
    with patch("core.media_generator.get_google_client", return_value=None):
        result = await generate_google_image("a dog", client=None)
        assert result is None

@pytest.mark.anyio
async def test_generate_google_image_success():
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data.data = b"\xFF\xD8\xFF\xE0fakejpegdata"
    mock_res.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    mock_client.models.generate_content.return_value = mock_res

    image_bytes = await generate_google_image("a cybernetic tiger", client=mock_client)
    assert image_bytes == b"\xFF\xD8\xFF\xE0fakejpegdata"
    assert mock_client.models.generate_content.called

@pytest.mark.anyio
async def test_generate_google_image_fallback():
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data.data = b"fallback_jpeg"
    mock_res.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

    # Model 1 fails, Model 2 succeeds
    mock_client.models.generate_content.side_effect = [
        Exception("Model 1 unavailable"),
        mock_res
    ]

    image_bytes = await generate_google_image("a golden flower", client=mock_client)
    assert image_bytes == b"fallback_jpeg"
    assert mock_client.models.generate_content.call_count == 2

@pytest.mark.anyio
async def test_generate_google_image_empty_prompt():
    result = await generate_google_image("   ")
    assert result is None

@pytest.mark.anyio
async def test_generate_google_video_no_client():
    with patch("core.media_generator.get_google_client", return_value=None):
        result = await generate_google_video("a video", client=None)
        assert result is None

@pytest.mark.anyio
async def test_generate_google_video_with_bytes(tmp_path):
    mock_client = MagicMock()
    mock_op = MagicMock()
    mock_op.name = "operations/veo_op_123"
    mock_client.models.generate_videos.return_value = mock_op

    # Mock operation polling
    done_op = MagicMock()
    done_op.done = True
    done_video = MagicMock()
    done_video.video.video_bytes = b"\x00\x00\x00\x18ftypmp42"
    done_video.video.uri = None
    done_op.response.generated_videos = [done_video]
    done_op.error = None
    mock_client.operations.get.return_value = done_op

    video_path = await generate_google_video(
        prompt="A dolphin jumping at sunset",
        client=mock_client,
        output_dir=tmp_path,
        max_polls=2
    )

    assert video_path is not None
    assert video_path.exists()
    assert video_path.read_bytes() == b"\x00\x00\x00\x18ftypmp42"

@pytest.mark.anyio
async def test_generate_google_video_with_uri(tmp_path):
    mock_client = MagicMock()
    mock_op = MagicMock()
    mock_op.name = "operations/veo_op_456"
    mock_client.models.generate_videos.return_value = mock_op

    # Mock operation polling with file URI
    done_op = MagicMock()
    done_op.done = True
    done_video = MagicMock()
    done_video.video.video_bytes = None
    done_video.video.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123xyz"
    done_op.response.generated_videos = [done_video]
    done_op.error = None
    mock_client.operations.get.return_value = done_op
    mock_client.files.download.return_value = b"downloaded_video_bytes"

    video_path = await generate_google_video(
        prompt="A mountain landscape timelapse",
        client=mock_client,
        output_dir=tmp_path,
        max_polls=2
    )

    assert video_path is not None
    assert video_path.exists()
    assert video_path.read_bytes() == b"downloaded_video_bytes"

@pytest.mark.anyio
async def test_generate_google_video_failure():
    mock_client = MagicMock()
    mock_client.models.generate_videos.side_effect = Exception("Veo API error")

    result = await generate_google_video("invalid prompt", client=mock_client)
    assert result is None

@pytest.mark.anyio
async def test_text_to_speech_vietnamese_and_english(tmp_path):
    created_voices = []

    def fake_communicator_factory(text, voice):
        created_voices.append(voice)
        mock_comm = AsyncMock()

        async def fake_save(path_str):
            p = Path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"ID3fakeaudio")

        mock_comm.save = fake_save
        return mock_comm

    # Vietnamese text
    vi_output = tmp_path / "vi.mp3"
    vi_res = await text_to_speech(
        "Xin chào, tôi là trợ lý ảo cá nhân của bạn",
        output_path=vi_output,
        communicator_factory=fake_communicator_factory
    )
    assert vi_res is not None
    assert vi_res.exists()
    assert "vi-VN-HoaiMyNeural" in created_voices

    # English text
    en_output = tmp_path / "en.mp3"
    en_res = await text_to_speech(
        "Hello, this is your AI assistant",
        output_path=en_output,
        communicator_factory=fake_communicator_factory
    )
    assert en_res is not None
    assert en_res.exists()
    assert "en-US-AriaNeural" in created_voices

    # Explicit preferred voice
    pref_output = tmp_path / "pref.mp3"
    pref_res = await text_to_speech(
        "Preferred voice test",
        output_path=pref_output,
        preferred_voice="en-GB-SoniaNeural",
        communicator_factory=fake_communicator_factory
    )
    assert pref_res is not None
    assert "en-GB-SoniaNeural" in created_voices

@pytest.mark.anyio
async def test_text_to_speech_fallback_on_error(tmp_path):
    call_count = 0

    def failing_first_communicator(text, voice):
        nonlocal call_count
        call_count += 1
        mock_comm = AsyncMock()
        if call_count == 1:
            mock_comm.save.side_effect = Exception("Voice 1 network error")
        else:
            async def fake_save(path_str):
                p = Path(path_str)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fallback_audio")
            mock_comm.save = fake_save
        return mock_comm

    out_file = tmp_path / "fallback.mp3"
    res = await text_to_speech(
        "Testing voice fallback mechanism",
        output_path=out_file,
        communicator_factory=failing_first_communicator
    )
    assert res is not None
    assert res.read_bytes() == b"fallback_audio"

@pytest.mark.anyio
async def test_text_to_speech_empty():
    res = await text_to_speech("")
    assert res is None
    res_space = await text_to_speech("   ***   ")
    assert res_space is None

@pytest.mark.anyio
async def test_text_to_speech_gemini_tts(tmp_path):
    mock_client = MagicMock()
    mock_interaction = MagicMock()
    mock_interaction.output_audio.data = base64.b64encode(b"RIFFfakeaudio").decode("utf-8")
    mock_client.interactions.create.return_value = mock_interaction

    out_file = tmp_path / "gemini_voice.wav"
    res = await text_to_speech(
        "Xin chào bạn nhé",
        output_path=out_file,
        preferred_voice="Kore",
        client=mock_client
    )

    assert res is not None
    assert res.exists()
    assert res.read_bytes() == b"RIFFfakeaudio"
    assert mock_client.interactions.create.called
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash-tts"
    assert call_kwargs["generation_config"]["speech_config"] == [{"voice": "Kore"}]

