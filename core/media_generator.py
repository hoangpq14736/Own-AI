"""
Multimodal Media Generation Module (Image, Video, Neural TTS).
Implements prompt boundary validation, dependency injection, and clean resource management per ECC.
"""

from __future__ import annotations
import asyncio
import base64
import re
import uuid
import time
from pathlib import Path
from typing import Optional, Callable, Any
import logging

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

import edge_tts
import config
from core.exceptions import ValidationError, MediaGenerationError

logger = logging.getLogger(__name__)

# Vietnamese character detection for TTS
VIETNAMESE_REGEX = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE
)

# Allowed maximum prompt lengths
MAX_IMAGE_PROMPT_LEN = 1000
MAX_VIDEO_PROMPT_LEN = 1000
MAX_TTS_TEXT_LEN = 1000


def get_google_client(api_key: Optional[str] = None):
    """Lazy initializer for Google GenAI client to support testing and clean configuration."""
    key = config.GEMINI_API_KEY if api_key is None else api_key
    if not key or not key.strip():
        return None
    return genai.Client(api_key=key.strip())


def validate_prompt(prompt: str, max_len: int = 1000) -> str:
    """Validates and sanitizes prompt strings at system boundaries."""
    if not prompt or not prompt.strip():
        raise ValidationError("Prompt cannot be empty.")
    cleaned = prompt.strip()
    if len(cleaned) > max_len:
        raise ValidationError(f"Prompt exceeds maximum allowed length of {max_len} characters.")
    return cleaned


# ==============================================================================
# 1. Google Native Image Generation
# ==============================================================================

async def generate_google_image(
    prompt: str,
    client=None
) -> Optional[bytes]:
    """
    Generates a high-resolution image natively using Google Gemini Image Models.
    Allows injecting a client for unit testing.
    """
    try:
        clean_prompt = validate_prompt(prompt, MAX_IMAGE_PROMPT_LEN)
    except ValidationError as e:
        logger.warning(f"Image generation rejected: {e}")
        return None

    g_client = client or get_google_client()
    if not g_client:
        logger.error("Google GenAI client unavailable (missing API key).")
        return None

    image_models = [
        "gemini-3.1-flash-image",
        "gemini-2.5-flash-image",
        "gemini-3-pro-image",
    ]

    for model_name in image_models:
        try:
            logger.info(f"Generating image with Google model '{model_name}' for: {clean_prompt}")

            def _call_model():
                return g_client.models.generate_content(
                    model=model_name,
                    contents=f"Generate an image: {clean_prompt}"
                )

            res = await asyncio.to_thread(_call_model)

            if res and getattr(res, "candidates", None):
                for part in res.candidates[0].content.parts:
                    if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        logger.info(f"Successfully generated Google image ({len(part.inline_data.data)} bytes)")
                        return part.inline_data.data

        except Exception as e:
            logger.warning(f"Google image model '{model_name}' failed: {e}. Trying fallback...")
            continue

    logger.error("All Google image models failed.")
    return None


# ==============================================================================
# 2. Google Veo Video Generation
# ==============================================================================

async def generate_google_video(
    prompt: str,
    client=None,
    output_dir: Optional[Path] = None,
    max_polls: int = 18
) -> Optional[Path]:
    """
    Generates a cinematic video natively using Google Veo.
    Polls operation until completion and downloads the MP4 video.
    Allows injecting client and output_dir for testability.
    """
    try:
        clean_prompt = validate_prompt(prompt, MAX_VIDEO_PROMPT_LEN)
    except ValidationError as e:
        logger.warning(f"Video generation rejected: {e}")
        return None

    g_client = client or get_google_client()
    if not g_client:
        logger.error("Google GenAI client unavailable (missing API key).")
        return None

    target_dir = output_dir or config.MEDIA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    video_models = [
        "veo-3.1-fast-generate-preview",
        "veo-3.1-lite-generate-preview",
    ]

    for model_name in video_models:
        try:
            logger.info(f"Initiating Google Veo video generation with '{model_name}' for: {clean_prompt}")

            def _start_video():
                return g_client.models.generate_videos(
                    model=model_name,
                    prompt=clean_prompt
                )

            operation = await asyncio.to_thread(_start_video)
            op_name = getattr(operation, "name", None)
            logger.info(f"Veo operation launched: {op_name}")

            # Poll operation until done
            for poll_idx in range(max_polls):
                await asyncio.sleep(1 if client else 4)

                def _check_op():
                    return g_client.operations.get(operation)

                current_op = await asyncio.to_thread(_check_op)
                logger.info(f"Veo polling [{poll_idx+1}/{max_polls}]: done={getattr(current_op, 'done', False)}")

                if getattr(current_op, "done", False):
                    if getattr(current_op, "response", None) and hasattr(current_op.response, "generated_videos"):
                        gen_video = current_op.response.generated_videos[0].video
                        output_path = target_dir / f"video_{uuid.uuid4().hex[:8]}.mp4"

                        if getattr(gen_video, "video_bytes", None):
                            with open(output_path, "wb") as f:
                                f.write(gen_video.video_bytes)
                            return output_path

                        elif getattr(gen_video, "uri", None):
                            uri = gen_video.uri
                            match = re.search(r"files/([^:]+)", uri)
                            if match:
                                file_id = f"files/{match.group(1)}"
                                file_bytes = await asyncio.to_thread(g_client.files.download, file=file_id)
                            else:
                                import urllib.request
                                req = urllib.request.Request(uri, headers={"x-goog-api-key": config.GEMINI_API_KEY})
                                with urllib.request.urlopen(req) as resp:
                                    file_bytes = resp.read()

                            with open(output_path, "wb") as f:
                                f.write(file_bytes)

                            logger.info(f"Google Veo video saved: {output_path} ({output_path.stat().st_size} bytes)")
                            return output_path

                    elif getattr(current_op, "error", None):
                        logger.error(f"Veo operation returned error: {current_op.error}")
                        break

        except Exception as e:
            logger.warning(f"Google Veo model '{model_name}' error: {e}. Trying next...")
            continue

    logger.error("All Google Veo video models failed.")
    return None


# ==============================================================================
# 3. Neural Voice Generation (TTS)
# ==============================================================================

async def text_to_speech(
    text: str,
    output_path: Optional[Path] = None,
    preferred_voice: Optional[str] = None,
    client: Optional[Any] = None,
    communicator_factory=None
) -> Optional[Path]:
    """
    Converts text to realistic human speech using Google Gemini TTS (gemini-3.8-flash-tts).
    Supports 130+ languages including Vietnamese, with fallback to edge-tts if needed.
    """
    if not text or not text.strip():
        return None

    clean_text = re.sub(r"[\*\_\[\]\(\)\#\-\`\>\~]", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    if not clean_text:
        return None

    if len(clean_text) > MAX_TTS_TEXT_LEN:
        clean_text = clean_text[:MAX_TTS_TEXT_LEN] + "..."

    voice_name = preferred_voice or "Kore"

    if not output_path:
        output_path = config.MEDIA_DIR / f"voice_{uuid.uuid4().hex[:8]}.wav"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Primary Engine: Google Gemini TTS
    ai_client = client if client is not None else get_google_client()
    if ai_client and hasattr(ai_client, "interactions") and not communicator_factory:
        for model in ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]:
            try:
                interaction = await asyncio.to_thread(
                    ai_client.interactions.create,
                    model=model,
                    input=[{
                        "type": "user_input",
                        "content": [{
                            "type": "text",
                            "text": clean_text,
                        }],
                    }],
                    response_format={"type": "audio"},
                    generation_config={
                        "speech_config": [{"voice": voice_name}]
                    }
                )
                if interaction.output_audio and hasattr(interaction.output_audio, "data"):
                    audio_bytes = base64.b64decode(interaction.output_audio.data)
                    output_path.write_bytes(audio_bytes)
                    if output_path.exists() and output_path.stat().st_size > 0:
                        logger.info(f"Generated speech with Google Gemini TTS ({model}, {voice_name})")
                        return output_path
            except Exception as e:
                logger.warning(f"Google Gemini TTS with {model} failed: {e}. Trying fallback...")

    # 2. Secondary Fallback: Edge-TTS (or mock communicator_factory in unit tests)
    if VIETNAMESE_REGEX.search(clean_text):
        edge_primary = "vi-VN-HoaiMyNeural"
        edge_fallback = "vi-VN-NamMinhNeural"
    else:
        edge_primary = "en-US-AriaNeural"
        edge_fallback = "en-US-GuyNeural"

    if preferred_voice:
        edge_candidates = [preferred_voice, edge_primary, edge_fallback, "en-US-AriaNeural"]
    else:
        edge_candidates = [edge_primary, edge_fallback, "en-US-AriaNeural"]

    for v in dict.fromkeys(edge_candidates):
        try:
            if communicator_factory:
                communicate = communicator_factory(clean_text, v)
            else:
                communicate = edge_tts.Communicate(clean_text, v)

            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Generated speech with Edge-TTS fallback ({v})")
                return output_path
        except Exception as e:
            logger.warning(f"TTS voice {v} failed: {e}. Trying fallback...")

    logger.error("All TTS voice options failed.")
    return None

