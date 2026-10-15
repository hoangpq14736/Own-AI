"""
Multimodal Gemini Client.
Features inline byte streaming, File API fallback, 503/429 failover, and dependency injection.
"""

import os
import mimetypes
import asyncio
from pathlib import Path
from typing import List, Optional, Any, Union
import logging

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from core.models import ChatMessage
from core.exceptions import GeminiAPIError, ValidationError

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Robust asynchronous Gemini client supporting text, audio, images, video, and documents.
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.8-flash",
        client: Optional[Any] = None,
        disable_safety_filters: bool = True
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.model_name = model_name.strip() if model_name else "gemini-3.8-flash"
        self.disable_safety_filters = disable_safety_filters

        if client:
            self.client = client
        elif self.api_key and GENAI_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

        # Fallback models in priority order
        self.fallback_models = [
            self.model_name,
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
        ]

    def _get_safety_settings(self) -> list:
        """Returns relaxed safety settings disabling aggressive content censorship."""
        if not GENAI_AVAILABLE:
            return []
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    def _sync_generate(
        self,
        model: str,
        contents: list,
        system_instruction: Optional[str] = None,
        enable_search: bool = False,
    ) -> str:
        """Synchronous wrapper for generate_content with optional Google Search grounding."""
        if not self.client:
            raise GeminiAPIError("GenAI client is not initialized.")

        config = types.GenerateContentConfig()
        if system_instruction:
            config.system_instruction = system_instruction

        if self.disable_safety_filters:
            config.safety_settings = self._get_safety_settings()

        if enable_search:
            config.tools = [{"google_search": {}}]

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        main_text = getattr(response, "text", "") or ""

        # Fallback to direct candidate parts if response.text returned None/empty
        if not main_text and hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                parts_text = [getattr(p, "text", "") for p in cand.content.parts if getattr(p, "text", "")]
                if parts_text:
                    main_text = "".join(parts_text)

            if not main_text:
                finish_reason = getattr(cand, "finish_reason", None)
                finish_name = getattr(finish_reason, "name", str(finish_reason)) if finish_reason else "UNKNOWN"
                logger.warning(f"Gemini candidate returned empty text (finish_reason: {finish_name})")
                if finish_name == "SAFETY":
                    main_text = "⚠️ Phản hồi bị bộ lọc an toàn của Google Gemini từ chối (FinishReason: SAFETY)."
                elif finish_name == "MAX_TOKENS":
                    main_text = "⚠️ Phản hồi bị ngắt do vượt giới hạn độ dài (FinishReason: MAX_TOKENS)."
                elif finish_name == "RECITATION":
                    main_text = "⚠️ Phản hồi bị chặn do phát hiện nội dung bản quyền (FinishReason: RECITATION)."
                else:
                    main_text = f"Received empty response from {model} (finish_reason: {finish_name})."

        if not main_text:
            main_text = f"Received empty response from {model}."

        # Format and append sources if search was enabled and grounding metadata exists
        if enable_search and hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            metadata = getattr(candidate, "grounding_metadata", None)
            if metadata:
                chunks = getattr(metadata, "grounding_chunks", None)
                if chunks:
                    sources = []
                    seen_urls = set()
                    for chunk in chunks:
                        web = getattr(chunk, "web", None)
                        if web and getattr(web, "uri", None):
                            uri = web.uri
                            if uri not in seen_urls:
                                seen_urls.add(uri)
                                title = getattr(web, "title", uri) or uri
                                sources.append(f"• [{title}]({uri})")
                    if sources:
                        main_text += "\n\n🌐 **Nguồn tra cứu:**\n" + "\n".join(sources[:5])

        return main_text

    async def generate_multimodal(
        self,
        prompt: str,
        media_path: Optional[Path] = None,
        system_instruction: Optional[str] = None,
        chat_history: Optional[List[Union[ChatMessage, dict]]] = None,
        model_override: Optional[str] = None,
        enable_search: bool = False,
    ) -> str:
        """
        Ultra-fast async multimodal generation with inline byte transmission
        and automatic 503/429 high-demand fallback.
        """
        if not self.api_key and not self.client:
            return "⚠️ Error: GEMINI_API_KEY is not configured in .env."

        # Format short history
        history_context = ""
        if chat_history:
            history_context = "\n[Recent Chat History]\n"
            for turn in chat_history[-4:]:
                role = turn.role if isinstance(turn, ChatMessage) else turn.get("role", "user")
                content = turn.content if isinstance(turn, ChatMessage) else turn.get("content", "")
                role_label = "User" if role == "user" else "Assistant"
                history_context += f"{role_label}: {content}\n"
            history_context += "[End History]\n\n"

        full_prompt = (history_context + (prompt or "")).strip()
        contents = []
        uploaded_file = None

        try:
            # Inline bytes for files < 15MB (sub-second transmission)
            if media_path and media_path.exists():
                file_size = media_path.stat().st_size
                mime_type, _ = mimetypes.guess_type(str(media_path))
                if media_path.suffix.lower() in [".oga", ".ogg"]:
                    mime_type = "audio/ogg"

                if file_size < 15 * 1024 * 1024:
                    with open(media_path, "rb") as f:
                        file_bytes = f.read()
                    contents.append(
                        types.Part.from_bytes(
                            data=file_bytes,
                            mime_type=mime_type or "application/octet-stream"
                        )
                    )
                else:
                    logger.info(f"Uploading large file {media_path} ({file_size} bytes) to File API...")
                    uploaded_file = await asyncio.to_thread(
                        self.client.files.upload,
                        file=str(media_path),
                        config=types.UploadFileConfig(mime_type=mime_type) if mime_type else None
                    )
                    contents.append(uploaded_file)

            contents.append(full_prompt if full_prompt else "Please analyze this.")

            # Determine candidate models to try
            candidates = [model_override] if model_override else list(dict.fromkeys(self.fallback_models))

            last_err = None
            last_empty_result = None
            for candidate in candidates:
                try:
                    result = await asyncio.to_thread(
                        self._sync_generate,
                        candidate,
                        contents,
                        system_instruction,
                        enable_search,
                    )
                    # If valid response is returned, return it
                    if result and not result.startswith("Received empty response") and not result.startswith("⚠️ Phản hồi bị"):
                        return result

                    logger.warning(f"Model {candidate} returned empty/blocked response ('{result[:60]}'). Trying fallback...")
                    last_empty_result = result
                except Exception as e:
                    err_str = str(e)
                    if any(code in err_str for code in ["503", "429", "high demand", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND"]):
                        logger.warning(f"Model {candidate} busy/unavailable ({err_str}). Trying fast fallback...")
                        last_err = e
                        continue
                    else:
                        raise e

            if last_empty_result:
                return last_empty_result

            if last_err:
                raise last_err

        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            return f"❌ Error communicating with Gemini API: {str(e)}"

        finally:
            if uploaded_file and hasattr(uploaded_file, "name") and self.client:
                try:
                    await asyncio.to_thread(self.client.files.delete, name=uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {uploaded_file.name}: {e}")
