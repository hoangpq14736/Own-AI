"""
Web Content Fetcher and Cleaner for Document Summarization.
Extracts clean, readable text from web URLs with strict timeouts and error handling per ECC.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Optional
import aiohttp

from core.exceptions import AppError

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

@dataclass(frozen=True)
class WebPageContent:
    url: str
    title: str
    text: str
    length: int
    error: Optional[str] = None


def extract_clean_text_from_html(html: str) -> Tuple[str, str]:
    """
    Strips script, style, navigation and formatting tags, extracting title and clean content text.
    """
    if not html:
        return "", ""

    # Extract title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "No Title"
    title = re.sub(r"\s+", " ", title)

    # Remove script, style, head, header, footer, nav, noscript
    cleaned = re.sub(r"<(script|style|header|footer|nav|noscript)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    # Remove HTML comments
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
    # Replace block level tags with newlines
    cleaned = re.sub(r"</?(p|div|h[1-6]|li|tr|article|section)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Unescape common HTML entities
    cleaned = (
        cleaned.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Clean up multi-spaces and excessive newlines
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    full_text = "\n".join(lines)
    return title, full_text


async def fetch_web_page(url: str, timeout_seconds: float = 15.0) -> WebPageContent:
    """
    Asynchronously downloads and parses clean text from a web URL.
    """
    if not url or not url.startswith(("http://", "https://")):
        return WebPageContent(
            url=url,
            title="",
            text="",
            length=0,
            error="URL không hợp lệ. Phải bắt đầu bằng http:// hoặc https://"
        )

    headers = {"User-Agent": USER_AGENT}
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return WebPageContent(
                        url=url,
                        title="",
                        text="",
                        length=0,
                        error=f"Không thể truy cập trang web (Mã phản hồi: {resp.status})."
                    )

                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return WebPageContent(
                        url=url,
                        title="",
                        text="",
                        length=0,
                        error=f"Định dạng nội dung không được hỗ trợ ({content_type}). Chỉ hỗ trợ HTML hoặc Text."
                    )

                html = await resp.text(errors="replace")
                title, text = extract_clean_text_from_html(html)

                # Cap at 15,000 characters to keep prompt optimal
                capped_text = text[:15000] if len(text) > 15000 else text

                return WebPageContent(
                    url=url,
                    title=title,
                    text=capped_text,
                    length=len(capped_text),
                    error=None
                )

    except aiohttp.ClientConnectorError as e:
        return WebPageContent(url=url, title="", text="", length=0, error=f"Lỗi kết nối tới máy chủ: {e}")
    except asyncio.TimeoutError:
        return WebPageContent(url=url, title="", text="", length=0, error=f"Quá thời gian tải trang ({timeout_seconds}s timeout).")
    except Exception as e:
        return WebPageContent(url=url, title="", text="", length=0, error=f"Lỗi tải bài viết: {str(e)}")
