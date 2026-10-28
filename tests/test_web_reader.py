import pytest
from core.web_reader import extract_clean_text_from_html, fetch_web_page

def test_extract_clean_text_from_html():
    raw_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tin Tức Công Nghệ AI 2026</title>
        <style>body { color: red; }</style>
        <script>console.log('tracker');</script>
    </head>
    <body>
        <header><nav>Menu Home Contact</nav></header>
        <article>
            <h1>Google Ra Mắt Gemini 3.8</h1>
            <p>Hôm nay Google công bố phiên bản Gemini 3.8 với khả năng xử lý đa phương thức vượt trội.</p>
            <p>Model này hỗ trợ lập trình tự động và tối ưu chi phí &amp; token.</p>
        </article>
        <footer>Bản quyền 2026</footer>
    </body>
    </html>
    """
    title, text = extract_clean_text_from_html(raw_html)
    assert title == "Tin Tức Công Nghệ AI 2026"
    assert "Google Ra Mắt Gemini 3.8" in text
    assert "khả năng xử lý đa phương thức" in text
    assert "chi phí & token" in text
    # Should not contain stripped tags
    assert "console.log" not in text
    assert "Menu Home" not in text
    assert "Bản quyền" not in text

@pytest.mark.asyncio
async def test_fetch_web_page_invalid_url():
    result = await fetch_web_page("invalid-url")
    assert result.error is not None
    assert "URL không hợp lệ" in result.error
