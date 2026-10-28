import pytest
from unittest.mock import MagicMock
from core.gemini_client import GeminiClient


@pytest.mark.asyncio
async def test_get_embedding_uninitialized():
    client = GeminiClient(api_key="")
    emb = await client.get_embedding("test text")
    assert emb == []


@pytest.mark.asyncio
async def test_get_embedding_empty_text():
    mock_sdk = MagicMock()
    client = GeminiClient(api_key="fake-key", client=mock_sdk)
    emb = await client.get_embedding("")
    assert emb == []


@pytest.mark.asyncio
async def test_get_embedding_success():
    mock_sdk = MagicMock()
    mock_resp = MagicMock()
    mock_item = MagicMock()
    mock_item.values = [0.12, 0.34, 0.56]
    mock_resp.embeddings = [mock_item]

    mock_sdk.models.embed_content.return_value = mock_resp
    client = GeminiClient(api_key="fake-key", client=mock_sdk)

    emb = await client.get_embedding("Hello world")
    assert emb == [0.12, 0.34, 0.56]
    mock_sdk.models.embed_content.assert_called_once_with(
        model="text-embedding-004",
        contents="Hello world",
    )
