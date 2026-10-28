import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from core.webhook_server import WebhookServer


@pytest.mark.asyncio
async def test_webhook_health():
    server = WebhookServer(port=0)
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "own-ai-webhook"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_webhook_download_post_json():
    received_msgs = []

    async def mock_notifier(msg: str):
        received_msgs.append(msg)

    server = WebhookServer(port=0, notifier=mock_notifier)
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        resp = await client.post(
            "/webhook/download",
            json={"name": "Ubuntu.iso", "size": "4.2 GB", "status": "Xong"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["delivered"] is True
        assert len(received_msgs) == 1
        assert "Ubuntu.iso" in received_msgs[0]
        assert "4.2 GB" in received_msgs[0]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_webhook_download_with_secret_authorized():
    received_msgs = []

    async def mock_notifier(msg: str):
        received_msgs.append(msg)

    server = WebhookServer(port=0, secret="super-secret", notifier=mock_notifier)
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        # Unauthorized without secret
        resp = await client.post("/webhook/download", json={"name": "Test"})
        assert resp.status == 401

        # Authorized with header
        resp2 = await client.post(
            "/webhook/download",
            headers={"X-Webhook-Secret": "super-secret"},
            json={"name": "TestSecret", "size": "100MB"},
        )
        assert resp2.status == 200
        assert len(received_msgs) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_webhook_download_get():
    received_msgs = []

    async def mock_notifier(msg: str):
        received_msgs.append(msg)

    server = WebhookServer(port=0, notifier=mock_notifier)
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        resp = await client.get("/webhook/download?name=video.mp4&size=500MB")
        assert resp.status == 200
        assert len(received_msgs) == 1
        assert "video.mp4" in received_msgs[0]
    finally:
        await client.close()
