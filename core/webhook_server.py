"""
Lightweight Webhook Server for external download notifications (Aria2, qBittorrent, etc.).
Runs asynchronously inside the existing asyncio loop without blocking the bot.
"""

import logging
from typing import Callable, Awaitable, Optional, Dict, Any
from aiohttp import web

logger = logging.getLogger(__name__)


class WebhookServer:
    """
    HTTP Webhook Server for receiving download notifications from Aria2, qBittorrent, etc.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        secret: str = "",
        notifier: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.host = host
        self.port = port
        self.secret = secret
        self.notifier = notifier
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_post("/webhook/download", self.handle_download)
        self.app.router.add_get("/webhook/download", self.handle_download)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok", "service": "own-ai-webhook"})

    async def handle_download(self, request: web.Request) -> web.Response:
        """Handles download complete notifications."""
        if self.secret:
            token = (
                request.headers.get("X-Webhook-Secret")
                or request.query.get("secret", "")
            )
            if token != self.secret:
                return web.json_response({"error": "Unauthorized"}, status=401)

        data: Dict[str, Any] = {}
        if request.method == "POST":
            try:
                data = await request.json()
            except Exception:
                try:
                    post_data = await request.post()
                    data = dict(post_data)
                except Exception:
                    data = {}
        else:
            data = dict(request.query)

        name = (
            data.get("name")
            or data.get("file")
            or data.get("title")
            or "Tệp tin không tên"
        )
        size = data.get("size") or "Không rõ kích thước"
        status = data.get("status", "Hoàn thành")

        message = (
            f"📥 *Thông Báo Tải Về ({status})*\n\n"
            f"📦 *Tệp:* `{name}`\n"
            f"💾 *Dung lượng:* `{size}`"
        )

        if self.notifier:
            try:
                await self.notifier(message)
            except Exception as e:
                logger.error(f"Failed to dispatch Telegram notification: {e}")
                return web.json_response({"status": "error", "message": str(e)}, status=500)

        return web.json_response({"status": "ok", "delivered": True})

    async def start(self) -> None:
        """Starts the aiohttp TCPSite server."""
        if self.port <= 0:
            logger.info("Webhook server disabled (port <= 0).")
            return
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info(f"Webhook server running on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Gracefully shuts down the webhook runner."""
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
            logger.info("Webhook server stopped.")
