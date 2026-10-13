"""
Userbot Helper Module.
Enables sending commands (e.g. /leech, /mirror) as an authentic Telegram User account
using Telethon, overcoming Telegram's restriction where bots cannot see or trigger other bots.
"""

import logging
from typing import Optional, Any

try:
    from telethon import TelegramClient
    from telethon.sessions import StringSession
    TELETHON_AVAILABLE = True
except ImportError:
    TelegramClient = None  # type: ignore
    StringSession = None    # type: ignore
    TELETHON_AVAILABLE = False

logger = logging.getLogger(__name__)


class UserbotClient:
    """
    Manages an optional Telethon client acting as the user's personal Telegram account.
    """

    def __init__(
        self,
        api_id: Optional[int] = None,
        api_hash: str = "",
        session_string: str = "",
        client: Optional[Any] = None,
    ):
        self.api_id = api_id
        self.api_hash = (api_hash or "").strip()
        self.session_string = (session_string or "").strip()
        self._custom_client = client
        self._client: Optional[Any] = None
        self._is_started = False
        self._username: Optional[str] = None
        self._user_id: Optional[int] = None

    @property
    def is_configured(self) -> bool:
        """Returns True if all required credentials are present."""
        if self._custom_client:
            return True
        return bool(self.api_id and self.api_hash and self.session_string and TELETHON_AVAILABLE)

    @property
    def is_connected(self) -> bool:
        """Returns True if the client is currently connected and active."""
        return self._is_started and self._client is not None

    @property
    def username(self) -> Optional[str]:
        return self._username

    @property
    def user_id(self) -> Optional[int]:
        return self._user_id

    async def start(self) -> bool:
        """
        Starts the Telethon client with the configured StringSession.
        Returns True if connected successfully, False otherwise.
        """
        if not self.is_configured and not self._custom_client:
            logger.info("Userbot is not configured (missing TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_USER_SESSION).")
            return False

        if self._custom_client:
            self._client = self._custom_client
            self._is_started = True
            logger.info("Userbot started with custom injected client.")
            try:
                me = await self._client.get_me()
                self._user_id = getattr(me, "id", None)
                self._username = getattr(me, "username", None) or getattr(me, "first_name", "User")
            except Exception as e:
                logger.debug(f"Unable to get me from mock client: {e}")
            return True

        if not TELETHON_AVAILABLE:
            logger.warning("telethon library is not installed. Userbot cannot start.")
            return False

        try:
            logger.info("Connecting Telethon Userbot client...")
            self._client = TelegramClient(StringSession(self.session_string), self.api_id, self.api_hash)
            await self._client.start()
            self._is_started = True

            me = await self._client.get_me()
            self._user_id = me.id
            self._username = getattr(me, "username", None) or getattr(me, "first_name", "User")
            logger.info(f"✅ Userbot connected successfully as @{self._username} (ID: {self._user_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to start Userbot client: {e}")
            self._client = None
            self._is_started = False
            return False

    async def stop(self) -> None:
        """Disconnects the Telethon client cleanly."""
        if self._client and self._is_started:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting userbot: {e}")
            finally:
                self._is_started = False
                self._client = None
                self._username = None
                self._user_id = None
                logger.info("Userbot client disconnected.")

    async def send_command(
        self,
        chat_id: Any,
        command: str,
        reply_to_msg_id: Optional[int] = None,
    ) -> bool:
        """
        Sends a command message into the specified chat as the authenticated user.
        If reply_to_msg_id is specified, replies to that message.
        Returns True on success, False on error.
        """
        if not self.is_connected:
            logger.warning("Cannot dispatch command: Userbot is not connected.")
            return False

        try:
            logger.info(f"Userbot dispatching command to chat {chat_id}: '{command}' (reply_to={reply_to_msg_id})")
            await self._client.send_message(
                entity=chat_id,
                message=command,
                reply_to=reply_to_msg_id,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send command via Userbot: {e}")
            return False
