"""
Application Configuration and Settings Module.
Implements schema-based boundary validation and immutability according to ECC standards.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from core.exceptions import ConfigurationError

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    """
    Immutable Application Settings.
    Validates environment configuration at startup.
    """
    gemini_api_key: str = Field(default="")
    telegram_bot_token: str = Field(default="")
    model_name: str = Field(default="gemini-3.8-flash")
    data_dir: Path = Field(default=BASE_DIR / "data")
    media_dir: Path = Field(default=BASE_DIR / "data" / "media")
    db_path: Path = Field(default=BASE_DIR / "data" / "memory.db")
    allowed_user_ids: Tuple[int, ...] = Field(default_factory=tuple)
    daily_report_hour: int = Field(default=8)
    enable_daily_report: bool = Field(default=True)
    enable_auto_backup: bool = Field(default=True)
    admin_chat_id: Optional[int] = Field(default=None)
    auto_update_on_restart: bool = Field(default=True)
    git_branch: str = Field(default="main")
    git_repo_url: str = Field(default="https://github.com/hoangpq14736/Own-AI.git")
    enable_group_free_chat: bool = Field(default=False)

    model_config = {
        "frozen": True,
        "arbitrary_types_allowed": True
    }

    def get_admin_chat_id(self) -> Optional[int]:
        """Returns the target chat ID for automated reports and backups."""
        if self.admin_chat_id is not None:
            return self.admin_chat_id
        if self.allowed_user_ids:
            return self.allowed_user_ids[0]
        return None

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, v) -> Tuple[int, ...]:
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        if isinstance(v, str):
            cleaned = [int(uid.strip()) for uid in v.split(",") if uid.strip().isdigit()]
            return tuple(cleaned)
        return tuple()

    def validate_for_bot(self) -> None:
        """Validates that all critical secrets for running the Telegram bot are configured."""
        if not self.telegram_bot_token.strip():
            raise ConfigurationError(
                "TELEGRAM_BOT_TOKEN is missing or empty. Please set it in your .env file."
            )
        if not self.gemini_api_key.strip():
            raise ConfigurationError(
                "GEMINI_API_KEY is missing or empty. Please set it in your .env file."
            )

    def validate_for_gemini(self) -> None:
        """Validates that Gemini API key is configured."""
        if not self.gemini_api_key.strip():
            raise ConfigurationError(
                "GEMINI_API_KEY is missing or empty. Please set it in your .env file."
            )


def load_settings_from_env() -> Settings:
    """Loads and validates settings from environment variables."""
    raw_data_dir = os.getenv("DATA_DIR", "./data")
    data_path = Path(raw_data_dir)
    if not data_path.is_absolute():
        data_path = (BASE_DIR / data_path).resolve()

    media_path = data_path / "media"
    data_path.mkdir(parents=True, exist_ok=True)
    media_path.mkdir(parents=True, exist_ok=True)

    raw_ids = os.getenv("ALLOWED_USER_IDS", "").strip()
    raw_admin = os.getenv("ADMIN_CHAT_ID", "").strip()
    admin_id = None
    if raw_admin:
        try:
            admin_id = int(raw_admin)
        except ValueError:
            admin_id = None

    daily_hour = int(os.getenv("DAILY_REPORT_HOUR", "8").strip() or 8)
    enable_report = os.getenv("ENABLE_DAILY_REPORT", "true").lower() in ("true", "1", "yes")
    enable_backup = os.getenv("ENABLE_AUTO_BACKUP", "true").lower() in ("true", "1", "yes")
    auto_update = os.getenv("AUTO_UPDATE_ON_RESTART", "true").lower() in ("true", "1", "yes")
    git_branch = os.getenv("GIT_BRANCH", "main").strip() or "main"
    git_repo_url = os.getenv("GIT_REPO_URL", "https://github.com/hoangpq14736/Own-AI.git").strip()
    enable_group_free = os.getenv("ENABLE_GROUP_FREE_CHAT", "false").lower() in ("true", "1", "yes")

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        model_name=os.getenv("MODEL_NAME", "gemini-3.8-flash").strip(),
        data_dir=data_path,
        media_dir=media_path,
        db_path=data_path / "memory.db",
        allowed_user_ids=raw_ids,
        daily_report_hour=daily_hour,
        enable_daily_report=enable_report,
        enable_auto_backup=enable_backup,
        admin_chat_id=admin_id,
        auto_update_on_restart=auto_update,
        git_branch=git_branch,
        git_repo_url=git_repo_url,
        enable_group_free_chat=enable_group_free,
    )


# Active singleton settings instance
settings = load_settings_from_env()

# Backward-compatible global exports
GEMINI_API_KEY = settings.gemini_api_key
TELEGRAM_BOT_TOKEN = settings.telegram_bot_token
MODEL_NAME = settings.model_name
DATA_DIR = settings.data_dir
MEDIA_DIR = settings.media_dir
DB_PATH = settings.db_path
ALLOWED_USER_IDS = list(settings.allowed_user_ids)
ADMIN_CHAT_ID = settings.get_admin_chat_id()
AUTO_UPDATE_ON_RESTART = settings.auto_update_on_restart
GIT_BRANCH = settings.git_branch
ENABLE_GROUP_FREE_CHAT = settings.enable_group_free_chat
