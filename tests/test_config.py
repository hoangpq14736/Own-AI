import pytest
from pathlib import Path
from config import Settings, ConfigurationError, load_settings_from_env

def test_settings_default_values():
    s = Settings()
    assert s.model_name == "gemini-3.8-flash"
    assert isinstance(s.allowed_user_ids, tuple)
    assert len(s.allowed_user_ids) == 0

def test_settings_allowed_user_ids_parsing():
    # String with spaces and commas
    s = Settings(allowed_user_ids="12345, 67890, abc, 999")
    assert s.allowed_user_ids == (12345, 67890, 999)

    # List of ints
    s2 = Settings(allowed_user_ids=[111, 222])
    assert s2.allowed_user_ids == (111, 222)

def test_settings_immutability():
    s = Settings(model_name="gemini-3.8-flash")
    with pytest.raises(Exception):
        s.model_name = "other-model"

def test_settings_validation_missing_keys():
    s = Settings(gemini_api_key="", telegram_bot_token="")
    with pytest.raises(ConfigurationError):
        s.validate_for_bot()

    with pytest.raises(ConfigurationError):
        s.validate_for_gemini()

def test_settings_validation_success():
    s = Settings(gemini_api_key="fake-gemini-key", telegram_bot_token="fake-bot-token")
    # Should not raise
    s.validate_for_bot()
    s.validate_for_gemini()

def test_load_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-456")
    monkeypatch.setenv("MODEL_NAME", "gemini-test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "custom_data"))
    monkeypatch.setenv("ALLOWED_USER_IDS", "1001,1002")

    loaded = load_settings_from_env()
    assert loaded.gemini_api_key == "test-key-123"
    assert loaded.telegram_bot_token == "test-token-456"
    assert loaded.model_name == "gemini-test"
    assert loaded.allowed_user_ids == (1001, 1002)
    assert loaded.data_dir.exists()

def test_admin_chat_id_group_negative(monkeypatch):
    # Telegram supergroups have negative IDs, e.g. -1001234567890
    monkeypatch.setenv("ADMIN_CHAT_ID", "-1001234567890")
    s = load_settings_from_env()
    assert s.admin_chat_id == -1001234567890
    assert s.get_admin_chat_id() == -1001234567890

def test_enable_group_free_chat(monkeypatch):
    monkeypatch.setenv("ENABLE_GROUP_FREE_CHAT", "true")
    s = load_settings_from_env()
    assert s.enable_group_free_chat is True

def test_disable_safety_filters_config(monkeypatch):
    # Default is True
    s_default = Settings()
    assert s_default.disable_safety_filters is True

    # From env true
    monkeypatch.setenv("DISABLE_SAFETY_FILTERS", "true")
    s_true = load_settings_from_env()
    assert s_true.disable_safety_filters is True

    # From env false
    monkeypatch.setenv("DISABLE_SAFETY_FILTERS", "false")
    s_false = load_settings_from_env()
    assert s_false.disable_safety_filters is False

