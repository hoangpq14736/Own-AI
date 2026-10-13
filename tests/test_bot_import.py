"""
Test to ensure bot.py imports and defines all required dependencies without NameError.
"""
import pytest

def test_bot_imports_and_symbols():
    import bot
    assert hasattr(bot, "SchedulerService")
    assert hasattr(bot, "create_daily_health_report")
    assert hasattr(bot, "create_memory_backup")
    assert hasattr(bot, "safe_edit_text")
    assert hasattr(bot, "send_safe_response")
    assert hasattr(bot, "resolve_mltb_target")
    assert hasattr(bot, "check_update_command")
    assert hasattr(bot, "update_command")
    assert hasattr(bot, "userbot_command")
    assert hasattr(bot, "userbot")
    assert hasattr(bot, "post_init_hook")
    assert hasattr(bot, "post_shutdown_hook")
