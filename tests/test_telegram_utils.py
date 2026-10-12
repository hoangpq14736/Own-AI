import pytest
from core.telegram_utils import (
    strip_bot_mention,
    is_message_targeted_at_bot,
    can_user_execute_command,
    extract_mltb_command,
)

def test_strip_bot_mention():
    bot_user = "MyAssistantBot"
    assert strip_bot_mention("@MyAssistantBot what is 2+2?", bot_user) == "what is 2+2?"
    assert strip_bot_mention("Hello @MyAssistantBot, please help", bot_user) == "Hello , please help"
    assert strip_bot_mention("No mention here", bot_user) == "No mention here"
    assert strip_bot_mention("", bot_user) == ""
    assert strip_bot_mention("some text", None) == "some text"

def test_is_message_targeted_at_bot_private_chat():
    # Private chat messages are always targeted
    assert is_message_targeted_at_bot(chat_type="private", text="Hi there") is True
    assert is_message_targeted_at_bot(chat_type="sender", text="Hi there") is True

def test_is_message_targeted_at_bot_group_chat():
    bot_user = "MyAssistantBot"

    # Casual message without mention in group -> False
    assert is_message_targeted_at_bot(
        chat_type="supergroup",
        text="What's for lunch everyone?",
        bot_username=bot_user,
        is_reply_to_bot=False
    ) is False

    # Message with mention -> True
    assert is_message_targeted_at_bot(
        chat_type="group",
        text="@MyAssistantBot prepare a mirror command",
        bot_username=bot_user,
        is_reply_to_bot=False
    ) is True

    # Caption with mention -> True
    assert is_message_targeted_at_bot(
        chat_type="supergroup",
        text="",
        caption="Check this photo @myassistantbot",
        bot_username=bot_user,
        is_reply_to_bot=False
    ) is True

    # Reply to bot without mention -> True
    assert is_message_targeted_at_bot(
        chat_type="supergroup",
        text="Yes, proceed with that",
        bot_username=bot_user,
        is_reply_to_bot=True
    ) is True

    # Admin / Control Room Chat without mention -> True
    assert is_message_targeted_at_bot(
        chat_type="supergroup",
        text="leech this file for me https://example.com/test",
        bot_username=bot_user,
        is_reply_to_bot=False,
        is_admin_chat=True,
    ) is True

    # Free Chat Mode in groups without mention -> True
    assert is_message_targeted_at_bot(
        chat_type="group",
        text="can you reply?",
        bot_username=bot_user,
        is_reply_to_bot=False,
        enable_free_chat=True,
    ) is True

    # Empty text/caption without reply -> False
    assert is_message_targeted_at_bot(
        chat_type="supergroup",
        text=None,
        caption=None,
        bot_username=bot_user,
        is_reply_to_bot=False
    ) is False

def test_can_user_execute_command():
    allowed = (12345, 67890)
    assert can_user_execute_command(12345, allowed) is True
    assert can_user_execute_command(99999, allowed) is False

    # Empty allowed tuple allows all
    assert can_user_execute_command(99999, tuple()) is True


def test_extract_mltb_command_from_text():
    # Inside markdown code block
    text1 = "Here is your leech command:\n```text\n/leech https://drive.google.com/file/d/123/edit\n```"
    assert extract_mltb_command(text=text1) == "/leech https://drive.google.com/file/d/123/edit"

    # Command with flags
    text2 = "/mirror magnet:?xt=urn:btih:abc12345 -z -s"
    assert extract_mltb_command(text=text2) == "/mirror magnet:?xt=urn:btih:abc12345 -z -s"

    # Conversational text without MLTB command
    text3 = "Sure, I can help you with your coding problem!"
    assert extract_mltb_command(text=text3) is None


def test_extract_mltb_command_from_user_prompt():
    # Direct user prompt with URL
    prompt1 = "leech this file for me https://drive.google.com/file/d/1ydXWtUGagoTzBWf5Iz0yaVfZLnWUWCYf/edit"
    assert extract_mltb_command(user_prompt=prompt1) == "/leech https://drive.google.com/file/d/1ydXWtUGagoTzBWf5Iz0yaVfZLnWUWCYf/edit"

    # User prompt asking for mirror and zip
    prompt2 = "please mirror this file https://example.com/movie.mkv with zip"
    assert extract_mltb_command(user_prompt=prompt2) == "/mirror https://example.com/movie.mkv -z"

    # User prompt replying to media
    prompt3 = "leech this picture for me"
    assert extract_mltb_command(user_prompt=prompt3, has_reply_media=True) == "/leech"

    # User prompt replying to media for mirror
    assert extract_mltb_command(user_prompt="mirror this photo", has_reply_media=True) == "/mirror"

    # User prompt asking for clone
    prompt_clone = "clone this folder https://drive.google.com/drive/folders/abc123"
    assert extract_mltb_command(user_prompt=prompt_clone) == "/clone https://drive.google.com/drive/folders/abc123"

    # User prompt asking for ytdl with unzip and select flags
    prompt_ytdl = "ytdl https://youtube.com/watch?v=123 unzip and select files"
    assert extract_mltb_command(user_prompt=prompt_ytdl) == "/ytdl https://youtube.com/watch?v=123 -uz -s"

    # User prompt without media reply
    prompt4 = "leech this picture for me"
    assert extract_mltb_command(user_prompt=prompt4, has_reply_media=False) is None
