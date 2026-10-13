"""
Telegram Userbot StringSession Generator.
Generates an authenticated TELEGRAM_USER_SESSION string using Telethon.
Run this script locally or on your VPS:
    python scripts/generate_session.py
"""

import sys

try:
    from telethon.sync import TelegramClient
    from telethon.sessions import StringSession
except ImportError:
    print("Error: telethon is not installed.")
    print("Please install it by running: pip install telethon")
    sys.exit(1)


def main():
    print("=" * 65)
    print("   🤖 Own AI — Telegram Userbot StringSession Generator")
    print("=" * 65)
    print("Get your API_ID and API_HASH from https://my.telegram.org")
    print("-" * 65)

    api_id_input = input("1. Enter your Telegram API_ID: ").strip()
    if not api_id_input.isdigit():
        print("❌ Error: API_ID must be a numeric integer.")
        sys.exit(1)
    api_id = int(api_id_input)

    api_hash = input("2. Enter your Telegram API_HASH: ").strip()
    if not api_hash:
        print("❌ Error: API_HASH cannot be empty.")
        sys.exit(1)

    print("\nConnecting to Telegram servers...")
    print("You will be asked for your phone number (e.g. +84912345678) and the Telegram login code.\n")

    try:
        with TelegramClient(StringSession(), api_id, api_hash) as client:
            session_str = client.session.save()
            me = client.get_me()
            username = getattr(me, "username", None) or getattr(me, "first_name", "User")

            print("\n" + "=" * 65)
            print(f"✅ SUCCESS! Authenticated as: @{username} (ID: {me.id})")
            print("=" * 65)
            print("Copy the lines below and add them to your .env file on VPS:\n")
            print(f"TELEGRAM_API_ID={api_id}")
            print(f"TELEGRAM_API_HASH={api_hash}")
            print(f"TELEGRAM_USER_SESSION={session_str}")
            print("=" * 65)
            print("⚠️ Keep your TELEGRAM_USER_SESSION secret! Do not share it publicly.")
    except Exception as e:
        print(f"\n❌ Failed to generate session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
