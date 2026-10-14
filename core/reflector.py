"""
Background Reflection and Continuous Self-Learning Engine.
Implements the Smart 1-Request pre-filter and outputs immutable ReflectionResult models.
"""

import json
import logging
import re
from typing import Optional, Dict, Any

from core.memory import MemoryEngine
from core.models import ReflectionResult
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Keywords in English and Vietnamese that signal a rule, correction, or personal fact
LEARNING_TRIGGERS_REGEX = re.compile(
    r"\b("
    # English triggers
    r"i am|i'm|my name|i live|i work|from now on|always|never|remember|don't|do not|"
    r"stop doing|prefer|you should|call me|rule|instead of|wrong|that's not|thats not|"
    r"when i|for me|my favorite|my preference|"
    # Vietnamese triggers
    r"tôi là|mình là|tên tôi|tên mình|tôi ở|tôi sống|tôi làm|từ giờ|từ nay|"
    r"nhớ là|nhớ rằng|nên nhớ|hãy nhớ|ghi nhớ|lưu ý|khi nào tôi|luôn luôn|đừng|không được|thay vì|sai rồi|không phải|"
    r"gọi tôi là|quy tắc|sở thích|thích|ghét"
    r")\b",
    re.IGNORECASE
)

REFLECTION_PROMPT = """
You are an intelligent Memory & Self-Learning Engine.
Your job is to analyze the user's latest message to determine if the user is:
1. Correcting the assistant ("No, that's wrong", "Don't do X, do Y instead").
2. Giving a persistent preference or rule ("Always answer in Vietnamese", "Be concise", "Use bullet points").
3. Sharing a lasting personal fact ("My name is Alex", "I am a cloud architect", "I live in Vietnam").

If the user is just asking normal questions, chatting, or saying hello, DO NOT extract anything.

User's Latest Message: "{user_message}"
Assistant's Prior Response: "{last_assistant_response}"

Respond strictly with a valid JSON object matching this schema:
{{
  "should_learn": true/false,
  "type": "rule" | "fact",
  "content": "A clear, concise, actionable sentence summarizing what to remember or follow in future turns",
  "category": "formatting" | "behavior" | "personal" | "general",
  "trigger_context": "Brief condition when this applies (e.g. 'when writing code' or 'general')"
}}
"""


class Reflector:
    """
    Analyzes conversation turns for explicit learning signals and updates persistent memory.
    """
    def __init__(self, gemini_client, memory_engine: MemoryEngine):
        self.client = gemini_client
        self.memory = memory_engine

    def should_trigger_reflection(self, user_message: str) -> bool:
        """
        Smart Pre-Filter:
        Only triggers reflection if the user's message contains learning, preference,
        or correction signals. Normal queries (e.g. 'what is 2+2') bypass reflection completely,
        ensuring casual chats use strictly 1 API call.
        """
        if not user_message or len(user_message.strip()) < 3:
            return False
        return bool(LEARNING_TRIGGERS_REGEX.search(user_message))

    async def reflect_and_learn(
        self,
        user_id: str,
        user_message: str,
        last_assistant_response: str = ""
    ) -> Optional[ReflectionResult]:
        """
        Asynchronously analyzes a conversation turn and saves any new rule or fact to SQLite.
        Only runs if the pre-filter indicates potential learning signals.
        Returns an immutable ReflectionResult instance.
        """
        if not user_id or not user_id.strip():
            return None

        if not self.should_trigger_reflection(user_message):
            return None

        clean_uid = user_id.strip()
        logger.info(f"Learning signals detected in message: '{user_message[:50]}...'. Running reflection...")

        prompt = REFLECTION_PROMPT.format(
            user_message=user_message,
            last_assistant_response=last_assistant_response
        )

        try:
            raw_response = await self.client.generate_multimodal(
                prompt=prompt,
                system_instruction="You are a strict JSON extractor. Only output valid JSON without backticks.",
                model_override="gemini-3.5-flash-lite"
            )

            # Strip possible markdown code fences or extract JSON substring
            cleaned = raw_response.strip()
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            data = json.loads(cleaned)

            if data.get("should_learn") and data.get("content"):
                item_type = data.get("type", "rule").lower()
                content = str(data.get("content", "")).strip()
                category = str(data.get("category", "general")).strip()
                trigger = str(data.get("trigger_context", "general")).strip()

                if not content:
                    return None

                if item_type == "rule":
                    rule_id = self.memory.add_rule(
                        user_id=clean_uid,
                        rule=content,
                        category=category,
                        trigger_context=trigger
                    )
                    logger.info(f"Learned rule #{rule_id} for user {clean_uid}: {content}")
                    return ReflectionResult(
                        should_learn=True,
                        type="rule",
                        content=content,
                        category=category,
                        trigger_context=trigger,
                        id=rule_id
                    )

                elif item_type == "fact":
                    fact_id = self.memory.add_fact(
                        user_id=clean_uid,
                        fact=content
                    )
                    logger.info(f"Learned fact #{fact_id} for user {clean_uid}: {content}")
                    return ReflectionResult(
                        should_learn=True,
                        type="fact",
                        content=content,
                        category=category,
                        trigger_context=trigger,
                        id=fact_id
                    )

        except Exception as e:
            logger.debug(f"Reflection did not extract any rules: {e}")

        return None
