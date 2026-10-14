import pytest
from unittest.mock import AsyncMock
from core.memory import MemoryEngine
from core.reflector import Reflector

@pytest.fixture
def setup_reflector(tmp_path):
    db_path = tmp_path / "test_memory.db"
    mem = MemoryEngine(db_path)
    mock_client = AsyncMock()
    ref = Reflector(mock_client, mem)
    return ref, mock_client, mem

@pytest.mark.anyio
async def test_reflector_extracts_rule_with_trigger(setup_reflector):
    reflector, mock_client, memory = setup_reflector
    user_id = "test_user_456"

    # Simulate Gemini returning a structured JSON rule
    mock_client.generate_multimodal.return_value = """
    ```json
    {
      "should_learn": true,
      "type": "rule",
      "content": "Do not summarize with bullet points; use numbered lists instead.",
      "category": "formatting",
      "trigger_context": "summaries"
    }
    ```
    """

    result = await reflector.reflect_and_learn(
        user_id=user_id,
        user_message="From now on, remember that you must use numbered lists!",
        last_assistant_response="Here is a summary: • Item 1 • Item 2"
    )

    assert result is not None
    assert result.type == "rule"
    assert result["type"] == "rule"
    assert "numbered lists" in result.content

    rules = memory.list_rules(user_id)
    assert len(rules) == 1
    assert "numbered lists" in rules[0].rule

@pytest.mark.anyio
async def test_reflector_extracts_fact(setup_reflector):
    reflector, mock_client, memory = setup_reflector
    user_id = "test_user_fact"

    mock_client.generate_multimodal.return_value = """
    {
      "should_learn": true,
      "type": "fact",
      "content": "User is a cloud engineer living in Vietnam",
      "category": "personal"
    }
    """

    result = await reflector.reflect_and_learn(
        user_id=user_id,
        user_message="Remember that I am a cloud engineer living in Vietnam",
        last_assistant_response="Understood!"
    )

    assert result is not None
    assert result.type == "fact"
    assert "Vietnam" in result.content

    facts = memory.list_facts(user_id)
    assert len(facts) == 1
    assert "Vietnam" in facts[0]

@pytest.mark.anyio
async def test_reflector_should_learn_false(setup_reflector):
    reflector, mock_client, memory = setup_reflector
    user_id = "test_user_no_learn"

    mock_client.generate_multimodal.return_value = """
    {
      "should_learn": false,
      "content": ""
    }
    """

    result = await reflector.reflect_and_learn(
        user_id=user_id,
        user_message="I prefer apples today",
        last_assistant_response="Apples are delicious."
    )

    assert result is None
    assert len(memory.list_rules(user_id)) == 0

@pytest.mark.anyio
async def test_reflector_invalid_json(setup_reflector):
    reflector, mock_client, memory = setup_reflector
    user_id = "test_user_invalid"

    mock_client.generate_multimodal.return_value = "Sorry, cannot parse this instruction."

    result = await reflector.reflect_and_learn(
        user_id=user_id,
        user_message="Remember this important thing",
        last_assistant_response=""
    )

    assert result is None

@pytest.mark.anyio
async def test_reflector_bypasses_casual_chat_without_api_call(setup_reflector):
    reflector, mock_client, memory = setup_reflector
    user_id = "test_user_456"

    # Casual question with no learning triggers
    result = await reflector.reflect_and_learn(
        user_id=user_id,
        user_message="What is 2 + 2?",
        last_assistant_response="It is 4."
    )

    # Pre-filter should block it immediately without even calling the API
    assert result is None
    assert mock_client.generate_multimodal.call_count == 0
    assert len(memory.list_rules(user_id)) == 0

@pytest.mark.anyio
async def test_reflector_empty_inputs(setup_reflector):
    reflector, mock_client, memory = setup_reflector
    assert await reflector.reflect_and_learn("", "I am user") is None
    assert await reflector.reflect_and_learn("user", "") is None

def test_should_trigger_reflection(setup_reflector):
    reflector, _, _ = setup_reflector
    # Should trigger
    assert reflector.should_trigger_reflection("I am from Vietnam") is True
    assert reflector.should_trigger_reflection("from now on always use bullet points") is True
    assert reflector.should_trigger_reflection("Tôi là kỹ sư đám mây") is True
    assert reflector.should_trigger_reflection("Nhớ rằng đừng bao giờ dùng dấu chấm than") is True
    assert reflector.should_trigger_reflection("nên nhớ, khi nào tôi bảo đó là nhạc âm thanh thì lưu ở mục Gdrive:/Mirror/Musics") is True

    # Conversational confirmations to an assistant proposal
    assert reflector.should_trigger_reflection(
        user_message="đúng rồi nè",
        last_assistant_response="Bạn có muốn mình ghi nhớ luôn thư mục Gdrive:/Mirror/Documents làm mặc định cho các file tài liệu từ giờ trở đi không?"
    ) is True
    assert reflector.should_trigger_reflection(
        user_message="chốt luôn gái ơi. cám ơn gái nhiều",
        last_assistant_response="Bạn có muốn mình chốt luôn thư mục Gdrive:/Mirror/Music làm mặc định cho các file âm thanh từ giờ trở đi không?"
    ) is True
    assert reflector.should_trigger_reflection(
        user_message="ok em",
        current_assistant_response="Đã ghi nhớ! Từ bây giờ, đối với các file tài liệu, mình sẽ tự động thêm cờ -up Gdrive:/Mirror/Documents."
    ) is True

    # Should NOT trigger (normal chat)
    assert reflector.should_trigger_reflection("what is the weather today?") is False
    assert reflector.should_trigger_reflection("how does photosynthesis work?") is False
    assert reflector.should_trigger_reflection("hello bot") is False
    assert reflector.should_trigger_reflection("") is False
    assert reflector.should_trigger_reflection("hi") is False


@pytest.mark.anyio
async def test_reflector_extracts_rule_from_conversational_confirmation(setup_reflector):
    reflector, mock_client, memory = setup_reflector
    user_id = "test_user_confirm"

    mock_client.generate_multimodal.return_value = """
    {
      "should_learn": true,
      "type": "rule",
      "content": "Save document files to Gdrive:/Mirror/Documents using the -up flag",
      "category": "behavior",
      "trigger_context": "when downloading documents"
    }
    """

    result = await reflector.reflect_and_learn(
        user_id=user_id,
        user_message="đúng rồi nè",
        last_assistant_response="Bạn có muốn mình ghi nhớ luôn thư mục Gdrive:/Mirror/Documents làm mặc định cho các file tài liệu từ giờ trở đi không?",
        current_assistant_response="Đã ghi nhớ! Từ bây giờ, đối với các file tài liệu mình sẽ tự động thêm cờ -up Gdrive:/Mirror/Documents."
    )

    assert result is not None
    assert result.type == "rule"
    assert "Gdrive:/Mirror/Documents" in result.content

    rules = memory.list_rules(user_id)
    assert len(rules) == 1
    assert "Gdrive:/Mirror/Documents" in rules[0].rule

