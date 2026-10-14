import pytest
from core.memory import MemoryEngine
from core.exceptions import ValidationError

@pytest.fixture
def temp_memory(tmp_path):
    db_path = tmp_path / "test_memory.db"
    return MemoryEngine(db_path)

def test_add_and_list_rules(temp_memory):
    user_id = "test_user_123"
    rule_id = temp_memory.add_rule(
        user_id=user_id,
        rule="Always respond using bullet points",
        category="formatting",
        trigger_context="when summarizing"
    )
    assert rule_id is not None

    rules = temp_memory.list_rules(user_id)
    assert len(rules) == 1
    assert rules[0].rule == "Always respond using bullet points"
    assert rules[0]["rule"] == "Always respond using bullet points"
    assert rules[0].category == "formatting"

def test_prevent_duplicate_rules(temp_memory):
    user_id = "test_user_123"
    id1 = temp_memory.add_rule(user_id, "Always speak English")
    id2 = temp_memory.add_rule(user_id, "Always speak English")
    assert id1 == id2
    assert len(temp_memory.list_rules(user_id)) == 1

def test_delete_rule(temp_memory):
    user_id = "test_user_123"
    rule_id = temp_memory.add_rule(user_id, "Temporary rule")
    assert len(temp_memory.list_rules(user_id)) == 1

    deleted = temp_memory.delete_rule(user_id, rule_id)
    assert deleted is True
    assert len(temp_memory.list_rules(user_id)) == 0

    # Delete non-existent rule
    assert temp_memory.delete_rule(user_id, 9999) is False
    assert temp_memory.delete_rule("", 1) is False

def test_clear_rules(temp_memory):
    user_id = "test_user_clear"
    temp_memory.add_rule(user_id, "Rule 1")
    temp_memory.add_rule(user_id, "Rule 2")
    assert len(temp_memory.list_rules(user_id)) == 2

    cleared = temp_memory.clear_rules(user_id)
    assert cleared == 2
    assert len(temp_memory.list_rules(user_id)) == 0
    assert temp_memory.clear_rules("") == 0

def test_facts_management(temp_memory):
    user_id = "test_user_facts"
    fid1 = temp_memory.add_fact(user_id, "Loves Linux")
    fid2 = temp_memory.add_fact(user_id, "Based in Hanoi")

    facts = temp_memory.list_facts(user_id)
    assert facts == ["Loves Linux", "Based in Hanoi"]

    detailed = temp_memory.list_facts_detailed(user_id)
    assert len(detailed) == 2
    assert detailed[0].fact == "Loves Linux"
    assert detailed[0].user_id == user_id

    assert temp_memory.list_facts("") == []
    assert temp_memory.list_facts_detailed("") == []

def test_build_memory_prompt(temp_memory):
    user_id = "test_user_123"
    temp_memory.add_rule(user_id, "Always format code in Python", category="coding", trigger_context="when coding")
    temp_memory.add_fact(user_id, "User is a cloud engineer")

    prompt = temp_memory.build_memory_system_prompt(user_id)
    assert "Always format code in Python" in prompt
    assert "User is a cloud engineer" in prompt
    assert "[PERSISTENT MEMORY & SELF-LEARNED RULES]" in prompt
    assert "[PERSONAL COMPANION DIRECTIVE]" in prompt

    # Empty user has base MLTB prompt and companion directive without persistent memory block
    empty_prompt = temp_memory.build_memory_system_prompt("non_existent_user")
    assert "[MLTB KNOWLEDGE BASE]" in empty_prompt
    assert "[PERSONAL COMPANION DIRECTIVE]" in empty_prompt
    assert "[PERSISTENT MEMORY & SELF-LEARNED RULES]" not in empty_prompt

def test_chat_history(temp_memory):
    user_id = "test_user_123"
    temp_memory.add_history(user_id, "user", "Hello there!")
    temp_memory.add_history(user_id, "assistant", "Hi! How can I help?")

    history = temp_memory.get_recent_history(user_id, limit=5)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0]["role"] == "user"
    assert history[1].role == "assistant"

    # Limit and empty tests
    assert temp_memory.get_recent_history(user_id, limit=0) == []
    assert temp_memory.get_recent_history("", limit=5) == []

    # Clear history
    cleared = temp_memory.clear_history(user_id)
    assert cleared == 2
    assert len(temp_memory.get_recent_history(user_id)) == 0
    assert temp_memory.clear_history("") == 0

def test_validation_errors(temp_memory):
    with pytest.raises(ValidationError):
        temp_memory.add_rule("", "Some rule")
    with pytest.raises(ValidationError):
        temp_memory.add_rule("user", "")

    with pytest.raises(ValidationError):
        temp_memory.add_fact("", "Some fact")
    with pytest.raises(ValidationError):
        temp_memory.add_fact("user", "")

    with pytest.raises(ValidationError):
        temp_memory.add_history("", "user", "content")
