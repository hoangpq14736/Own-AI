import pytest
from dataclasses import FrozenInstanceError
from core.models import LearnedRule, UserFact, ChatMessage, VPSMetrics, ReflectionResult

def test_learned_rule_immutability_and_access():
    rule = LearnedRule(
        id=1,
        user_id="user123",
        category="behavior",
        rule="Be concise",
        trigger_context="coding",
        created_at="2026-10-13T00:00:00Z"
    )
    # Attribute access
    assert rule.id == 1
    assert rule.category == "behavior"

    # Dict-like access
    assert rule["rule"] == "Be concise"
    assert rule.get("trigger_context") == "coding"
    assert rule.get("non_existent", "default") == "default"
    assert "category" in rule
    assert "non_existent" not in rule

    # Immutability
    with pytest.raises(FrozenInstanceError):
        rule.rule = "Be verbose"

    # to_dict
    d = rule.to_dict()
    assert d["id"] == 1
    assert d["rule"] == "Be concise"

    # KeyError on bad dict key
    with pytest.raises(KeyError):
        _ = rule["bad_key"]

    # TypeError on non-string key
    with pytest.raises(TypeError):
        _ = rule[0]

def test_user_fact_immutability_and_access():
    fact = UserFact(id=2, user_id="user123", fact="User speaks Vietnamese")
    assert fact["fact"] == "User speaks Vietnamese"
    assert "fact" in fact
    assert fact.get("fact") == "User speaks Vietnamese"
    assert fact.to_dict()["id"] == 2

    with pytest.raises(FrozenInstanceError):
        fact.fact = "Other"

    with pytest.raises(KeyError):
        _ = fact["invalid"]

    with pytest.raises(TypeError):
        _ = fact[0]

def test_chat_message_immutability_and_access():
    msg = ChatMessage(role="user", content="Hello", media_type="text")
    assert msg["role"] == "user"
    assert "content" in msg
    assert msg.get("media_type") == "text"
    assert msg.to_dict()["role"] == "user"

    with pytest.raises(FrozenInstanceError):
        msg.content = "Changed"

    with pytest.raises(KeyError):
        _ = msg["invalid"]

    with pytest.raises(TypeError):
        _ = msg[0]

def test_vps_metrics_immutability_and_access():
    metrics = VPSMetrics(
        cpu_percent=12.5,
        cpu_count=4,
        ram_used_gb=4.2,
        ram_total_gb=24.0,
        ram_percent=17.5,
        disk_used_gb=30.0,
        disk_free_gb=170.0,
        disk_total_gb=200.0,
        disk_percent=15.0,
        uptime="3d 4h 12m",
        warning=None,
        text="Sample stats"
    )
    assert metrics["cpu_percent"] == 12.5
    assert "ram_total_gb" in metrics
    assert metrics.get("uptime") == "3d 4h 12m"
    assert metrics.to_dict()["cpu_count"] == 4

    with pytest.raises(FrozenInstanceError):
        metrics.cpu_percent = 99.0

    with pytest.raises(KeyError):
        _ = metrics["unknown"]

    with pytest.raises(TypeError):
        _ = metrics[0]

def test_reflection_result_immutability_and_access():
    res = ReflectionResult(
        should_learn=True,
        type="rule",
        content="Speak clearly",
        category="formatting",
        trigger_context="speech",
        id=5
    )
    assert res["content"] == "Speak clearly"
    assert "category" in res
    assert res.get("id") == 5
    assert res.to_dict()["should_learn"] is True

    with pytest.raises(FrozenInstanceError):
        res.content = "Other"

    with pytest.raises(KeyError):
        _ = res["invalid"]

    with pytest.raises(TypeError):
        _ = res[0]

def test_backup_result_immutability_and_access():
    from core.models import BackupResult
    b = BackupResult(
        success=True,
        db_path="/tmp/memory.db",
        size_bytes=1024,
        created_at="2026-10-13T00:00:00Z"
    )
    assert b["success"] is True
    assert "size_bytes" in b
    assert b.get("error") is None
    assert b.to_dict()["size_bytes"] == 1024

    with pytest.raises(FrozenInstanceError):
        b.success = False

    with pytest.raises(KeyError):
        _ = b["missing"]

    with pytest.raises(TypeError):
        _ = b[0]

def test_scheduled_report_immutability_and_access():
    from core.models import ScheduledReport
    r = ScheduledReport(
        report_type="daily_health",
        text="Report text",
        timestamp="2026-10-13T00:00:00Z"
    )
    assert r["report_type"] == "daily_health"
    assert "text" in r
    assert r.get("report_type") == "daily_health"
    assert r.to_dict()["report_type"] == "daily_health"

    with pytest.raises(FrozenInstanceError):
        r.report_type = "other"

    with pytest.raises(KeyError):
        _ = r["missing"]

    with pytest.raises(TypeError):
        _ = r[0]
