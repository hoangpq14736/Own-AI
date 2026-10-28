import pytest
from unittest.mock import MagicMock, patch
from core.agent_tools import (
    get_vps_status,
    get_docker_containers,
    restart_docker_container,
    create_scheduled_reminder,
    get_all_agent_tools,
    set_agent_context,
)
from core.models import VPSMetrics

def test_get_all_agent_tools():
    tools = get_all_agent_tools()
    assert len(tools) == 4
    for t in tools:
        assert callable(t)

def test_get_vps_status():
    mock_metrics = VPSMetrics(
        cpu_percent=10.0,
        cpu_count=4,
        ram_used_gb=2.0,
        ram_total_gb=24.0,
        ram_percent=8.3,
        disk_used_gb=10.0,
        disk_free_gb=190.0,
        disk_total_gb=200.0,
        disk_percent=5.0,
        uptime="1d 2h",
        text="Mock VPS Stats"
    )
    with patch("core.agent_tools.get_vps_health", return_value=mock_metrics):
        res = get_vps_status()
        assert res == "Mock VPS Stats"

def test_get_docker_containers():
    with patch("core.agent_tools.list_docker_containers", return_value="Mock Docker List"):
        res = get_docker_containers()
        assert res == "Mock Docker List"

def test_restart_docker_container():
    with patch("core.agent_tools.restart_mltb_container", return_value="Restarted mirrorbot"):
        res = restart_docker_container("mirrorbot-app-1")
        assert res == "Restarted mirrorbot"

def test_create_scheduled_reminder_without_context():
    set_agent_context(None, None)
    res = create_scheduled_reminder("Họp", "30m")
    assert "chưa được khởi tạo" in res

def test_create_scheduled_reminder_with_context():
    mock_memory = MagicMock()
    mock_memory.add_reminder.return_value = 42
    set_agent_context(mock_memory, "12345")

    res = create_scheduled_reminder("Họp dự án", "30m")
    assert "thành công (#42)" in res
    assert "Họp dự án" in res
    mock_memory.add_reminder.assert_called_once()
