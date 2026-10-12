import pytest
from unittest.mock import MagicMock
from core.vps_manager import (
    get_vps_health,
    list_docker_containers,
    get_mltb_logs,
    restart_mltb_container,
    validate_container_name,
)
from core.exceptions import ValidationError

def test_validate_container_name():
    assert validate_container_name("mirrorbot-app-1") == "mirrorbot-app-1"
    assert validate_container_name("app_test.1") == "app_test.1"

    with pytest.raises(ValidationError):
        validate_container_name("")

    with pytest.raises(ValidationError):
        validate_container_name("app; rm -rf /")

    with pytest.raises(ValidationError):
        validate_container_name("name with spaces")

    with pytest.raises(ValidationError):
        validate_container_name("a" * 65)

def test_get_vps_health():
    stats = get_vps_health()
    assert "cpu_percent" in stats
    assert "ram_used_gb" in stats
    assert "disk_free_gb" in stats
    assert "uptime" in stats
    assert stats["ram_total_gb"] > 0
    assert "VPS Health" in stats["text"]

def test_list_docker_containers_with_mock():
    # Mock container objects
    mock_c1 = MagicMock()
    mock_c1.name = "mirrorbot-app-1"
    mock_c1.status = "running"
    mock_c1.image.tags = ["mirrorbot:latest"]

    mock_c2 = MagicMock()
    mock_c2.name = "portainer"
    mock_c2.status = "exited"
    mock_c2.image.tags = []
    mock_c2.image.short_id = "portainer123"

    mock_client = MagicMock()
    mock_client.containers.list.return_value = [mock_c1, mock_c2]

    result = list_docker_containers(client=mock_client)
    assert "mirrorbot-app-1" in result
    assert "portainer" in result
    assert "🟢" in result
    assert "🔴" in result

def test_list_docker_containers_empty():
    mock_client = MagicMock()
    mock_client.containers.list.return_value = []
    result = list_docker_containers(client=mock_client)
    assert "No Docker containers found" in result

def test_list_docker_containers_error():
    mock_client = MagicMock()
    mock_client.containers.list.side_effect = Exception("Docker daemon down")
    result = list_docker_containers(client=mock_client)
    assert "Error querying Docker" in result

def test_get_mltb_logs_success():
    mock_container = MagicMock()
    mock_container.name = "mirrorbot-app-1"
    mock_container.logs.return_value = b"[INFO] Download completed: Ubuntu.iso\n[INFO] Uploaded to GDrive"

    mock_client = MagicMock()
    mock_client.containers.list.return_value = [mock_container]

    result = get_mltb_logs("mirrorbot-app-1", lines=10, client=mock_client)
    assert "Ubuntu.iso" in result
    assert "Recent Logs for `mirrorbot-app-1`" in result

def test_get_mltb_logs_empty():
    mock_container = MagicMock()
    mock_container.name = "mirrorbot-app-1"
    mock_container.logs.return_value = b""

    mock_client = MagicMock()
    mock_client.containers.list.return_value = [mock_container]

    result = get_mltb_logs("mirrorbot-app-1", lines=10, client=mock_client)
    assert "has no recent logs" in result

def test_get_mltb_logs_not_found():
    mock_client = MagicMock()
    mock_client.containers.list.return_value = []

    result = get_mltb_logs("missing-container", client=mock_client)
    assert "not found" in result

def test_get_mltb_logs_invalid_name():
    result = get_mltb_logs("invalid name!")
    assert "Invalid container name" in result

def test_restart_mltb_container_success():
    mock_container = MagicMock()
    mock_container.name = "mirrorbot-app-1"
    mock_container.restart.return_value = None

    mock_client = MagicMock()
    mock_client.containers.list.return_value = [mock_container]

    result = restart_mltb_container("mirrorbot-app-1", client=mock_client)
    assert "Successfully restarted `mirrorbot-app-1`" in result
    mock_container.restart.assert_called_once_with(timeout=10)

def test_restart_mltb_container_not_found():
    mock_client = MagicMock()
    mock_client.containers.list.return_value = []

    result = restart_mltb_container("missing-container", client=mock_client)
    assert "not found" in result

def test_restart_mltb_container_error():
    mock_container = MagicMock()
    mock_container.name = "mirrorbot-app-1"
    mock_container.restart.side_effect = Exception("Restart timeout")

    mock_client = MagicMock()
    mock_client.containers.list.return_value = [mock_container]

    result = restart_mltb_container("mirrorbot-app-1", client=mock_client)
    assert "Failed to restart" in result
