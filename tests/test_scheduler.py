import pytest
import sqlite3
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from datetime import datetime, timezone

from core.scheduler import (
    create_memory_backup,
    create_daily_health_report,
    SchedulerService,
)
from core.models import VPSMetrics, BackupResult
from config import Settings

@pytest.fixture
def sample_db(tmp_path):
    db_file = tmp_path / "memory.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE test (id INT, val TEXT)")
    conn.execute("INSERT INTO test VALUES (1, 'hello')")
    conn.commit()
    conn.close()
    return db_file

def test_create_memory_backup_success(sample_db, tmp_path):
    backup_dir = tmp_path / "backups"
    result = create_memory_backup(sample_db, backup_dir=backup_dir)

    assert result.success is True
    assert result["success"] is True
    assert Path(result.db_path).exists()
    assert result.size_bytes > 0

    # Verify backup contents are intact
    conn = sqlite3.connect(result.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT val FROM test WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    assert row[0] == "hello"

def test_create_memory_backup_missing_file(tmp_path):
    non_existent = tmp_path / "missing.db"
    result = create_memory_backup(non_existent)
    assert result.success is False
    assert "does not exist" in result.error

def test_create_memory_backup_pruning(sample_db, tmp_path):
    backup_dir = tmp_path / "backups_prune"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create 8 dummy old backups
    for i in range(8):
        f = backup_dir / f"memory_backup_2026010{i}_000000.db"
        f.write_text("dummy")

    # Creating a new backup should prune down to 7
    result = create_memory_backup(sample_db, backup_dir=backup_dir)
    assert result.success is True
    existing = list(backup_dir.glob("memory_backup_*.db"))
    assert len(existing) <= 7

def test_create_daily_health_report():
    mock_metrics = VPSMetrics(
        cpu_percent=15.0,
        cpu_count=4,
        ram_used_gb=4.5,
        ram_total_gb=24.0,
        ram_percent=18.7,
        disk_used_gb=25.0,
        disk_free_gb=175.0,
        disk_total_gb=200.0,
        disk_percent=12.5,
        uptime="5d 2h 10m",
        warning=None,
        text="Normal stats"
    )

    def mock_vps():
        return mock_metrics

    def mock_docker():
        return "🐳 **Active Docker Containers:**\n\n🟢 **mirrorbot-app-1**\n🟢 **portainer**"

    mock_memory = MagicMock()
    mock_memory.list_rules.return_value = ["rule1", "rule2"]
    mock_memory.list_facts.return_value = ["fact1"]

    report = create_daily_health_report(
        vps_func=mock_vps,
        docker_func=mock_docker,
        memory_engine=mock_memory,
        user_id="12345"
    )

    assert report.report_type == "daily_health"
    assert "Good Morning" in report.text
    assert "mirrorbot-app-1" in report.text
    assert "Active Learned Rules: `2`" in report.text
    assert "Remembered Facts: `1`" in report.text

@pytest.mark.anyio
async def test_scheduler_service_send_report(sample_db):
    mock_bot = AsyncMock()
    settings = Settings(
        db_path=sample_db,
        allowed_user_ids=(12345,),
        enable_daily_report=True
    )
    mock_memory = MagicMock()
    mock_memory.list_rules.return_value = []
    mock_memory.list_facts.return_value = []

    service = SchedulerService(
        bot=mock_bot,
        settings=settings,
        memory_engine=mock_memory
    )

    success = await service.send_report(chat_id=12345)
    assert success is True
    mock_bot.send_message.assert_called_once()
    assert "Good Morning" in mock_bot.send_message.call_args[1]["text"]

@pytest.mark.anyio
async def test_scheduler_service_send_backup(sample_db):
    mock_bot = AsyncMock()
    settings = Settings(
        db_path=sample_db,
        allowed_user_ids=(12345,),
        enable_auto_backup=True
    )
    mock_memory = MagicMock()

    service = SchedulerService(
        bot=mock_bot,
        settings=settings,
        memory_engine=mock_memory
    )

    success = await service.send_backup(chat_id=12345)
    assert success is True
    mock_bot.send_document.assert_called_once()
    assert "Memory Database Backup" in mock_bot.send_document.call_args[1]["caption"]

@pytest.mark.anyio
async def test_scheduler_service_watchdog_disk_alert(sample_db):
    mock_bot = AsyncMock()
    settings = Settings(db_path=sample_db)
    mock_memory = MagicMock()

    service = SchedulerService(
        bot=mock_bot,
        settings=settings,
        memory_engine=mock_memory
    )

    # Mock high disk (>85%)
    mock_high_disk = VPSMetrics(
        cpu_percent=10.0,
        cpu_count=4,
        ram_used_gb=10.0,
        ram_total_gb=24.0,
        ram_percent=40.0,
        disk_used_gb=190.0,
        disk_free_gb=10.0,
        disk_total_gb=200.0,
        disk_percent=95.0,
        uptime="10d",
        warning="Disk high",
        text="text"
    )

    with patch("core.scheduler.get_vps_health", return_value=mock_high_disk):
        await service._check_watchdog(chat_id=12345)
        mock_bot.send_message.assert_called_once()
        assert "CRITICAL VPS ALERT" in mock_bot.send_message.call_args[1]["text"]

@pytest.mark.anyio
async def test_scheduler_service_start_stop(sample_db):
    mock_bot = AsyncMock()
    settings = Settings(
        db_path=sample_db,
        allowed_user_ids=(12345,),
        enable_daily_report=False
    )
    mock_memory = MagicMock()

    service = SchedulerService(
        bot=mock_bot,
        settings=settings,
        memory_engine=mock_memory,
        sleep_interval=0.05
    )

    task = service.start()
    assert service._running is True
    assert task is not None

    # Let loop run for a tiny fraction of a second
    await asyncio.sleep(0.1)

    service.stop()
    assert service._running is False
    await asyncio.sleep(0.05)
    assert task.cancelled() or task.done()

@pytest.mark.anyio
async def test_scheduler_loop_fires_report(sample_db):
    mock_bot = AsyncMock()
    now_hour = datetime.now().hour
    settings = Settings(
        db_path=sample_db,
        allowed_user_ids=(12345,),
        enable_daily_report=True,
        daily_report_hour=now_hour,
        enable_auto_backup=True
    )
    mock_memory = MagicMock()
    mock_memory.list_rules.return_value = []
    mock_memory.list_facts.return_value = []

    service = SchedulerService(
        bot=mock_bot,
        settings=settings,
        memory_engine=mock_memory,
        sleep_interval=0.02
    )

    task = service.start()
    await asyncio.sleep(0.08)
    service.stop()
    await asyncio.sleep(0.02)

    # Should have triggered send_report and send_backup
    assert mock_bot.send_message.called
    assert mock_bot.send_document.called

@pytest.mark.anyio
async def test_scheduler_docker_crash_event(sample_db):
    mock_bot = AsyncMock()
    settings = Settings(
        db_path=sample_db,
        allowed_user_ids=(12345,),
    )
    mock_memory = MagicMock()

    service = SchedulerService(
        bot=mock_bot,
        settings=settings,
        memory_engine=mock_memory,
        sleep_interval=0.1
    )
    service._running = True

    mock_event = {
        "Action": "die",
        "Actor": {
            "Attributes": {
                "name": "mirrorbot-app-1",
                "exitCode": "137"
            }
        }
    }

    def fake_events_gen():
        yield mock_event

    with patch("core.scheduler.get_docker_events_generator", return_value=fake_events_gen()):
        task = asyncio.create_task(service._listen_docker_events(12345))
        await asyncio.sleep(0.1)
        service._running = False
        await asyncio.sleep(0.05)
        task.cancel()

    assert mock_bot.send_message.called
    call_args = mock_bot.send_message.call_args[1]
    assert "DOCKER CONTAINER CRASH ALERT" in call_args["text"]
    assert "mirrorbot-app-1" in call_args["text"]
    assert "137" in call_args["text"]
