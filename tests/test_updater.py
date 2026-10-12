"""
Tests for Core Updater Module (TDD - Red Phase).
Verifies git inspection, update detection, atomic pull, dependency installation,
and safe startup reload behavior.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.models import UpdateInfo
from core.updater import (
    get_current_commit_info,
    check_for_updates,
    apply_update,
    check_and_apply_update_on_startup,
    restart_bot_process,
)
from config import Settings


class TestUpdateInfoModel:
    def test_update_info_properties_and_immutability(self):
        info = UpdateInfo(
            current_commit="abc1234",
            current_commit_message="Initial commit",
            latest_commit="def5678",
            latest_commit_message="New feature",
            has_update=True,
            behind_count=1,
            commits_behind=("def5678 New feature",),
            error=None,
        )

        assert info.current_commit == "abc1234"
        assert info.has_update is True
        assert info.behind_count == 1
        assert "def5678 New feature" in info.commits_behind
        assert info["current_commit"] == "abc1234"
        assert "latest_commit" in info
        assert info.get("behind_count") == 1

        d = info.to_dict()
        assert d["has_update"] is True
        assert d["behind_count"] == 1

        with pytest.raises((AttributeError, TypeError)):
            info.has_update = False  # Immutable


class TestGitInspection:
    @patch("subprocess.run")
    def test_get_current_commit_info_success(self, mock_run):
        # 1st call for rev-parse, 2nd call for commit message
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc1234\n"),
            MagicMock(returncode=0, stdout="feat: awesome feature\n"),
        ]

        commit_hash, commit_msg = get_current_commit_info(Path("/test/dir"))
        assert commit_hash == "abc1234"
        assert commit_msg == "feat: awesome feature"

    @patch("subprocess.run")
    def test_get_current_commit_info_git_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["git"])
        commit_hash, commit_msg = get_current_commit_info(Path("/test/dir"))
        assert commit_hash == "unknown"
        assert commit_msg == "unknown"

    @patch("subprocess.run")
    def test_get_current_commit_info_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("git not found")
        commit_hash, commit_msg = get_current_commit_info(Path("/test/dir"))
        assert commit_hash == "unknown"
        assert commit_msg == "unknown"


class TestCheckForUpdates:
    @patch("subprocess.run")
    def test_check_for_updates_up_to_date(self, mock_run):
        # 1. safe.directory
        # 2. git fetch
        # 3. current commit hash
        # 4. current commit msg
        # 5. latest remote commit hash
        # 6. latest remote commit msg
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout="abc1234\n"),
            MagicMock(returncode=0, stdout="fix: minor bug\n"),
            MagicMock(returncode=0, stdout="abc1234\n"),
            MagicMock(returncode=0, stdout="fix: minor bug\n"),
        ]

        info = check_for_updates(repo_dir=Path("/test/repo"), branch="main")
        assert info.has_update is False
        assert info.behind_count == 0
        assert info.current_commit == "abc1234"
        assert info.latest_commit == "abc1234"
        assert info.error is None

    @patch("subprocess.run")
    def test_check_for_updates_updates_available(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),  # git safe.directory
            MagicMock(returncode=0, stdout=""),  # git fetch
            MagicMock(returncode=0, stdout="abc1234\n"),  # current hash
            MagicMock(returncode=0, stdout="fix: old bug\n"),  # current msg
            MagicMock(returncode=0, stdout="def5678\n"),  # remote hash
            MagicMock(returncode=0, stdout="feat: new feature\n"),  # remote msg
            MagicMock(returncode=0, stdout="def5678 feat: new feature\n456789a docs: update readme\n"),  # git log
        ]

        info = check_for_updates(repo_dir=Path("/test/repo"), branch="main")
        assert info.has_update is True
        assert info.behind_count == 2
        assert info.current_commit == "abc1234"
        assert info.latest_commit == "def5678"
        assert len(info.commits_behind) == 2
        assert "def5678 feat: new feature" in info.commits_behind

    @patch("subprocess.run")
    def test_check_for_updates_network_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git", "fetch"], timeout=20)

        info = check_for_updates(repo_dir=Path("/test/repo"))
        assert info.has_update is False
        assert info.error is not None
        assert "timed out" in info.error.lower()


class TestApplyUpdate:
    @patch("subprocess.run")
    def test_apply_update_success_without_pip(self, mock_run):
        # 1. safe.directory
        # 2. git pull
        # 3. git diff --name-only HEAD@{1} HEAD
        # 4. git rev-parse --short HEAD
        # 5. git log -1 --pretty=%s
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout="Fast-forward\n bot.py | 5 ++\n"),
            MagicMock(returncode=0, stdout="bot.py\ncore/updater.py\n"),
            MagicMock(returncode=0, stdout="def5678\n"),
            MagicMock(returncode=0, stdout="feat: updated bot\n"),
        ]

        success, msg = apply_update(repo_dir=Path("/test/repo"))
        assert success is True
        assert "def5678" in msg
        # Ensure pip was not invoked
        assert not any("pip" in str(call.args[0]) for call in mock_run.call_args_list)

    @patch("subprocess.run")
    def test_apply_update_success_with_pip_install(self, mock_run):
        # 1. safe.directory
        # 2. git pull
        # 3. git diff (contains requirements.txt)
        # 4. pip install -r requirements.txt
        # 5. rev-parse
        # 6. git log
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout="Fast-forward\n requirements.txt | 1 +\n"),
            MagicMock(returncode=0, stdout="requirements.txt\nbot.py\n"),
            MagicMock(returncode=0, stdout="Successfully installed packages\n"),
            MagicMock(returncode=0, stdout="9876543\n"),
            MagicMock(returncode=0, stdout="feat: added dependencies\n"),
        ]

        success, msg = apply_update(repo_dir=Path("/test/repo"))
        assert success is True
        assert "9876543" in msg
        # Verify pip install was called
        pip_called = any(isinstance(call.args[0], list) and "pip" in call.args[0] for call in mock_run.call_args_list)
        assert pip_called is True

    @patch("subprocess.run")
    def test_apply_update_pull_failure(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            subprocess.CalledProcessError(1, ["git", "pull"], stderr="error: merge conflict"),
        ]

        success, msg = apply_update(repo_dir=Path("/test/repo"))
        assert success is False
        assert "merge conflict" in msg or "failed" in msg.lower()


class TestStartupAutoUpdate:
    def test_startup_disabled_in_settings(self):
        settings = Settings(auto_update_on_restart=False)
        with patch("core.updater.check_for_updates") as mock_check:
            result = check_and_apply_update_on_startup(settings)
            assert result is False
            mock_check.assert_not_called()

    def test_startup_already_updated_env_flag(self, monkeypatch):
        monkeypatch.setenv("_ALREADY_UPDATED", "1")
        settings = Settings(auto_update_on_restart=True)
        with patch("core.updater.check_for_updates") as mock_check:
            result = check_and_apply_update_on_startup(settings)
            assert result is False
            mock_check.assert_not_called()

    @patch("core.updater.restart_bot_process")
    @patch("core.updater.apply_update")
    @patch("core.updater.check_for_updates")
    def test_startup_updates_and_restarts(self, mock_check, mock_apply, mock_restart, monkeypatch):
        monkeypatch.delenv("_ALREADY_UPDATED", raising=False)
        settings = Settings(auto_update_on_restart=True)

        mock_check.return_value = UpdateInfo(
            current_commit="abc1234",
            current_commit_message="Old",
            latest_commit="def5678",
            latest_commit_message="New",
            has_update=True,
            behind_count=1,
            commits_behind=("def5678 New",),
        )
        mock_apply.return_value = (True, "Updated to def5678")

        result = check_and_apply_update_on_startup(settings)
        assert result is True
        mock_apply.assert_called_once()
        mock_restart.assert_called_once()

    @patch("core.updater.restart_bot_process")
    @patch("core.updater.apply_update")
    @patch("core.updater.check_for_updates")
    def test_startup_no_updates_available(self, mock_check, mock_apply, mock_restart, monkeypatch):
        monkeypatch.delenv("_ALREADY_UPDATED", raising=False)
        settings = Settings(auto_update_on_restart=True)

        mock_check.return_value = UpdateInfo(
            current_commit="abc1234",
            current_commit_message="Current",
            latest_commit="abc1234",
            latest_commit_message="Current",
            has_update=False,
            behind_count=0,
        )

        result = check_and_apply_update_on_startup(settings)
        assert result is False
        mock_apply.assert_not_called()
        mock_restart.assert_not_called()

    @patch("core.updater.apply_update")
    @patch("core.updater.check_for_updates")
    def test_startup_update_fails(self, mock_check, mock_apply, monkeypatch):
        monkeypatch.delenv("_ALREADY_UPDATED", raising=False)
        settings = Settings(auto_update_on_restart=True)
        mock_check.return_value = UpdateInfo(
            current_commit="abc1234",
            current_commit_message="Old",
            latest_commit="def5678",
            latest_commit_message="New",
            has_update=True,
            behind_count=1,
        )
        mock_apply.return_value = (False, "Conflict error")

        result = check_and_apply_update_on_startup(settings)
        assert result is False

    @patch("core.updater.check_for_updates")
    def test_startup_info_has_error(self, mock_check, monkeypatch):
        monkeypatch.delenv("_ALREADY_UPDATED", raising=False)
        settings = Settings(auto_update_on_restart=True)
        mock_check.return_value = UpdateInfo(
            current_commit="abc1234",
            current_commit_message="Cur",
            latest_commit="unknown",
            latest_commit_message="unknown",
            has_update=False,
            error="Network offline",
        )
        result = check_and_apply_update_on_startup(settings)
        assert result is False


class TestUpdaterEdgeCases:
    @patch("shutil.which")
    def test_check_for_updates_git_not_found(self, mock_which):
        mock_which.return_value = None
        info = check_for_updates()
        assert info.has_update is False
        assert "not found" in info.error.lower()

    @patch("subprocess.run")
    def test_check_for_updates_called_process_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["git", "fetch"], stderr="fatal: remote not found")
        info = check_for_updates()
        assert info.has_update is False
        assert "remote not found" in info.error

    @patch("subprocess.run")
    def test_check_for_updates_generic_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("Disk full")
        info = check_for_updates()
        assert info.has_update is False
        assert "Disk full" in info.error

    @patch("subprocess.run")
    def test_apply_update_generic_exception(self, mock_run):
        mock_run.side_effect = RuntimeError("Permission denied")
        success, msg = apply_update()
        assert success is False
        assert "Permission denied" in msg

    @patch("os.execv")
    def test_restart_bot_process(self, mock_execv):
        restart_bot_process()
        mock_execv.assert_called_once()
