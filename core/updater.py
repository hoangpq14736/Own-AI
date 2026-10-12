"""
Repository Updater Module.
Enables self-updating capabilities, git commit inspection, dependency upgrades,
and zero-downtime process reloads for VPS and Docker environments.
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import config
from core.models import UpdateInfo
from config import Settings

logger = logging.getLogger(__name__)


def _ensure_git_safe_directory(repo_dir: Path) -> None:
    """Configures Git safe.directory to prevent dubious ownership errors in Docker."""
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as e:
        logger.debug(f"Failed to set git safe.directory: {e}")


def get_current_commit_info(repo_dir: Optional[Path] = None) -> Tuple[str, str]:
    """
    Returns (short_commit_hash, commit_subject) for the current HEAD.
    Falls back to ('unknown', 'unknown') if Git is not present or an error occurs.
    """
    base_dir = repo_dir or config.BASE_DIR
    try:
        rev_res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        commit_hash = rev_res.stdout.strip()

        msg_res = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        commit_msg = msg_res.stdout.strip()
        return commit_hash, commit_msg
    except Exception as e:
        logger.debug(f"Unable to get git commit info: {e}")
        return "unknown", "unknown"


def check_for_updates(
    repo_dir: Optional[Path] = None,
    remote: str = "origin",
    branch: str = "main",
    timeout: int = 20,
) -> UpdateInfo:
    """
    Fetches the remote repository branch and compares local HEAD to remote HEAD.
    Returns an immutable UpdateInfo model.
    """
    base_dir = repo_dir or config.BASE_DIR

    if not shutil.which("git"):
        return UpdateInfo(
            current_commit="unknown",
            current_commit_message="unknown",
            latest_commit="unknown",
            latest_commit_message="unknown",
            has_update=False,
            error="Git executable not found in PATH",
        )

    try:
        _ensure_git_safe_directory(base_dir)

        # 1. Fetch remote branch
        subprocess.run(
            ["git", "fetch", remote, branch],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )

        # 2. Local HEAD info
        curr_hash, curr_msg = get_current_commit_info(base_dir)

        # 3. Remote branch HEAD info
        remote_ref = f"{remote}/{branch}"
        rem_rev = subprocess.run(
            ["git", "rev-parse", "--short", remote_ref],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        latest_hash = rem_rev.stdout.strip()

        rem_msg = subprocess.run(
            ["git", "log", "-1", "--pretty=%s", remote_ref],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        latest_msg = rem_msg.stdout.strip()

        # 4. Compare commits
        if curr_hash != latest_hash:
            log_res = subprocess.run(
                ["git", "log", f"HEAD..{remote_ref}", "--oneline"],
                cwd=str(base_dir),
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            commits = [c.strip() for c in log_res.stdout.splitlines() if c.strip()]
            return UpdateInfo(
                current_commit=curr_hash,
                current_commit_message=curr_msg,
                latest_commit=latest_hash,
                latest_commit_message=latest_msg,
                has_update=True,
                behind_count=len(commits),
                commits_behind=tuple(commits),
                error=None,
            )
        else:
            return UpdateInfo(
                current_commit=curr_hash,
                current_commit_message=curr_msg,
                latest_commit=latest_hash,
                latest_commit_message=latest_msg,
                has_update=False,
                behind_count=0,
                commits_behind=(),
                error=None,
            )

    except subprocess.TimeoutExpired as e:
        err_msg = f"Network timed out while fetching updates from {remote}/{branch}"
        logger.warning(err_msg)
        curr_hash, curr_msg = get_current_commit_info(base_dir)
        return UpdateInfo(
            current_commit=curr_hash,
            current_commit_message=curr_msg,
            latest_commit="unknown",
            latest_commit_message="unknown",
            has_update=False,
            error=err_msg,
        )
    except subprocess.CalledProcessError as e:
        err_msg = f"Git error: {e.stderr or str(e)}"
        logger.warning(err_msg)
        curr_hash, curr_msg = get_current_commit_info(base_dir)
        return UpdateInfo(
            current_commit=curr_hash,
            current_commit_message=curr_msg,
            latest_commit="unknown",
            latest_commit_message="unknown",
            has_update=False,
            error=err_msg,
        )
    except Exception as e:
        err_msg = f"Unexpected error during update check: {e}"
        logger.warning(err_msg)
        curr_hash, curr_msg = get_current_commit_info(base_dir)
        return UpdateInfo(
            current_commit=curr_hash,
            current_commit_message=curr_msg,
            latest_commit="unknown",
            latest_commit_message="unknown",
            has_update=False,
            error=err_msg,
        )


def apply_update(
    repo_dir: Optional[Path] = None,
    remote: str = "origin",
    branch: str = "main",
    timeout: int = 45,
) -> Tuple[bool, str]:
    """
    Pulls latest changes from remote branch into local repository.
    Automatically installs new dependencies if requirements.txt changed.
    Returns (success, message).
    """
    base_dir = repo_dir or config.BASE_DIR

    try:
        _ensure_git_safe_directory(base_dir)

        # 1. Pull changes
        pull_res = subprocess.run(
            ["git", "pull", remote, branch],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )

        # 2. Check if requirements.txt changed
        diff_res = subprocess.run(
            ["git", "diff", "--name-only", "HEAD@{1}", "HEAD"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        changed_files = diff_res.stdout.splitlines() if diff_res.stdout else []
        if any("requirements.txt" in f for f in changed_files):
            logger.info("requirements.txt changed in update. Installing updated packages...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=str(base_dir),
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )

        new_hash, new_msg = get_current_commit_info(base_dir)
        return True, f"Updated successfully to {new_hash} - {new_msg}"

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() if e.stderr else str(e)
        logger.error(f"Failed to apply update: {err_msg}")
        return False, f"Git pull failed: {err_msg}"
    except Exception as e:
        logger.error(f"Failed to apply update: {e}")
        return False, f"Update failed: {str(e)}"


def restart_bot_process() -> None:
    """
    Restarts the current Python process in-place using os.execv.
    Sets _ALREADY_UPDATED environment variable to prevent reload loops.
    """
    logger.info("Reloading bot process in-place with updated code...")
    os.environ["_ALREADY_UPDATED"] = "1"
    # Flush standard streams before execv
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable] + sys.argv)


def check_and_apply_update_on_startup(settings: Settings) -> bool:
    """
    Checks for remote repository changes upon bot restart or boot.
    If updates are available, pulls them and reloads the process cleanly.
    Returns True if an update was applied and restart triggered, False otherwise.
    """
    if not settings.auto_update_on_restart:
        logger.info("Startup auto-update is disabled in settings.")
        return False

    if os.environ.get("_ALREADY_UPDATED") == "1":
        logger.info("Bot was just reloaded from a recent update. Skipping startup update check.")
        return False

    logger.info(f"Checking for updates from {settings.git_repo_url} ({settings.git_branch})...")
    info = check_for_updates(
        branch=settings.git_branch,
    )

    if info.has_update:
        logger.info(
            f"🚀 New updates detected ({info.behind_count} commits behind). Latest: {info.latest_commit} - {info.latest_commit_message}"
        )
        success, msg = apply_update(branch=settings.git_branch)
        if success:
            logger.info(f"✅ {msg}. Restarting bot now to apply changes...")
            restart_bot_process()
            return True
        else:
            logger.warning(f"⚠️ Auto-update failed: {msg}. Continuing startup with existing code.")
            return False

    if info.error:
        logger.info(f"Update check warning: {info.error}. Continuing startup.")
    else:
        logger.info(f"Bot repository is up to date (running commit {info.current_commit}).")

    return False
