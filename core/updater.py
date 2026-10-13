"""
Repository Updater Module.
Enables self-updating capabilities, git commit inspection, dependency upgrades,
and zero-downtime process reloads for VPS and Docker environments.
"""

import re
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


def _git_env() -> dict:
    """Returns environment variables with interactive prompts disabled."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _build_git_cmd(args: list, use_token: bool = True) -> list:
    """Builds git command with optional GitHub token authorization header."""
    cmd = ["git", "-c", "credential.helper="]
    token = getattr(config.settings, "github_token", "").strip() if use_token else ""
    if token:
        cmd.extend(["-c", f"http.extraHeader=Authorization: token {token}"])
    cmd.extend(args)
    return cmd


def _clean_git_remote_url(repo_dir: Path, remote: str = "origin") -> bool:
    """
    Sanitizes remote URL to remove any embedded credentials/tokens.
    E.g. transforms 'https://ghp_xxx@github.com/user/repo.git' to 'https://github.com/user/repo.git'.
    Returns True if the URL was changed, False otherwise.
    """
    try:
        res = subprocess.run(
            ["git", "remote", "get-url", remote],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            env=_git_env(),
        )
        current_url = res.stdout.strip()
        if "@github.com" in current_url:
            clean_url = re.sub(r"https?://[^@]+@github\.com", "https://github.com", current_url)
            if clean_url != current_url:
                logger.info(f"Sanitizing git remote '{remote}' URL from {current_url} to {clean_url}")
                subprocess.run(
                    ["git", "remote", "set-url", remote, clean_url],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                    env=_git_env(),
                )
                return True
    except Exception as e:
        logger.debug(f"Unable to inspect or clean git remote url: {e}")
    return False


def _run_git_cmd_with_fallback(
    args: list,
    cwd: Path,
    timeout: int = 20,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Executes a git command. If a GITHUB_TOKEN is present and authentication fails
    (which happens if the token is invalid or if the repo was made public with an old token),
    automatically sanitizes the remote and retries anonymously for public repos.
    """
    token = getattr(config.settings, "github_token", "").strip()
    if token:
        try:
            return subprocess.run(
                _build_git_cmd(args, use_token=True),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                env=_git_env(),
            )
        except subprocess.CalledProcessError as e:
            raw_err = ((e.stderr or "") + " " + (e.stdout or "")).lower()
            auth_failures = [
                "could not read username",
                "authentication failed",
                "invalid credentials",
                "401",
                "terminal prompts disabled",
            ]
            if any(fail in raw_err for fail in auth_failures):
                logger.info(
                    "Git operation with GITHUB_TOKEN failed (token invalid or repo is public). "
                    "Sanitizing remote and attempting anonymous fallback..."
                )
                remote = "origin"
                if len(args) >= 2 and args[0] in ("fetch", "pull"):
                    remote = args[1]
                _clean_git_remote_url(cwd, remote)
                return subprocess.run(
                    _build_git_cmd(args, use_token=False),
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=check,
                    env=_git_env(),
                )
            raise
    else:
        try:
            return subprocess.run(
                _build_git_cmd(args, use_token=False),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                env=_git_env(),
            )
        except subprocess.CalledProcessError as e:
            raw_err = ((e.stderr or "") + " " + (e.stdout or "")).lower()
            auth_failures = [
                "could not read username",
                "authentication failed",
                "invalid credentials",
                "401",
                "terminal prompts disabled",
            ]
            if any(fail in raw_err for fail in auth_failures):
                remote = "origin"
                if len(args) >= 2 and args[0] in ("fetch", "pull"):
                    remote = args[1]
                if _clean_git_remote_url(cwd, remote):
                    logger.info("Sanitized embedded token from remote URL. Retrying git operation...")
                    return subprocess.run(
                        _build_git_cmd(args, use_token=False),
                        cwd=str(cwd),
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=check,
                        env=_git_env(),
                    )
            raise


def _ensure_git_safe_directory(repo_dir: Path) -> None:
    """Configures Git safe.directory to prevent dubious ownership errors in Docker."""
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=_git_env(),
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
        _run_git_cmd_with_fallback(["fetch", remote, branch], cwd=base_dir, timeout=timeout)

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
        raw_err = e.stderr.strip() if e.stderr else str(e)
        if any(term in raw_err.lower() for term in ["could not read username", "authentication failed", "invalid credentials"]):
            err_msg = "Repository is Private. Make it Public on GitHub or add GITHUB_TOKEN to your .env file."
        else:
            err_msg = f"Git error: {raw_err}"
        logger.warning(f"{err_msg} (detail: {raw_err})")
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
        pull_res = _run_git_cmd_with_fallback(["pull", remote, branch], cwd=base_dir, timeout=timeout)

        # 2. Check if requirements.txt changed
        diff_res = subprocess.run(
            ["git", "diff", "--name-only", "HEAD@{1}", "HEAD"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_git_env(),
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
        if any(term in err_msg.lower() for term in ["could not read username", "authentication failed", "invalid credentials"]):
            return False, "Repository is Private. Make it Public on GitHub (Settings -> Danger Zone) or add GITHUB_TOKEN to .env."
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
