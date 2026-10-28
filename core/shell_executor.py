"""
Safe Shell Command Executor for Remote VPS Administration.
Enforces security guardrails, timeout bounds, and safe output handling per ECC standards.
"""

from __future__ import annotations
import asyncio
import re
import logging
from dataclasses import dataclass
from typing import Optional

from core.exceptions import SecurityError, ValidationError

logger = logging.getLogger(__name__)

# Patterns that pose critical risk to system integrity or data loss
DISALLOWED_PATTERNS = [
    re.compile(r"\brm\b.*(-[a-zA-Z]*r|--recursive)", re.IGNORECASE),  # Any recursive deletion
    re.compile(r"\b(mkfs(\.[a-z0-9]+)?|dd|fdisk|parted|sfdisk)\b", re.IGNORECASE),
    re.compile(r"\b(shutdown|reboot|poweroff|halt|init\s+[06])\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),  # forkbomb
    re.compile(r"\bchmod\b.*(-[a-zA-Z]*R\s+)?(000|777)\s+/", re.IGNORECASE),
]

@dataclass(frozen=True)
class ShellExecutionResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    is_truncated: bool
    full_output: str

    @property
    def formatted_text(self) -> str:
        icon = "🟢" if self.exit_code == 0 else f"🔴 (Exit {self.exit_code})"
        parts = [
            f"💻 **Lệnh:** `{self.command}`",
            f"⏱️ **Thời gian:** `{self.duration_s:.2f}s` | **Kết quả:** {icon}",
        ]

        body = self.stdout if self.stdout.strip() else self.stderr
        if not body.strip():
            body = "(Không có output)"

        # Truncate if longer than 3000 chars for Telegram Markdown
        if len(body) > 3000:
            preview = body[:3000] + "\n... [Đã cắt ngắn vì output quá dài]"
            parts.append(f"\n```text\n{preview}\n```")
        else:
            parts.append(f"\n```text\n{body}\n```")

        return "\n".join(parts)


def validate_shell_command(command: str) -> str:
    """Validates command string and checks against destructive pattern blacklist."""
    if not command or not command.strip():
        raise ValidationError("Lệnh không được để trống.")

    cleaned = command.strip()
    for pattern in DISALLOWED_PATTERNS:
        if pattern.search(cleaned):
            raise SecurityError(
                f"Lệnh bị chặn do phát hiện nguy cơ phá hoại hệ thống (Pattern: {pattern.pattern})."
            )
    return cleaned


async def execute_safe_shell_command(
    command: str,
    timeout: float = 30.0,
    working_dir: Optional[str] = None,
) -> ShellExecutionResult:
    """
    Executes a shell command asynchronously with strict timeout and security controls.
    """
    validated_cmd = validate_shell_command(command)
    start_time = asyncio.get_event_loop().time()

    try:
        proc = await asyncio.create_subprocess_shell(
            validated_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            duration = asyncio.get_event_loop().time() - start_time
            return ShellExecutionResult(
                command=validated_cmd,
                exit_code=124,
                stdout="",
                stderr=f"⏱️ Lệnh vượt quá giới hạn thời gian cho phép ({timeout}s). Đã hủy tiến trình.",
                duration_s=duration,
                is_truncated=False,
                full_output=f"Timeout after {timeout}s",
            )

        duration = asyncio.get_event_loop().time() - start_time
        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        combined = f"{stdout}\n{stderr}".strip() if stderr else stdout
        is_truncated = len(combined) > 3000

        return ShellExecutionResult(
            command=validated_cmd,
            exit_code=proc.returncode if proc.returncode is not None else 1,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            is_truncated=is_truncated,
            full_output=combined,
        )

    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        return ShellExecutionResult(
            command=validated_cmd,
            exit_code=-1,
            stdout="",
            stderr=f"❌ Lỗi thực thi tiến trình: {str(e)}",
            duration_s=duration,
            is_truncated=False,
            full_output=str(e),
        )
