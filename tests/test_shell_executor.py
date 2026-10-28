import pytest
import sys
from core.shell_executor import (
    validate_shell_command,
    execute_safe_shell_command,
)
from core.exceptions import SecurityError, ValidationError

def test_validate_shell_command_valid():
    assert validate_shell_command("ls -la /app") == "ls -la /app"
    assert validate_shell_command("docker ps") == "docker ps"
    assert validate_shell_command("python --version") == "python --version"

def test_validate_shell_command_empty():
    with pytest.raises(ValidationError):
        validate_shell_command("")

    with pytest.raises(ValidationError):
        validate_shell_command("   ")

def test_validate_shell_command_disallowed():
    destructive_cmds = [
        "rm -rf /",
        "rm -rf /app",
        "rm -rf ~",
        "rm -rf *",
        "rm --recursive --force /var",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown -h now",
        "reboot",
        "init 0",
        ":(){ :|:& };:",
        "chmod -R 777 /",
    ]
    for cmd in destructive_cmds:
        with pytest.raises(SecurityError):
            validate_shell_command(cmd)

@pytest.mark.asyncio
async def test_execute_safe_shell_command_echo():
    cmd = "echo hello_antigravity"
    result = await execute_safe_shell_command(cmd)
    assert result.exit_code == 0
    assert "hello_antigravity" in result.stdout
    assert "🟢" in result.formatted_text

@pytest.mark.asyncio
async def test_execute_safe_shell_command_failure():
    # Running a command that exits with error
    cmd = f'"{sys.executable}" -c "import sys; sys.exit(42)"'
    result = await execute_safe_shell_command(cmd)
    assert result.exit_code == 42
    assert "Exit 42" in result.formatted_text

@pytest.mark.asyncio
async def test_execute_safe_shell_command_timeout():
    # Command sleeps for 2 seconds with 0.2s timeout
    cmd = f'"{sys.executable}" -c "import time; time.sleep(2)"'
    result = await execute_safe_shell_command(cmd, timeout=0.2)
    assert result.exit_code == 124
    assert "giới hạn thời gian" in result.stderr
