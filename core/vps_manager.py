"""
VPS Resource & Docker Container Supervisor.
Implements strict boundary validation, immutability, and dependency injection per ECC standards.
"""

import time
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import psutil

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from core.models import VPSMetrics
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Valid container name pattern: alphanumeric, underscore, dot, and hyphen (1-64 chars)
CONTAINER_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_\.\-]{1,64}$")

# ==============================================================================
# 1. VPS Resource & Health Monitoring
# ==============================================================================

def get_vps_health() -> VPSMetrics:
    """
    Collects live CPU, RAM, Disk, and Uptime metrics.
    Returns an immutable VPSMetrics instance.
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count(logical=True) or 1

    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)
    ram_percent = ram.percent

    # Support cross-platform disk usage lookup
    disk_path = "/" if hasattr(psutil, "disk_usage") and not psutil.WINDOWS else "C:\\"
    try:
        disk = psutil.disk_usage(disk_path)
    except Exception:
        disk = psutil.disk_usage(".")

    disk_used_gb = disk.used / (1024 ** 3)
    disk_free_gb = disk.free / (1024 ** 3)
    disk_total_gb = disk.total / (1024 ** 3)
    disk_percent = disk.percent

    # Calculate Uptime
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime_delta = datetime.now() - boot_time
    days = uptime_delta.days
    hours, remainder = divmod(uptime_delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    uptime_str = f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"

    warning = None
    if disk_percent > 85.0:
        warning = f"⚠️ *WARNING: Disk is {disk_percent}% full! Consider clearing temp downloads.*"

    formatted_text = (
        f"📊 **VPS Health & Resources:**\n\n"
        f"• ⚙️ **CPU:** `{cpu_percent}%` ({cpu_count} Cores)\n"
        f"• 🧠 **RAM:** `{ram_used_gb:.1f} GB` / `{ram_total_gb:.1f} GB` ({ram_percent}%)\n"
        f"• 💾 **Disk:** `{disk_free_gb:.1f} GB free` / `{disk_total_gb:.1f} GB` ({disk_percent}% used)\n"
        f"• ⏱️ **Uptime:** `{uptime_str}`"
    )

    if warning:
        formatted_text += f"\n\n{warning}"

    return VPSMetrics(
        cpu_percent=cpu_percent,
        cpu_count=cpu_count,
        ram_used_gb=round(ram_used_gb, 2),
        ram_total_gb=round(ram_total_gb, 2),
        ram_percent=ram_percent,
        disk_used_gb=round(disk_used_gb, 2),
        disk_free_gb=round(disk_free_gb, 2),
        disk_total_gb=round(disk_total_gb, 2),
        disk_percent=disk_percent,
        uptime=uptime_str,
        warning=warning,
        text=formatted_text,
    )

# ==============================================================================
# 2. Docker & MLTB Container Supervisor
# ==============================================================================

def validate_container_name(container_name: str) -> str:
    """Validates container name to prevent command/path injection."""
    if not container_name or not CONTAINER_NAME_REGEX.match(container_name):
        raise ValidationError(
            f"Invalid container name: '{container_name}'. Must be 1-64 alphanumeric characters, underscores, dots, or hyphens."
        )
    return container_name.strip()

def _get_docker():
    if not DOCKER_AVAILABLE:
        return None
    try:
        return docker.from_env()
    except Exception as e:
        logger.debug(f"Docker connection unavailable: {e}")
        return None

def list_docker_containers(client=None) -> str:
    """
    Lists all active Docker containers on the host.
    Allows injecting a client for unit testing.
    """
    docker_client = client or _get_docker()
    if not docker_client:
        return "⚠️ Docker is not connected or accessible from this environment."

    try:
        containers = docker_client.containers.list(all=True)
        if not containers:
            return "📦 No Docker containers found."

        text = "🐳 **Active Docker Containers:**\n\n"
        for c in containers:
            status_icon = "🟢" if c.status == "running" else "🔴"
            tag = c.image.tags[0] if getattr(c.image, "tags", None) else getattr(c.image, "short_id", "unknown")
            text += f"{status_icon} **{c.name}**\n   Status: `{c.status}` | Image: `{tag}`\n"
        return text
    except Exception as e:
        return f"❌ Error querying Docker: {str(e)}"

def get_mltb_logs(
    container_name: str = "mirrorbot-app-1",
    lines: int = 25,
    client=None
) -> str:
    """
    Fetches the last N lines of logs from MLTB or specified container.
    Enforces validation on container name and lines limit.
    """
    try:
        validated_name = validate_container_name(container_name)
    except ValidationError as e:
        return f"❌ {str(e)}"

    clamped_lines = max(1, min(lines, 200))

    docker_client = client or _get_docker()
    if not docker_client:
        return "⚠️ Docker socket is not accessible."

    try:
        target = None
        for c in docker_client.containers.list(all=True):
            if validated_name.lower() in c.name.lower():
                target = c
                break

        if not target:
            return f"❌ Container '{validated_name}' not found."

        raw_logs = target.logs(tail=clamped_lines)
        if isinstance(raw_logs, bytes):
            decoded_logs = raw_logs.decode("utf-8", errors="replace")
        else:
            decoded_logs = str(raw_logs)

        if not decoded_logs.strip():
            return f"📋 Container `{target.name}` has no recent logs."

        clean_logs = decoded_logs.strip()[-3500:]
        return f"📋 **Recent Logs for `{target.name}` (Last {clamped_lines} lines):**\n\n```text\n{clean_logs}\n```"

    except Exception as e:
        return f"❌ Error reading logs: {str(e)}"

def restart_mltb_container(
    container_name: str = "mirrorbot-app-1",
    client=None
) -> str:
    """
    Restarts the MLTB container safely.
    Enforces input validation and returns user-friendly messages.
    """
    try:
        validated_name = validate_container_name(container_name)
    except ValidationError as e:
        return f"❌ {str(e)}"

    docker_client = client or _get_docker()
    if not docker_client:
        return "⚠️ Docker socket is not accessible."

    try:
        target = None
        for c in docker_client.containers.list(all=True):
            if validated_name.lower() in c.name.lower():
                target = c
                break

        if not target:
            return f"❌ Container '{validated_name}' not found."

        logger.info(f"Restarting container: {target.name}")
        target.restart(timeout=10)
        return f"🔄 Successfully restarted `{target.name}`!"

    except Exception as e:
        return f"❌ Failed to restart `{validated_name}`: {str(e)}"
