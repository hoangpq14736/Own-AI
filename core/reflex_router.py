"""L1 Reflex Intent Router for fast, rule-based dispatching of user prompts."""

from __future__ import annotations

from dataclasses import dataclass
import re

__all__ = ["ReflexRoute", "route_prompt"]


@dataclass(frozen=True)
class ReflexRoute:
    """Routing decisions derived from reflex analysis of a prompt."""

    include_mltb: bool = False
    tool_domains: tuple[str, ...] = ()  # "vps_docker", "notes_reminders", "printer_3d"
    use_all_tools: bool = False
    matched_intents: tuple[str, ...] = ()


# Unicode-aware word boundary anchors
_WB_LEFT = r"(?<!\w)"
_WB_RIGHT = r"(?!\w)"

_MLTB_KEYWORDS = (
    r"https?://\S+",
    r"magnet:\?\S+",
    r"mirror",
    r"leech",
    r"clone",
    r"ytdl",
    r"torrent",
    r"magnet",
    r"rclone",
    r"gdrive",
    r"google drive",
    r"mega",
    r"pixeldrain",
    r"terabox",
    r"unzip",
    r"zip",
    r"giải nén",
    r"xả nén",
    r"nén zip",
    r"kéo file",
    r"tiến độ tải",
    r"tình trạng tải",
    r"hủy tải",
    r"dừng tải",
    r"cancelall",
    r"speedtest",
    r"tải\s+(?:link|video|nhạc|phim|file|torrent)",
)

_VPS_DOCKER_KEYWORDS = (
    r"vps",
    r"cpu",
    r"ram",
    r"disk",
    r"ổ đĩa",
    r"dung lượng",
    r"uptime",
    r"docker",
    r"container",
    r"restart",
    r"khởi động lại",
    r"mirrorbot",
    r"portainer",
    r"server",
    r"máy chủ",
    r"hệ thống",
)

_NOTES_REMINDERS_KEYWORDS = (
    r"nhắc",
    r"hẹn giờ",
    r"báo thức",
    r"nhắc nhở",
    r"reminder",
    r"remind",
    r"ghi chú",
    r"note",
    r"notes",
    r"bookmark",
    r"lưu lại",
    r"lưu giúp",
    r"ghi lại",
    r"tìm ghi chú",
    r"đã lưu",
)

_PRINTER_3D_KEYWORDS = (
    r"kobra",
    r"anycubic",
    r"máy in",
    r"in 3d",
    r"3d print",
    r"nozzle",
    r"bed",
    r"đầu phun",
    r"bàn nhiệt",
    r"gcode",
    r"stl",
    r"3mf",
    r"dfam",
    r"filament",
    r"nhựa in",
)

_GENERIC_ACTION_KEYWORDS = (
    r"kiểm tra",
    r"check",
    r"trạng thái",
    r"status",
    r"xem giúp",
    r"coi giúp",
    r"chạy lại",
)


def _compile_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    # Group expressions with word boundaries where appropriate
    expressions = [
        kw if kw.startswith(r"http") or kw.startswith(r"magnet")
        else f"{_WB_LEFT}{kw}{_WB_RIGHT}"
        for kw in keywords
    ]
    return re.compile("|".join(expressions), re.IGNORECASE | re.UNICODE)


_MLTB_RE = _compile_pattern(_MLTB_KEYWORDS)
_VPS_DOCKER_RE = _compile_pattern(_VPS_DOCKER_KEYWORDS)
_NOTES_REMINDERS_RE = _compile_pattern(_NOTES_REMINDERS_KEYWORDS)
_PRINTER_3D_RE = _compile_pattern(_PRINTER_3D_KEYWORDS)
_GENERIC_ACTION_RE = _compile_pattern(_GENERIC_ACTION_KEYWORDS)


def route_prompt(prompt: str) -> ReflexRoute:
    """Analyze a prompt in sub-millisecond time and return routing targets.

    Fails loudly if input is not a string. Empty prompts yield empty routes.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"route_prompt expects str, got {type(prompt).__name__}")

    if not prompt.strip():
        return ReflexRoute()

    matched_intents: list[str] = []
    tool_domains: list[str] = []

    include_mltb = bool(_MLTB_RE.search(prompt))
    if include_mltb:
        matched_intents.append("mltb")

    if _VPS_DOCKER_RE.search(prompt):
        tool_domains.append("vps_docker")
        matched_intents.append("vps_docker")

    if _NOTES_REMINDERS_RE.search(prompt):
        tool_domains.append("notes_reminders")
        matched_intents.append("notes_reminders")

    if _PRINTER_3D_RE.search(prompt):
        tool_domains.append("printer_3d")
        matched_intents.append("printer_3d")

    use_all_tools = False
    if not include_mltb and not tool_domains and _GENERIC_ACTION_RE.search(prompt):
        use_all_tools = True
        matched_intents.append("generic_action")

    return ReflexRoute(
        include_mltb=include_mltb,
        tool_domains=tuple(tool_domains),
        use_all_tools=use_all_tools,
        matched_intents=tuple(matched_intents),
    )
