"""
Agent Function Calling Tools for Gemini Multimodal Assistant.
Provides high-level system inspection, memory scheduling, and web reading tools per ECC.
"""

from __future__ import annotations
from typing import Optional, List, Callable
import logging

from core.vps_manager import get_vps_health, list_docker_containers, restart_mltb_container
from core.memory import MemoryEngine

logger = logging.getLogger(__name__)

_CURRENT_MEMORY_ENGINE: Optional[MemoryEngine] = None
_CURRENT_USER_ID: Optional[str] = None


def set_agent_context(memory_engine: Optional[MemoryEngine] = None, user_id: Optional[str] = None):
    """Sets the active memory engine and user_id for tool executions."""
    global _CURRENT_MEMORY_ENGINE, _CURRENT_USER_ID
    _CURRENT_MEMORY_ENGINE = memory_engine
    _CURRENT_USER_ID = user_id


def get_vps_status() -> str:
    """Tra cứu chỉ số phần cứng VPS hiện tại gồm CPU, RAM, dung lượng ổ đĩa và thời gian hoạt động (uptime)."""
    try:
        metrics = get_vps_health()
        return metrics.text
    except Exception as e:
        return f"Lỗi tra cứu VPS: {e}"


def get_docker_containers() -> str:
    """Xem danh sách các container Docker đang chạy trên VPS (như mirrorbot-app-1, portainer, own-ai) và trạng thái của chúng."""
    try:
        return list_docker_containers()
    except Exception as e:
        return f"Lỗi truy vấn Docker: {e}"


def restart_docker_container(container_name: str) -> str:
    """Khởi động lại một container Docker trên VPS khi người dùng yêu cầu hoặc khi container bị đơ/treo."""
    try:
        return restart_mltb_container(container_name=container_name)
    except Exception as e:
        return f"Lỗi khởi động lại container: {e}"


def create_scheduled_reminder(task: str, remind_at_text: str) -> str:
    """
    Đặt lịch nhắc việc hoặc hẹn giờ thông báo cho người dùng.
    Tham số:
      - task: Nội dung công việc cần nhắc (ví dụ: 'Đi họp', 'Kiểm tra tải file').
      - remind_at_text: Thời gian nhắc (ví dụ: '30m', '1h', '18:30', '19h').
    """
    from core.reminder_parser import parse_reminder_text
    if not _CURRENT_MEMORY_ENGINE or not _CURRENT_USER_ID:
        return "⚠️ Không thể lưu nhắc nhở: Ngữ cảnh người dùng chưa được khởi tạo."
    try:
        remind_at, parsed_task = parse_reminder_text(f"{remind_at_text} {task}")
        rem_id = _CURRENT_MEMORY_ENGINE.add_reminder(
            user_id=_CURRENT_USER_ID,
            task=parsed_task,
            remind_at=remind_at
        )
        return f"✅ Đã đặt lịch nhắc việc thành công (#{rem_id}) vào lúc {remind_at}: '{parsed_task}'."
    except Exception as e:
        return f"❌ Lỗi khi đặt nhắc nhở: {str(e)}"


def save_personal_note(content: str, tags: str = "") -> str:
    """Lưu một ghi chú cá nhân, mã cấu hình, liên kết hoặc đoạn văn bản quan trọng vào kho ghi chú của người dùng."""
    if not _CURRENT_MEMORY_ENGINE or not _CURRENT_USER_ID:
        return "⚠️ Không thể lưu ghi chú: Ngữ cảnh người dùng chưa được khởi tạo."
    try:
        note_id = _CURRENT_MEMORY_ENGINE.add_note(user_id=_CURRENT_USER_ID, text=content, tags=tags)
        return f"✅ Đã lưu ghi chú #{note_id} thành công với thẻ [{tags}]: '{content[:50]}...'"
    except Exception as e:
        return f"❌ Lỗi khi lưu ghi chú: {e}"


def search_personal_notes(keyword: str) -> str:
    """Tìm kiếm trong danh sách ghi chú cá nhân hoặc bookmark của người dùng theo từ khóa hoặc chủ đề."""
    if not _CURRENT_MEMORY_ENGINE or not _CURRENT_USER_ID:
        return "⚠️ Không thể tìm kiếm ghi chú: Ngữ cảnh người dùng chưa được khởi tạo."
    try:
        notes = _CURRENT_MEMORY_ENGINE.search_notes(user_id=_CURRENT_USER_ID, query=keyword, top_k=5)
        if not notes:
            return f"Không tìm thấy ghi chú nào phù hợp với từ khóa '{keyword}'."
        results = [f"- [#{n.id}] {n.text} (Thẻ: {n.tags or 'không có'})" for n in notes]
        return "Các ghi chú tìm được:\n" + "\n".join(results)
    except Exception as e:
        return f"❌ Lỗi tìm kiếm ghi chú: {e}"


def get_all_agent_tools() -> List[Callable]:
    """Returns the list of callable functions for Gemini Function Calling."""
    return [
        get_vps_status,
        get_docker_containers,
        restart_docker_container,
        create_scheduled_reminder,
        save_personal_note,
        search_personal_notes,
    ]

