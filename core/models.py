"""
Domain models using immutable dataclasses following ECC principles.
Supports both object attribute access, dict-like indexing, and mapping membership checks.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class LearnedRule:
    """Immutable representation of a persistent learned rule."""
    id: int
    user_id: str
    category: str
    rule: str
    trigger_context: Optional[str] = None
    created_at: str = ""
    is_active: bool = True

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, not {type(key).__name__}")
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "category": self.category,
            "rule": self.rule,
            "trigger_context": self.trigger_context,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }


@dataclass(frozen=True)
class UserFact:
    """Immutable representation of a remembered user fact."""
    id: int
    user_id: str
    fact: str
    created_at: str = ""

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, not {type(key).__name__}")
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "fact": self.fact,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ChatMessage:
    """Immutable representation of a conversation turn in chat history."""
    role: str
    content: str
    media_type: Optional[str] = None
    created_at: str = ""

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, not {type(key).__name__}")
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "media_type": self.media_type,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class VPSMetrics:
    """Immutable representation of VPS health and resource metrics."""
    cpu_percent: float
    cpu_count: int
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    disk_used_gb: float
    disk_free_gb: float
    disk_total_gb: float
    disk_percent: float
    uptime: str
    warning: Optional[str] = None
    text: str = ""

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, not {type(key).__name__}")
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "ram_percent": self.ram_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_free_gb": self.disk_free_gb,
            "disk_total_gb": self.disk_total_gb,
            "disk_percent": self.disk_percent,
            "uptime": self.uptime,
            "warning": self.warning,
            "text": self.text,
        }


@dataclass(frozen=True)
class ReflectionResult:
    """Immutable output from the reflection agent."""
    should_learn: bool
    type: str  # 'rule' or 'fact'
    content: str
    category: str = "general"
    trigger_context: str = "general"
    id: Optional[int] = None

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, not {type(key).__name__}")
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_learn": self.should_learn,
            "type": self.type,
            "content": self.content,
            "category": self.category,
            "trigger_context": self.trigger_context,
            "id": self.id,
        }


@dataclass(frozen=True)
class BackupResult:
    """Immutable representation of a database backup operation."""
    success: bool
    db_path: str
    size_bytes: int
    created_at: str
    error: Optional[str] = None

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, not {type(key).__name__}")
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "db_path": self.db_path,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "error": self.error,
        }


@dataclass(frozen=True)
class ScheduledReport:
    """Immutable representation of a generated scheduled report."""
    report_type: str
    text: str
    timestamp: str

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, not {type(key).__name__}")
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_type": self.report_type,
            "text": self.text,
            "timestamp": self.timestamp,
        }
