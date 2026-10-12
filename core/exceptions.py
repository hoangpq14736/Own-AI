"""
Application Exception Hierarchy following ECC standards.
Provides clear, typed errors without silently swallowing exceptions.
"""

class AppError(Exception):
    """Base exception for all application errors."""
    pass

class ConfigurationError(AppError):
    """Raised when environment or application configuration is invalid."""
    pass

class ValidationError(AppError):
    """Raised when input validation fails at system boundaries."""
    pass

class SecurityError(AppError):
    """Raised when security constraints or authorization checks fail."""
    pass

class GeminiAPIError(AppError):
    """Raised when Gemini API encounters an unrecoverable failure."""
    pass

class MediaGenerationError(AppError):
    """Raised when media generation (Image/Video/TTS) fails."""
    pass

class DockerManagerError(AppError):
    """Raised when Docker container operations fail."""
    pass
