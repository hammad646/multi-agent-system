"""Structured logging configuration using structlog.

Enforces JSON format, context tracking (request_id, workflow_id),
writing strictly to stderr (never stdout), and secret/email redaction.
"""
import sys
from typing import Any
import structlog
from app.config import settings

SENSITIVE_KEYS = {
    "api_key",
    "gemini_api_key",
    "google_api_key",
    "anthropic_api_key",
    "pinecone_api_key",
    "github_pat",

    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "secret",
    "password",
    "client_secret",
}

EMAIL_BODY_KEYS = {
    "body",
    "email_body",
    "raw_body",
}


def redact_sensitive_data(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Redact secrets and sensitive tokens from log events."""
    for key, value in list(event_dict.items()):
        lower_key = str(key).lower()
        if any(sens in lower_key for sens in SENSITIVE_KEYS):
            event_dict[key] = "[REDACTED]"
        elif not settings.LOG_EMAIL_BODIES and lower_key in EMAIL_BODY_KEYS:
            event_dict[key] = "[REDACTED_EMAIL_BODY]"
        elif isinstance(value, dict):
            event_dict[key] = redact_sensitive_dict(value)
    return event_dict


def redact_sensitive_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact nested dictionary values."""
    sanitized: dict[str, Any] = {}
    for k, v in d.items():
        lower_k = str(k).lower()
        if any(sens in lower_k for sens in SENSITIVE_KEYS):
            sanitized[k] = "[REDACTED]"
        elif not settings.LOG_EMAIL_BODIES and lower_k in EMAIL_BODY_KEYS:
            sanitized[k] = "[REDACTED_EMAIL_BODY]"
        elif isinstance(v, dict):
            sanitized[k] = redact_sensitive_dict(v)
        else:
            sanitized[k] = v
    return sanitized


def configure_logging() -> None:
    """Configure structlog processors and output handler."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_sensitive_data,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a bound structured logger instance."""
    logger = structlog.get_logger(name or "assistant")
    return logger


def bind_context(
    request_id: str | None = None, workflow_id: str | None = None
) -> None:
    """Bind request_id and/or workflow_id to the current execution context."""
    kwargs: dict[str, Any] = {}
    if request_id:
        kwargs["request_id"] = request_id
    if workflow_id:
        kwargs["workflow_id"] = workflow_id
    if kwargs:
        structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear contextual bound variables."""
    structlog.contextvars.clear_contextvars()


# Initialize logging configuration upon module import
configure_logging()
