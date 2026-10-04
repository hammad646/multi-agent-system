"""Telemetry and LangSmith tracing setup per SPEC.md section 14 and 18."""
import os
from app.config import settings
from app.logging import get_logger

logger = get_logger("telemetry")


def setup_langsmith() -> bool:
    """Configure LangChain / LangSmith environment variables if tracing is enabled.

    Returns:
        bool: True if LangSmith tracing was configured and enabled, False otherwise.
    """
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT or "assistant"
        logger.info(
            "langsmith_tracing_enabled",
            project=settings.LANGSMITH_PROJECT,
        )
        return True
    return False
