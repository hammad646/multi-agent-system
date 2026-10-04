"""Hardening and reliability tests per SPEC.md section 14, 18, and 19.

Covers:
- PostgreSQL dialect DDL generation compatibility for all models
- LangSmith tracing configuration behavior
- Timezone awareness validation (rejection of naive datetimes)
- Error taxonomy and structured response compliance
"""
from datetime import datetime, timezone
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.config import settings
from app.db.models import DocumentModel, ChunkModel, ScheduledEmailModel, WorkflowModel
from app.errors import AssistantException, create_error_response
from app.scheduler.service import parse_and_validate_schedule_time
from app.telemetry import setup_langsmith


def test_postgres_ddl_compatibility():
    """Verify that all SQLAlchemy models compile clean, valid DDL on the PostgreSQL dialect."""
    pg_dialect = postgresql.dialect()
    models = [DocumentModel, ChunkModel, ScheduledEmailModel, WorkflowModel]

    for model in models:
        ddl = str(CreateTable(model.__table__).compile(dialect=pg_dialect))
        assert "CREATE TABLE" in ddl
        assert model.__tablename__ in ddl

        # Ensure timezone-aware timestamp columns compile to TIMESTAMP WITH TIME ZONE
        if hasattr(model, "created_at") or hasattr(model, "uploaded_at") or hasattr(model, "scheduled_at"):
            assert "WITH TIME ZONE" in ddl


def test_langsmith_setup_configuration(monkeypatch):
    """Test LangSmith environment variable configuration when enabled vs disabled."""
    # When disabled
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", False)
    assert setup_langsmith() is False

    # When enabled with API key
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "ls_test_key_12345")
    monkeypatch.setattr(settings, "LANGSMITH_PROJECT", "test_assistant_project")

    assert setup_langsmith() is True
    import os
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_API_KEY") == "ls_test_key_12345"
    assert os.environ.get("LANGCHAIN_PROJECT") == "test_assistant_project"


def test_naive_datetime_rejection():
    """Verify naive datetimes are strictly rejected across scheduling interfaces."""
    # Naive string without offset
    with pytest.raises(ValueError, match="Naive datetimes are rejected"):
        parse_and_validate_schedule_time(datetime(2026, 10, 15, 14, 0))

    # Timezone-aware datetime succeeds
    aware_dt = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
    utc_dt, tz = parse_and_validate_schedule_time(aware_dt)
    assert utc_dt.tzinfo is not None


def test_structured_error_taxonomy():
    """Verify that structured errors match the taxonomy and never leak raw errors."""
    err = create_error_response("api_error", "External API call failed", retryable=True)
    assert err["ok"] is False
    assert err["error"]["code"] == "api_error"
    assert err["error"]["retryable"] is True
    assert "external" in err["error"]["message"].lower()

    app_err = AssistantException("rate_limited", "Rate limit exceeded", retryable=True)
    assert app_err.code == "rate_limited"
    assert app_err.retryable is True
    assert app_err.to_dict()["ok"] is False
