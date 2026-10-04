"""Unit tests for Phase 1 Foundation components and guardrails."""
from datetime import datetime, timezone, timedelta
import io
import json
import os
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings, Settings
from app.logging import get_logger, bind_context, clear_context, redact_sensitive_data
from app.state import RouteDecision, PendingAction, AgentResult, AgentState
from app.errors import create_error_response, VALID_ERROR_CODES
from app.db.base import Base
from app.db.models import DocumentModel, ChunkModel, ScheduledEmailModel, WorkflowModel, utc_now
from app.db.repositories import DocumentRepository, ScheduledEmailRepository, WorkflowRepository
from app.testing.fakes import (
    FakeLLM,
    FakePinecone,
    FakeGmailClient,
    FakeCalendarClient,
    FakeGitHubClient,
)


@pytest.fixture
def in_memory_session():
    """Create an isolated in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


# --- 1. Config tests ---
def test_config_settings_defaults():
    """Verify settings loads properly with expected SPEC defaults."""
    assert settings.MODEL in {"gemini-2.5-flash", "gemini-3.8-flash"}
    assert hasattr(settings, "GEMINI_API_KEY")
    assert settings.PINECONE_INDEX == "pdf-rag"

    assert settings.TIMEZONE == "Asia/Karachi"
    assert settings.APPROVE_EVENT_CREATE is True
    assert settings.MAX_EMAIL_RECIPIENTS == 10
    assert settings.MAX_LATE_MINUTES == 60
    assert settings.LOG_EMAIL_BODIES is False
    assert settings.MOCK_MODE is False


def test_config_forbidden_flags_do_not_exist():
    """Verify non-configurable approval flags do NOT exist in Settings."""
    cfg = Settings()
    assert not hasattr(cfg, "APPROVE_EMAIL_SEND")
    assert not hasattr(cfg, "APPROVE_EMAIL_SCHEDULE")
    assert not hasattr(cfg, "APPROVE_EMAIL_DELETE")
    assert not hasattr(cfg, "APPROVE_EVENT_UPDATE")
    assert not hasattr(cfg, "APPROVE_EVENT_DELETE")


# --- 2. Logging and Redaction tests ---
def test_logging_redacts_secrets():
    """Verify secrets and API keys are redacted from logs."""
    event = {
        "event": "user_action",
        "api_key": "super-secret-key",
        "nested": {"token": "secret-token", "safe_val": 123},
        "email_body": "Confidential email body text",
    }
    redacted = redact_sensitive_data(None, "info", event)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert redacted["nested"]["safe_val"] == 123
    assert redacted["email_body"] == "[REDACTED_EMAIL_BODY]"


def test_logging_context_binding():
    """Verify request_id and workflow_id can be bound without error."""
    clear_context()
    bind_context(request_id="req-123", workflow_id="wf-456")
    logger = get_logger("test")
    # Logger bound without throwing
    assert logger is not None
    clear_context()


# --- 3. State Models tests ---
def test_state_models_instantiation():
    """Verify state models parse and validate correctly."""
    route = RouteDecision(
        intent="calendar",
        reasoning="User asked for tomorrow's schedule."
    )
    assert route.intent == "calendar"
    assert route.workflow is None

    action = PendingAction(
        id="act-1",
        tool="send_email",
        args={"to": "test@example.com"},
        summary="send_email to test@example.com",
        risk="high",
    )
    assert action.risk == "high"

    result = AgentResult(
        agent="calendar",
        status="ok",
        summary="Event created",
        data={"event_id": "evt-123"},
    )
    assert result.status == "ok"
    assert result.error is None


# --- 4. Errors Taxonomy tests ---
def test_error_taxonomy_format():
    """Verify error taxonomy response format matches SPEC.md section 5."""
    for code in VALID_ERROR_CODES:
        resp = create_error_response(code=code, message=f"Failed with {code}", retryable=True)
        assert resp["ok"] is False
        assert resp["error"]["code"] == code
        assert resp["error"]["message"] == f"Failed with {code}"
        assert resp["error"]["retryable"] is True


def test_invalid_error_code_defaults_to_unexpected():
    """Verify invalid error code falls back to 'unexpected'."""
    resp = create_error_response(code="non_existent_code", message="error")  # type: ignore
    assert resp["error"]["code"] == "unexpected"


# --- 5. Repositories tests ---
def test_document_repository(in_memory_session):
    """Verify DocumentRepository CRUD and cascade deletion."""
    repo = DocumentRepository(in_memory_session)

    doc = DocumentModel(
        document_id="doc-1",
        filename="test.pdf",
        title="Test Document",
        sha256="abc123sha",
        pages=2,
        chunk_count=2,
        status="ready",
    )
    chunks = [
        ChunkModel(chunk_id="chunk-1", page=1, text="First chunk text"),
        ChunkModel(chunk_id="chunk-2", page=2, text="Second chunk text"),
    ]

    # Create
    repo.add_document(doc, chunks)

    # Read by ID and SHA256
    retrieved = repo.get_document("doc-1")
    assert retrieved is not None
    assert retrieved.filename == "test.pdf"

    by_sha = repo.get_document_by_sha256("abc123sha")
    assert by_sha is not None
    assert by_sha.document_id == "doc-1"

    # Read chunks
    doc_chunks = repo.get_chunks_for_document("doc-1")
    assert len(doc_chunks) == 2

    # Cascade delete
    deleted = repo.delete_document("doc-1")
    assert deleted is True
    assert repo.get_document("doc-1") is None
    assert len(repo.get_chunks_for_document("doc-1")) == 0


def test_scheduled_email_repository_atomic_claim(in_memory_session):
    """Verify ScheduledEmailRepository creation, atomic claim, and mutual exclusion."""
    repo = ScheduledEmailRepository(in_memory_session)

    now = utc_now()
    email = ScheduledEmailModel(
        id="email-1",
        user_id="user-1",
        recipients="alice@example.com",
        subject="Meeting Agenda",
        body="Hello Alice",
        scheduled_at=now - timedelta(minutes=5),
        timezone="Asia/Karachi",
        status="pending",
    )
    repo.create(email)

    # 1. List pending
    pending = repo.list_pending(due_before=now)
    assert len(pending) == 1
    assert pending[0].id == "email-1"

    # 2. Worker 1 claims pending email -> SUCCESS
    claimed1 = repo.claim_pending("email-1")
    assert claimed1 is True

    # Check status changed to 'sending' and attempts incremented
    updated = repo.get("email-1")
    assert updated.status == "sending"
    assert updated.attempts == 1

    # 3. Worker 2 attempts concurrent claim -> FAILS (atomic mutual exclusion)
    claimed2 = repo.claim_pending("email-1")
    assert claimed2 is False

    # 4. Status update to sent
    repo.update_status("email-1", status="sent", sent_at=utc_now(), gmail_message_id="msg-abc")
    sent = repo.get("email-1")
    assert sent.status == "sent"
    assert sent.gmail_message_id == "msg-abc"


def test_scheduled_email_reset_stuck_sending(in_memory_session):
    """Verify emails stuck in 'sending' status can be reset to 'pending'."""
    repo = ScheduledEmailRepository(in_memory_session)
    email = ScheduledEmailModel(
        id="email-stuck",
        user_id="user-1",
        recipients="bob@example.com",
        subject="Stuck Email",
        body="Body",
        scheduled_at=utc_now() - timedelta(minutes=30),
        timezone="UTC",
        status="sending",
        created_at=utc_now() - timedelta(minutes=20),
    )
    repo.create(email)

    # Reset stuck jobs (threshold = 300 seconds)
    count = repo.reset_stuck_sending(threshold_seconds=300)
    assert count == 1

    refreshed = repo.get("email-stuck")
    assert refreshed.status == "pending"


def test_workflow_repository(in_memory_session):
    """Verify WorkflowRepository creation and status tracking."""
    repo = WorkflowRepository(in_memory_session)

    wf = repo.create(
        workflow_id="wf-1",
        workflow_type="schedule_and_email",
        steps={"step1": "completed"},
    )
    assert wf.id == "wf-1"
    assert wf.status == "pending"

    # Update status and progress
    updated = repo.update_status("wf-1", status="completed", steps={"step1": "completed", "step2": "completed"})
    assert updated.status == "completed"
    assert "step2" in updated.steps


# --- 6. Guardrails & Fakes Skeleton tests ---
def test_fakes_instantiation():
    """Verify testing fakes work offline."""
    llm = FakeLLM()
    assert llm.invoke("hello") == "Fake LLM response"

    pinecone = FakePinecone()
    idx = pinecone.Index("test-index")
    idx.upsert([("vec1", [0.1, 0.2], {"doc": "1"})], namespace="ns1")
    res = idx.query(vector=[0.1, 0.2], top_k=1, namespace="ns1")
    assert len(res["matches"]) == 1

    gmail = FakeGmailClient()
    draft = gmail.create_draft("to@ex.com", "Sub", "Body")
    assert draft["id"] is not None

    cal = FakeCalendarClient()
    evt = cal.create_event("Test Event", "2026-10-04T10:00:00Z", "2026-10-04T10:30:00Z")
    assert evt["id"] is not None

    gh = FakeGitHubClient()
    assert gh.get_repo("test/repo") is not None


def test_gitignore_contains_required_patterns():
    """Verify .gitignore blocks all secrets and local artifacts."""
    content = Path(".gitignore").read_text(encoding="utf-8")
    assert ".env" in content
    assert "token.json" in content
    assert "gcp-oauth.keys.json" in content
    assert "data/" in content
    assert ".venv/" in content


def test_guardrail_rejects_naive_datetime(in_memory_session):
    """Verify that naive datetimes are rejected per SPEC guardrail 7."""
    naive_dt = datetime(2026, 10, 4, 10, 0, 0)  # no tzinfo!
    email = ScheduledEmailModel(
        id="email-naive",
        user_id="user-1",
        recipients="bob@example.com",
        subject="Naive Date",
        body="Body",
        scheduled_at=naive_dt,
        timezone="UTC",
    )
    in_memory_session.add(email)
    with pytest.raises(Exception, match="Naive datetimes are rejected"):
        in_memory_session.commit()
    in_memory_session.rollback()


