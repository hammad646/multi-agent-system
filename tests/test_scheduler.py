"""Tests for persistent email scheduler per SPEC.md section 12, 18, and 19.

Covers:
- Scheduled email persisted with UTC datetime
- Worker sends due email without LLM involvement
- Worker restart keeps jobs (resets stuck sending rows)
- Missed email within window (<= MAX_LATE_MINUTES) sends on startup
- Missed email outside window (> MAX_LATE_MINUTES) marked failed with missed_window
- Atomic claim prevents two concurrent workers from double-sending
- Temporary failure retries up to 3 times before setting failed
- Auth error (invalid_grant) fails immediately with auth_error
- Cancellation of scheduled emails
"""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import ScheduledEmailModel, utc_now
from app.db.repositories import ScheduledEmailRepository
from app.scheduler.service import SchedulerService
from app.scheduler.worker import process_due_emails
from app.testing.fakes import FakeGmailClient


@pytest.fixture
def sqlite_session_factory():
    """Create a persistent shared SQLite database for scheduler tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    sm = sessionmaker(bind=engine)

    @contextmanager
    def _get_session():
        session = sm()
        try:
            yield session
        finally:
            session.close()

    try:
        yield _get_session
    finally:
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def fake_gmail():
    return FakeGmailClient()


def test_schedule_email_persistence(sqlite_session_factory):
    """Verify scheduling an email persists a record in DB with UTC timestamp."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    scheduled_time = (utc_now() + timedelta(hours=2)).isoformat()

    model = service.schedule_email(
        to="client@example.com",
        subject="Project Quote",
        body="Here is the project proposal.",
        scheduled_at=scheduled_time,
    )

    assert model.id is not None
    assert model.status == "pending"
    assert "client@example.com" in model.recipients
    assert model.scheduled_at.tzinfo is not None

    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        stored = repo.get(model.id)
        assert stored is not None
        assert stored.subject == "Project Quote"
        assert stored.status == "pending"


def test_worker_sends_due_email(sqlite_session_factory, fake_gmail):
    """Worker picks up due email, claims atomically, sends via Gmail client (no LLM)."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    due_time = (utc_now() - timedelta(minutes=1)).isoformat()

    email = service.schedule_email(
        to="subscriber@example.com",
        subject="Daily Brief",
        body="Top stories of the day.",
        scheduled_at=due_time,
    )

    counts = process_due_emails(
        session_factory=sqlite_session_factory,
        gmail_client=fake_gmail,
    )

    assert counts["sent"] == 1
    assert len(fake_gmail.sent_messages) == 1
    sent_msg = fake_gmail.sent_messages[0]
    assert sent_msg["to"] == ["subscriber@example.com"]
    assert sent_msg["subject"] == "Daily Brief"

    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        updated = repo.get(email.id)
        assert updated.status == "sent"
        assert updated.sent_at is not None
        assert updated.gmail_message_id is not None


def test_worker_restart_keeps_jobs(sqlite_session_factory, fake_gmail):
    """Worker restart resets rows stuck in 'sending' status back to 'pending'."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    due_time = (utc_now() - timedelta(minutes=2)).isoformat()

    email = service.schedule_email(
        to="ops@example.com",
        subject="Alert",
        body="CPU threshold exceeded.",
        scheduled_at=due_time,
    )

    # Simulate a crash leaving the row in 'sending' state
    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        claimed = repo.claim_pending(email.id)
        assert claimed is True
        crashed_row = repo.get(email.id)
        assert crashed_row.status == "sending"

    # New worker starts up and resets stuck sending rows
    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        reset_count = repo.reset_stuck_sending(threshold_seconds=0)
        assert reset_count == 1
        reset_row = repo.get(email.id)
        assert reset_row.status == "pending"

    # Now the restarted worker processes due emails and successfully delivers
    counts = process_due_emails(
        session_factory=sqlite_session_factory,
        gmail_client=fake_gmail,
    )
    assert counts["sent"] == 1
    assert len(fake_gmail.sent_messages) == 1


def test_missed_window_within_limit(sqlite_session_factory, fake_gmail):
    """Email due while offline sends on startup if lateness <= MAX_LATE_MINUTES."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    # Due 30 minutes ago (within limit of 60m)
    due_30m_ago = (utc_now() - timedelta(minutes=30)).isoformat()

    email = service.schedule_email(
        to="user@example.com",
        subject="Catchup Email",
        body="Sent after recovery.",
        scheduled_at=due_30m_ago,
    )

    counts = process_due_emails(
        session_factory=sqlite_session_factory,
        gmail_client=fake_gmail,
        max_late_minutes=60,
    )

    assert counts["sent"] == 1
    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        updated = repo.get(email.id)
        assert updated.status == "sent"


def test_missed_window_exceeded(sqlite_session_factory, fake_gmail):
    """Email due while offline > MAX_LATE_MINUTES is marked failed with missed_window."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    # Due 90 minutes ago (exceeds limit of 60m)
    due_90m_ago = (utc_now() - timedelta(minutes=90)).isoformat()

    email = service.schedule_email(
        to="late@example.com",
        subject="Expired Promo",
        body="This promo is already over.",
        scheduled_at=due_90m_ago,
    )

    counts = process_due_emails(
        session_factory=sqlite_session_factory,
        gmail_client=fake_gmail,
        max_late_minutes=60,
    )

    assert counts["sent"] == 0
    assert counts["missed_window"] == 1

    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        updated = repo.get(email.id)
        assert updated.status == "failed"
        assert "missed_window" in updated.last_error
    assert len(fake_gmail.sent_messages) == 0


def test_two_workers_atomic_claim_no_double_send(sqlite_session_factory):
    """Atomic claim ensures two workers cannot both claim and double-send an email."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    due_time = (utc_now() - timedelta(seconds=10)).isoformat()

    email = service.schedule_email(
        to="once@example.com",
        subject="Important",
        body="Send exactly once.",
        scheduled_at=due_time,
    )

    with sqlite_session_factory() as session1, sqlite_session_factory() as session2:
        repo1 = ScheduledEmailRepository(session1)
        repo2 = ScheduledEmailRepository(session2)

        # Worker 1 claims
        claim1 = repo1.claim_pending(email.id)
        # Worker 2 attempts to claim the same email
        claim2 = repo2.claim_pending(email.id)

        assert claim1 is True
        assert claim2 is False


def test_retry_and_eventual_failure(sqlite_session_factory, fake_gmail):
    """Temporary errors retry up to 3 times before setting status to failed."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    due_time = (utc_now() - timedelta(minutes=1)).isoformat()

    email = service.schedule_email(
        to="retry@example.com",
        subject="Flaky service",
        body="Try sending.",
        scheduled_at=due_time,
    )

    fake_gmail.simulate_failure = Exception("503 Service Unavailable")

    # Pass 1: attempt 1 -> reset to pending
    process_due_emails(session_factory=sqlite_session_factory, gmail_client=fake_gmail)
    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        row = repo.get(email.id)
        assert row.status == "pending"
        assert row.attempts == 1

    # Pass 2: attempt 2 -> reset to pending
    process_due_emails(session_factory=sqlite_session_factory, gmail_client=fake_gmail)
    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        row = repo.get(email.id)
        assert row.status == "pending"
        assert row.attempts == 2

    # Pass 3: attempt 3 -> marked failed
    process_due_emails(session_factory=sqlite_session_factory, gmail_client=fake_gmail)
    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        row = repo.get(email.id)
        assert row.status == "failed"
        assert row.attempts == 3
        assert "Max retries (3) exceeded" in row.last_error


def test_auth_error_fails_immediately(sqlite_session_factory, fake_gmail):
    """Invalid grant / auth error fails immediately without retrying up to 3 times."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    due_time = (utc_now() - timedelta(minutes=1)).isoformat()

    email = service.schedule_email(
        to="auth@example.com",
        subject="Security Notice",
        body="Token expired test.",
        scheduled_at=due_time,
    )

    fake_gmail.simulate_failure = Exception("invalid_grant: Token has been expired or revoked.")

    process_due_emails(session_factory=sqlite_session_factory, gmail_client=fake_gmail)

    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        row = repo.get(email.id)
        assert row.status == "failed"
        assert row.attempts == 1
        assert "auth_error" in row.last_error
        assert "python -m app.mcp_servers.google_auth" in row.last_error


def test_cancel_scheduled_email(sqlite_session_factory, fake_gmail):
    """Cancelled scheduled email is not picked up or sent by the worker."""
    service = SchedulerService(session_factory=sqlite_session_factory)
    due_time = (utc_now() - timedelta(minutes=1)).isoformat()

    email = service.schedule_email(
        to="cancelled@example.com",
        subject="Meeting Cancelled",
        body="Do not send.",
        scheduled_at=due_time,
    )

    cancelled = service.cancel_scheduled_email(email.id)
    assert cancelled is True

    counts = process_due_emails(session_factory=sqlite_session_factory, gmail_client=fake_gmail)
    assert counts["sent"] == 0
    assert len(fake_gmail.sent_messages) == 0

    with sqlite_session_factory() as session:
        repo = ScheduledEmailRepository(session)
        row = repo.get(email.id)
        assert row.status == "cancelled"
