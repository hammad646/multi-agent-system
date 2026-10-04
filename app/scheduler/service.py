"""Email scheduling service per SPEC.md section 11 and 12.

Provides application-level interface for creating, listing, cancelling,
and querying scheduled emails using ScheduledEmailRepository.
All SQL queries reside strictly in repositories.py (Guardrail 11).
"""
from datetime import datetime, timezone
import json
from typing import Any
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import get_session
from app.db.models import ScheduledEmailModel, utc_now
from app.db.repositories import ScheduledEmailRepository
from app.mcp_servers.gmail_server import validate_recipients


def parse_and_validate_schedule_time(scheduled_at: str | datetime, tz_name: str | None = None) -> tuple[datetime, str]:
    """Validate timezone-aware datetime and convert to UTC for storage."""
    target_tz = tz_name or settings.TIMEZONE
    if isinstance(scheduled_at, str):
        cleaned = scheduled_at.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
        except Exception as exc:
            raise ValueError(f"Invalid ISO 8601 datetime format for scheduled_at: '{scheduled_at}'") from exc
    elif isinstance(scheduled_at, datetime):
        dt = scheduled_at
    else:
        raise ValueError("scheduled_at must be an ISO 8601 string or datetime object.")

    if dt.tzinfo is None:
        raise ValueError(
            "Naive datetimes are rejected per SPEC.md guardrails. Timezone offset is required."
        )

    # Convert to UTC for storage
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt, target_tz


class SchedulerService:
    """Service layer for scheduled emails."""

    def __init__(self, session_factory: Any = None) -> None:
        self.session_factory = session_factory or get_session

    def schedule_email(
        self,
        to: str | list[str],
        subject: str,
        body: str,
        scheduled_at: str | datetime,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        user_id: str = "default_user",
        session: Session | None = None,
    ) -> ScheduledEmailModel:
        """Schedule an email for future delivery."""
        to_list, cc_list, bcc_list = validate_recipients(to, cc, bcc)
        utc_dt, target_tz = parse_and_validate_schedule_time(scheduled_at)

        email_model = ScheduledEmailModel(
            user_id=user_id,
            recipients=json.dumps(to_list),
            cc=json.dumps(cc_list) if cc_list else None,
            bcc=json.dumps(bcc_list) if bcc_list else None,
            subject=subject,
            body=body,
            scheduled_at=utc_dt,
            timezone=target_tz,
            status="pending",
            created_at=utc_now(),
        )

        def _do(sess: Session) -> ScheduledEmailModel:
            repo = ScheduledEmailRepository(sess)
            return repo.create(email_model)

        if session is not None:
            return _do(session)
        with self.session_factory() as sess:
            return _do(sess)

    def list_scheduled_emails(
        self,
        user_id: str | None = None,
        session: Session | None = None,
    ) -> list[ScheduledEmailModel]:
        """List all scheduled emails."""
        def _do(sess: Session) -> list[ScheduledEmailModel]:
            repo = ScheduledEmailRepository(sess)
            return repo.list_all(user_id=user_id)

        if session is not None:
            return _do(session)
        with self.session_factory() as sess:
            return _do(sess)

    def get_scheduled_email(
        self,
        email_id: str,
        session: Session | None = None,
    ) -> ScheduledEmailModel | None:
        """Get scheduled email by ID."""
        def _do(sess: Session) -> ScheduledEmailModel | None:
            repo = ScheduledEmailRepository(sess)
            return repo.get(email_id)

        if session is not None:
            return _do(session)
        with self.session_factory() as sess:
            return _do(sess)

    def cancel_scheduled_email(
        self,
        email_id: str,
        session: Session | None = None,
    ) -> bool:
        """Cancel a pending scheduled email."""
        def _do(sess: Session) -> bool:
            repo = ScheduledEmailRepository(sess)
            return repo.cancel(email_id)

        if session is not None:
            return _do(session)
        with self.session_factory() as sess:
            return _do(sess)
