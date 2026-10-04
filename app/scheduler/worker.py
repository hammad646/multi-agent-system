"""Background email scheduling worker per SPEC.md section 12.

Polls database for due scheduled emails, claims rows atomically,
and delivers them directly via the Gmail client.
Strictly NO LLM INVOLVEMENT at send time (Guardrail 5).
All SQL resides in app/db/repositories.py (Guardrail 11).
"""
from datetime import datetime, timezone
import json
import signal
import sys
import time
from typing import Any
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import get_session
from app.db.models import ScheduledEmailModel, utc_now
from app.db.repositories import ScheduledEmailRepository
from app.logging import get_logger
from app.mcp_servers.gmail_server import get_gmail_client, validate_recipients

logger = get_logger("scheduler.worker")

_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("scheduler_worker_stopping", signal=signum)


def parse_stored_recipients(raw: str | None) -> list[str]:
    """Parse recipients from JSON or comma-separated string."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str):
            return [parsed.strip()]
    except Exception:
        pass
    return [r.strip() for r in raw.split(",") if r.strip()]


def process_due_emails(
    session_factory: Any = None,
    gmail_client: Any = None,
    now_override: datetime | None = None,
    max_late_minutes: int | None = None,
) -> dict[str, int]:
    """Process a single pass of due scheduled emails.

    Returns:
        Summary counts: {"sent": int, "failed": int, "missed_window": int, "claimed": int}
    """
    factory = session_factory or get_session
    now_utc = now_override or utc_now()
    late_limit = max_late_minutes if max_late_minutes is not None else settings.MAX_LATE_MINUTES
    client = gmail_client or get_gmail_client()

    counts = {"sent": 0, "failed": 0, "missed_window": 0, "claimed": 0}

    with factory() as session:
        repo = ScheduledEmailRepository(session)
        due_emails = repo.list_pending(due_before=now_utc)

        for email in due_emails:
            # 1. Missed-window check: if due while offline and lateness > MAX_LATE_MINUTES
            scheduled_utc = (
                email.scheduled_at
                if email.scheduled_at.tzinfo
                else email.scheduled_at.replace(tzinfo=timezone.utc)
            )
            lateness_seconds = (now_utc - scheduled_utc).total_seconds()
            lateness_minutes = lateness_seconds / 60.0

            if lateness_minutes > late_limit:
                # Atomically claim before marking failed
                if repo.claim_pending(email.id):
                    repo.update_status(
                        email_id=email.id,
                        status="failed",
                        last_error=f"missed_window: Job due {lateness_minutes:.1f}m ago exceeds MAX_LATE_MINUTES ({late_limit}m)",
                    )
                    counts["missed_window"] += 1
                    counts["failed"] += 1
                    logger.warn(
                        "scheduled_email_missed_window",
                        email_id=email.id,
                        lateness_minutes=lateness_minutes,
                        max_late_minutes=late_limit,
                    )
                continue

            # 2. Atomic claim: UPDATE scheduled_emails SET status='sending', attempts=attempts+1
            # Guarantees that two concurrent workers never double-send the same email.
            if not repo.claim_pending(email.id):
                # Another worker claimed it or it was cancelled
                continue

            counts["claimed"] += 1
            # Refresh model instance to get updated attempts count
            updated_email = repo.get(email.id)
            current_attempts = updated_email.attempts if updated_email else email.attempts + 1

            # 3. Re-validate recipients before sending
            try:
                to_list = parse_stored_recipients(email.recipients)
                cc_list = parse_stored_recipients(email.cc)
                bcc_list = parse_stored_recipients(email.bcc)
                to_val, cc_val, bcc_val = validate_recipients(to_list, cc_list, bcc_list)
            except Exception as val_exc:
                repo.update_status(
                    email_id=email.id,
                    status="failed",
                    last_error=f"validation_error: {val_exc}",
                )
                counts["failed"] += 1
                logger.error("scheduled_email_invalid_recipients", email_id=email.id, error=str(val_exc))
                continue

            # 4. Direct delivery via Gmail client (Strictly NO LLM involvement)
            try:
                if hasattr(client, "send_email"):
                    send_res = client.send_email(
                        to=to_val,
                        subject=email.subject,
                        body=email.body,
                        cc=cc_val or None,
                        bcc=bcc_val or None,
                    )
                elif hasattr(client, "users"):
                    from app.mcp_servers.gmail_server import create_mime_message, encode_mime_message
                    mime_msg = create_mime_message(to_val, email.subject, email.body, cc_val or None, bcc_val or None)
                    raw = encode_mime_message(mime_msg)
                    send_res = client.users().messages().send(userId="me", body={"raw": raw}).execute()
                else:
                    # Generic send call
                    send_res = client.send(
                        to=to_val,
                        subject=email.subject,
                        body=email.body,
                    )

                msg_id = send_res.get("id") if isinstance(send_res, dict) else str(send_res)
                repo.update_status(
                    email_id=email.id,
                    status="sent",
                    sent_at=utc_now(),
                    gmail_message_id=msg_id,
                )
                counts["sent"] += 1
                logger.info(
                    "scheduled_email_sent_successfully",
                    email_id=email.id,
                    gmail_message_id=msg_id,
                    attempts=current_attempts,
                )

            except Exception as send_exc:
                err_str = str(send_exc)
                is_auth_error = (
                    "invalid_grant" in err_str.lower()
                    or "auth_error" in err_str.lower()
                    or "unauthorized" in err_str.lower()
                )

                if is_auth_error:
                    # In Google OAuth testing mode, tokens expire after 7 days
                    # Mark failed immediately rather than retrying forever
                    repo.update_status(
                        email_id=email.id,
                        status="failed",
                        last_error=(
                            f"auth_error: Google token expired or invalid ({err_str}). "
                            "Re-run 'python -m app.mcp_servers.google_auth' to re-authenticate."
                        ),
                    )
                    counts["failed"] += 1
                    logger.error("scheduled_email_auth_failure", email_id=email.id, error=err_str)

                elif current_attempts >= 3:
                    # Exceeded maximum retry attempts (3 retries)
                    repo.update_status(
                        email_id=email.id,
                        status="failed",
                        last_error=f"Max retries (3) exceeded: {err_str}",
                    )
                    counts["failed"] += 1
                    logger.error("scheduled_email_max_retries_exceeded", email_id=email.id, attempts=current_attempts)

                else:
                    # Temporary error: reset to pending for backoff retry on subsequent pass
                    repo.update_status(
                        email_id=email.id,
                        status="pending",
                        last_error=f"Temporary send failure (attempt {current_attempts}): {err_str}",
                    )
                    logger.warn(
                        "scheduled_email_temporary_error_retry_scheduled",
                        email_id=email.id,
                        attempts=current_attempts,
                        error=err_str,
                    )

    return counts


def run_worker(
    poll_interval_seconds: float = 15.0,
    run_once: bool = False,
    session_factory: Any = None,
) -> None:
    """Run worker daemon process polling for due emails."""
    factory = session_factory or get_session

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        # Signals may not be assignable on all threads/platforms
        pass

    logger.info("scheduler_worker_started", poll_interval_seconds=poll_interval_seconds)

    # Startup recovery: reset emails stuck in 'sending' status from previous crashed workers
    with factory() as session:
        repo = ScheduledEmailRepository(session)
        reset_count = repo.reset_stuck_sending(threshold_seconds=0)
        if reset_count > 0:
            logger.info("scheduler_reset_stuck_sending_on_startup", count=reset_count)

    while not _shutdown_requested:
        try:
            counts = process_due_emails(session_factory=factory)
            if counts["sent"] > 0 or counts["failed"] > 0:
                logger.info("scheduler_worker_pass_completed", **counts)
        except Exception as exc:
            logger.error("scheduler_worker_pass_error", error=str(exc))

        if run_once or _shutdown_requested:
            break

        time.sleep(poll_interval_seconds)

    logger.info("scheduler_worker_stopped")


if __name__ == "__main__":
    run_worker()
