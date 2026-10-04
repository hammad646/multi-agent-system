"""Email Agent per SPEC.md section 8, 10, 11, and 12.

Provides tools: search_emails, read_email, create_draft, list_drafts,
reply_to_email, send_email, delete_email, schedule_email,
list_scheduled_emails, cancel_scheduled_email.

Hard Rules:
- 'Write'/'draft' email creates a draft only. It never sends (Hard rule 4).
- Missing recipient returns needs_clarification.
- Validate recipients before any send.
- Show final To/Subject/Body in data.
"""
from datetime import datetime, timezone, timedelta
import re
from typing import Any
from langchain_core.tools import tool

from app.config import settings
from app.mcp_servers import gmail_server
from app.mcp_servers.gmail_server import validate_recipients
from app.scheduler.service import SchedulerService, parse_and_validate_schedule_time
from app.state import AgentResult


from app.guard import guard

def build_email_tools(
    scheduler_service: SchedulerService | None = None,
    guarded: bool = True,
) -> list[Any]:
    """Build LangChain tool wrappers for Gmail and Scheduler."""
    sched = scheduler_service or SchedulerService()

    @tool
    def search_emails(query: str = "", max_results: int = 10) -> dict[str, Any]:
        """Search Gmail messages."""
        return gmail_server.search_emails(query=query, max_results=max_results)

    @tool
    def read_email(message_id: str) -> dict[str, Any]:
        """Read full email message details by message ID."""
        return gmail_server.read_email(message_id=message_id)

    @tool
    def create_draft(
        to: str | list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an email draft without sending it."""
        return gmail_server.create_draft(to=to, subject=subject, body=body, cc=cc, bcc=bcc)

    @tool
    def list_drafts(max_results: int = 10) -> dict[str, Any]:
        """List Gmail drafts."""
        return gmail_server.list_drafts(max_results=max_results)

    @tool
    def send_draft(draft_id: str) -> dict[str, Any]:
        """Send a saved draft by draft ID."""
        return gmail_server.send_draft(draft_id=draft_id)

    @tool
    def send_email(
        to: str | list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send an email immediately."""
        return gmail_server.send_email(to=to, subject=subject, body=body, cc=cc, bcc=bcc)

    @tool
    def reply_to_email(
        message_id: str,
        body: str,
        reply_all: bool = False,
    ) -> dict[str, Any]:
        """Reply to an existing email message."""
        return gmail_server.reply_to_email(message_id=message_id, body=body, reply_all=reply_all)

    @tool
    def delete_email(message_id: str) -> dict[str, Any]:
        """Move an email message to trash."""
        return gmail_server.delete_email(message_id=message_id)

    @tool
    def schedule_email(
        to: str | list[str],
        subject: str,
        body: str,
        scheduled_at: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> dict[str, Any]:
        """Schedule an email to be sent at a future timezone-aware ISO 8601 time."""
        try:
            model = sched.schedule_email(
                to=to,
                subject=subject,
                body=body,
                scheduled_at=scheduled_at,
                cc=cc,
                bcc=bcc,
            )
            return {
                "ok": True,
                "scheduled_email": {
                    "id": model.id,
                    "to": model.recipients,
                    "subject": model.subject,
                    "body": model.body,
                    "scheduled_at": model.scheduled_at.isoformat() if model.scheduled_at else None,
                    "status": model.status,
                },
                "message": f"Email successfully scheduled for {scheduled_at}.",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @tool
    def list_scheduled_emails(user_id: str | None = None) -> dict[str, Any]:
        """List all scheduled emails."""
        models = sched.list_scheduled_emails(user_id=user_id)
        emails = [
            {
                "id": m.id,
                "recipients": m.recipients,
                "subject": m.subject,
                "scheduled_at": m.scheduled_at.isoformat() if m.scheduled_at else None,
                "status": m.status,
                "attempts": m.attempts,
            }
            for m in models
        ]
        return {"ok": True, "scheduled_emails": emails, "count": len(emails)}

    @tool
    def cancel_scheduled_email(email_id: str) -> dict[str, Any]:
        """Cancel a pending scheduled email."""
        success = sched.cancel_scheduled_email(email_id)
        return {
            "ok": success,
            "message": f"Scheduled email {email_id} cancelled." if success else f"Email {email_id} not found or not pending.",
        }

    raw_tools = [
        search_emails,
        read_email,
        create_draft,
        list_drafts,
        send_draft,
        send_email,
        reply_to_email,
        delete_email,
        schedule_email,
        list_scheduled_emails,
        cancel_scheduled_email,
    ]
    if guarded:
        return [guard(t) for t in raw_tools]
    return raw_tools


class EmailAgent:
    """Agent for managing Gmail emails, drafts, and scheduled dispatches."""

    def __init__(
        self,
        llm: Any | None = None,
        scheduler_service: SchedulerService | None = None,
        guarded: bool = False,
    ) -> None:
        self.llm = llm
        self.scheduler_service = scheduler_service or SchedulerService()
        self.tools = build_email_tools(self.scheduler_service, guarded=guarded)
        self.tool_map = {t.name: t for t in self.tools}

    def _extract_emails(self, text: str) -> list[str]:
        """Extract valid email addresses from text."""
        pattern = r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"
        return re.findall(pattern, text)

    def _extract_subject_and_body(self, text: str) -> tuple[str, str]:
        """Extract subject and body heuristic from user query."""
        subject = "Update"
        body = text

        body_match = re.search(r"(?:saying|body:?|message:?)\s+[\"']?([^\"'\n]+)[\"']?", text, re.IGNORECASE)
        if body_match:
            body = body_match.group(1).strip()

        subj_match = re.search(
            r"(?:about|subject:?)\s+([^,\.\n]+?)(?=\s+(?:saying|body|message|$))",
            text,
            re.IGNORECASE,
        )
        if subj_match:
            subject = subj_match.group(1).strip()
        elif "about" in text.lower():
            after_about = text.lower().split("about", 1)[1]
            if "saying" in after_about:
                subject = after_about.split("saying")[0].strip()

        return subject, body

    async def ainvoke(self, query: str) -> AgentResult:
        """Asynchronous execution."""
        return self.invoke(query)

    def invoke(self, query: str) -> AgentResult:
        """Execute Email agent logic with strict safety guardrails."""
        q_lower = query.lower()

        # 1. Draft or Write request (Hard Rule 4: Creates draft only, NEVER sends)
        is_draft_request = "draft" in q_lower or "write an email" in q_lower or "write email" in q_lower or "compose" in q_lower
        is_schedule_request = "schedule" in q_lower and ("email" in q_lower or "mail" in q_lower)
        is_send_request = not is_draft_request and not is_schedule_request and ("send" in q_lower and ("email" in q_lower or "mail" in q_lower))

        if is_draft_request or is_schedule_request or is_send_request:
            recipients = self._extract_emails(query)
            if not recipients:
                # Per SPEC: Missing recipient -> needs_clarification
                action = "schedule" if is_schedule_request else ("send" if is_send_request else "draft")
                return AgentResult(
                    agent="email",
                    status="needs_clarification",
                    summary=f"Recipient is required to {action} an email.",
                    clarification="Who would you like to address this email to? Please provide a valid email address.",
                    data={"missing_field": "recipient"},
                )

            # Validate recipient format
            try:
                validate_recipients(recipients)
            except Exception as exc:
                return AgentResult(
                    agent="email",
                    status="error",
                    summary=f"Invalid recipient: {exc}",
                    error={"type": "validation_error", "message": str(exc)},
                )

            subject, body = self._extract_subject_and_body(query)

            # Handle Draft (Hard rule 4: draft only, NEVER sends)
            if is_draft_request:
                res = self.tool_map["create_draft"].invoke({
                    "to": recipients,
                    "subject": subject,
                    "body": body,
                })
                if not res.get("ok"):
                    err = res.get("error", {})
                    return AgentResult(
                        agent="email",
                        status="error",
                        summary=f"Failed to create draft: {err.get('message')}",
                        error=err,
                    )
                draft_data = res.get("data", {}).get("draft", {})
                return AgentResult(
                    agent="email",
                    status="ok",
                    summary=f"Draft email created for {', '.join(recipients)} with subject '{subject}'.",
                    data={
                        "draft": draft_data,
                        "to": recipients,
                        "subject": subject,
                        "body": body,
                    },
                )

            # Handle Schedule Email
            if is_schedule_request:
                # Extract time or check for ambiguous date
                # If query contains explicit ISO or relative time
                iso_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)", query)
                if iso_match:
                    scheduled_at = iso_match.group(0)
                else:
                    # Check if relative e.g. "in 2 minutes" or "tomorrow at 10am"
                    if "2 minutes" in q_lower or "two minutes" in q_lower:
                        now_tz = datetime.now(timezone.utc)
                        scheduled_at = (now_tz + timedelta(minutes=2)).isoformat()
                    else:
                        # Ambiguous schedule time -> needs_clarification (Hard rule 7)
                        return AgentResult(
                            agent="email",
                            status="needs_clarification",
                            summary="Ambiguous scheduled time. Clarification required.",
                            clarification=f"When exactly should this email be sent? Please provide a specific time and date in {settings.TIMEZONE}.",
                            data={"to": recipients, "subject": subject, "body": body},
                        )

                sched_res = self.tool_map["schedule_email"].invoke({
                    "to": recipients,
                    "subject": subject,
                    "body": body,
                    "scheduled_at": scheduled_at,
                })
                if sched_res.get("status") == "rejected":
                    return AgentResult(
                        agent="email",
                        status="rejected",
                        summary=sched_res.get("summary", "Action 'schedule_email' was rejected by user."),
                        data={"tool": "schedule_email", "rejected": True, "reason": sched_res.get("reason")},
                    )
                if not sched_res.get("ok"):
                    return AgentResult(
                        agent="email",
                        status="error",
                        summary=f"Failed to schedule email: {sched_res.get('error')}",
                        error={"type": "validation_error", "message": sched_res.get("error")},
                    )

                return AgentResult(
                    agent="email",
                    status="ok",
                    summary=f"Email scheduled for delivery to {', '.join(recipients)} at {scheduled_at}.",
                    data={
                        "scheduled_email": sched_res.get("scheduled_email"),
                        "to": recipients,
                        "subject": subject,
                        "body": body,
                    },
                )

            # Handle Send Email (Must display final To/Subject/Body in data)
            if is_send_request:
                send_res = self.tool_map["send_email"].invoke({
                    "to": recipients,
                    "subject": subject,
                    "body": body,
                })
                if send_res.get("status") == "rejected":
                    return AgentResult(
                        agent="email",
                        status="rejected",
                        summary=send_res.get("summary", "Action 'send_email' was rejected by user."),
                        data={"tool": "send_email", "rejected": True, "reason": send_res.get("reason")},
                    )
                if not send_res.get("ok"):
                    err = send_res.get("error", {})
                    return AgentResult(
                        agent="email",
                        status="error",
                        summary=f"Failed to send email: {err.get('message')}",
                        error=err,
                    )
                return AgentResult(
                    agent="email",
                    status="ok",
                    summary=f"Email sent to {', '.join(recipients)} with subject '{subject}'.",
                    data={
                        "sent": send_res.get("data", {}).get("sent"),
                        "to": recipients,
                        "subject": subject,
                        "body": body,
                    },
                )

        # 2. Search emails
        if "search" in q_lower or "find email" in q_lower or "inbox" in q_lower:
            search_query = ""
            for token in ["search emails for", "search email for", "find emails about", "search for"]:
                if token in q_lower:
                    search_query = q_lower.split(token, 1)[1].strip()
                    break

            res = self.tool_map["search_emails"].invoke({"query": search_query})
            if not res.get("ok"):
                return AgentResult(agent="email", status="error", summary=str(res.get("error")), error=res.get("error"))

            messages = res.get("data", {}).get("messages", [])
            return AgentResult(
                agent="email",
                status="ok",
                summary=f"Found {len(messages)} matching email(s).",
                data={"messages": messages},
            )

        # 3. List drafts
        if "list drafts" in q_lower or "show drafts" in q_lower or "drafts" in q_lower:
            res = self.tool_map["list_drafts"].invoke({})
            drafts = res.get("data", {}).get("drafts", [])
            return AgentResult(
                agent="email",
                status="ok",
                summary=f"Found {len(drafts)} draft(s).",
                data={"drafts": drafts},
            )

        # 4. List scheduled emails
        if "list scheduled" in q_lower or "show scheduled" in q_lower:
            res = self.tool_map["list_scheduled_emails"].invoke({})
            emails = res.get("scheduled_emails", [])
            return AgentResult(
                agent="email",
                status="ok",
                summary=f"Found {len(emails)} scheduled email(s).",
                data={"scheduled_emails": emails},
            )

        # Fallback ready response
        return AgentResult(
            agent="email",
            status="ok",
            summary="Email agent ready. Specify an action such as drafting, sending, or scheduling an email.",
            data={},
        )
