"""Explicit schedule_and_email workflow subgraph per SPEC.md section 9.

Flow:
parse_request -> find availability -> pick slot & find conflicts ->
APPROVAL (create event) -> create_event -> create_draft ->
APPROVAL (send email) -> send_email -> summarize

Rules:
- Passes data through state["artifacts"] (event, draft, etc.)
- Rejecting event approval: no event is created, workflow ends, summary says so.
- Rejecting email approval: event remains created, no email is sent, summary says so.
- Mid-step error: stops workflow, reports completed steps vs failed step.
- Persists progress to WorkflowRepository (workflows table).
"""
from datetime import datetime, timezone, timedelta
import json
import re
from typing import Any
from zoneinfo import ZoneInfo
from langgraph.types import interrupt

from app.config import settings
from app.db.base import get_session
from app.db.repositories import WorkflowRepository
from app.logging import get_logger
from app.mcp_servers import calendar_server, gmail_server
from app.state import AgentState, AgentResult

logger = get_logger("workflows.schedule_and_email")


def get_workflow_repo(session_factory: Any = None) -> tuple[WorkflowRepository, Any]:
    """Obtain WorkflowRepository and session context."""
    factory = session_factory or get_session
    session = factory()
    return WorkflowRepository(session), session


def parse_schedule_request(query: str) -> dict[str, Any]:
    """Parse attendee email, title/agenda, and date/time hints from user query."""
    # 1. Attendee email
    email_match = re.findall(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b", query)
    attendee = email_match[0] if email_match else "attendee@example.com"

    # 2. Title / Agenda
    title = "Meeting"
    if "for " in query.lower():
        after_for = query.lower().split("for ", 1)[1]
        title_cand = after_for.split("tomorrow")[0].split("at")[0].split("and")[0].strip()
        if title_cand:
            title = title_cand.title()
    elif "meeting with" in query.lower():
        title = "Sync Meeting"

    # 3. Target datetime calculation
    try:
        tz = ZoneInfo(settings.TIMEZONE)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)

    # Default to tomorrow at 14:00
    start_dt = (now + timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(minutes=30)

    return {
        "attendee": attendee,
        "title": title,
        "start_iso": start_dt.isoformat(),
        "end_iso": end_dt.isoformat(),
        "duration_minutes": 30,
        "agenda": f"Discussion regarding {title}",
    }


def execute_schedule_and_email_workflow(
    state: AgentState,
    session_factory: Any = None,
) -> dict[str, Any]:
    """Execute the schedule_and_email workflow with two distinct human approvals.

    Handles interrupts for event creation and email sending independently.
    """
    artifacts = dict(state.get("artifacts") or {})
    completed_steps: list[str] = list(artifacts.get("completed_steps", []))

    # Initialize workflow record in DB if not already present
    workflow_id = state.get("workflow_id") or f"wf_{state.get('request_id', 'req')[:16]}"
    factory = session_factory or get_session

    with factory() as session:
        repo = WorkflowRepository(session)
        wf_record = repo.get(workflow_id)
        if not wf_record:
            repo.create(workflow_id=workflow_id, workflow_type="schedule_and_email", steps={"completed_steps": []})
        else:
            try:
                persisted = json.loads(wf_record.steps) if isinstance(wf_record.steps, str) else wf_record.steps
                if isinstance(persisted, dict):
                    for k, v in persisted.items():
                        if k not in artifacts:
                            artifacts[k] = v
                    if "completed_steps" in persisted:
                        for s in persisted["completed_steps"]:
                            if s not in completed_steps:
                                completed_steps.append(s)
            except Exception as e:
                logger.warn("failed_to_restore_persisted_workflow_steps", error=str(e))

    artifacts["completed_steps"] = completed_steps

    # STEP 1: Parse request (idempotent)
    if "request" not in artifacts:
        query = ""
        for m in reversed(state.get("messages", [])):
            if hasattr(m, "content") and m.content:
                query = str(m.content)
                break
        req = parse_schedule_request(query)
        artifacts["request"] = req
        if "parse_request" not in completed_steps:
            completed_steps.append("parse_request")
        artifacts["completed_steps"] = completed_steps

        try:
            with factory() as session:
                repo = WorkflowRepository(session)
                repo.update_status(workflow_id, "running", steps=artifacts)
        except Exception:
            pass

    req = artifacts["request"]
    title = req["title"]
    attendee = req["attendee"]
    start_iso = req["start_iso"]
    end_iso = req["end_iso"]

    # STEP 2: Find availability and conflicts
    if "find_availability" not in completed_steps:
        try:
            conflict_res = calendar_server.find_conflicts(start_iso=start_iso, end_iso=end_iso)
            if conflict_res.get("data", {}).get("has_conflicts"):
                conflicts = conflict_res.get("data", {}).get("conflicts", [])
                logger.warn("workflow_calendar_conflicts_detected", count=len(conflicts))
            completed_steps.append("find_availability")
            artifacts["completed_steps"] = completed_steps
            with factory() as session:
                repo = WorkflowRepository(session)
                repo.update_status(workflow_id, "running", steps=artifacts)
        except Exception as exc:
            try:
                with factory() as session:
                    repo = WorkflowRepository(session)
                    repo.update_status(workflow_id, "failed", steps=artifacts)
            except Exception:
                pass
            err_result = AgentResult(
                agent="workflow",
                status="error",
                summary=f"Workflow failed at step 'find_availability': {exc}. Completed steps: {', '.join(completed_steps)}.",
                error={"code": "api_error", "message": str(exc)},
            )
            return {"workflow_id": workflow_id, "workflow_status": "failed", "artifacts": artifacts, "results": [err_result.model_dump()]}

    # STEP 3 & 4: APPROVAL 1 (Calendar event creation)
    # The interrupt must be invoked deterministically on every replay
    decision = interrupt({
        "type": "approval_request",
        "workflow": "schedule_and_email",
        "step": "create_event",
        "tool": "create_event",
        "args": {
            "title": title,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "attendees": [attendee],
        },
        "summary": f"Create calendar event '{title}' for {attendee} from {start_iso} to {end_iso}",
    })

    if not decision or not decision.get("approved"):
        # Rejecting event approval: NO event created, workflow ends
        reason = decision.get("reason", "user rejected") if decision else "user rejected"
        logger.info("workflow_event_approval_rejected", reason=reason)
        artifacts["rejected_at"] = "create_event"
        artifacts["reason"] = reason
        try:
            with factory() as session:
                repo = WorkflowRepository(session)
                repo.update_status(workflow_id, "rejected", steps=artifacts)
        except Exception:
            pass

        result = AgentResult(
            agent="workflow",
            status="rejected",
            summary=f"Calendar event creation was rejected: {reason}. No calendar event was created and no email was sent.",
            data={"step": "create_event", "rejected": True, "reason": reason},
        )
        return {
            "workflow_id": workflow_id,
            "workflow_status": "rejected",
            "artifacts": artifacts,
            "results": [result.model_dump()],
        }

    # User approved event creation; guard actual creation against duplicate execution
    if "create_event" not in completed_steps:
        edits = decision.get("edits", {})
        final_title = edits.get("title", title)
        final_start = edits.get("start_iso", start_iso)
        final_end = edits.get("end_iso", end_iso)

        try:
            create_res = calendar_server.create_event(
                title=final_title,
                start_iso=final_start,
                end_iso=final_end,
                attendees=[attendee],
                description=req.get("agenda", ""),
            )
            if not create_res.get("ok"):
                err_msg = create_res.get("error", {}).get("message", "Failed to create event")
                raise Exception(err_msg)

            event_obj = create_res.get("data", {}).get("event") or create_res.get("data", {})
            artifacts["event"] = event_obj
            completed_steps.append("create_event")
            artifacts["completed_steps"] = completed_steps

            with factory() as session:
                repo = WorkflowRepository(session)
                repo.update_status(workflow_id, "running", steps=artifacts)
        except Exception as exc:
            try:
                with factory() as session:
                    repo = WorkflowRepository(session)
                    repo.update_status(workflow_id, "failed", steps=artifacts)
            except Exception:
                pass
            err_result = AgentResult(
                agent="workflow",
                status="error",
                summary=f"Workflow failed at step 'create_event': {exc}. Completed steps: {', '.join(completed_steps)}.",
                error={"code": "api_error", "message": str(exc)},
            )
            return {"workflow_id": workflow_id, "workflow_status": "failed", "artifacts": artifacts, "results": [err_result.model_dump()]}

    # STEP 5: Create email draft (agenda + event details)
    if "create_draft" not in completed_steps:
        event = artifacts["event"]
        draft_subject = f"Invitation: {event.get('summary', title)}"
        draft_body = (
            f"Hi,\n\nYou are invited to {event.get('summary', title)}.\n"
            f"Time: {event.get('start', {}).get('dateTime', start_iso)}\n"
            f"Agenda: {req.get('agenda')}\n\nBest regards."
        )
        try:
            draft_res = gmail_server.create_draft(
                to=attendee,
                subject=draft_subject,
                body=draft_body,
            )
            if not draft_res.get("ok"):
                err_msg = draft_res.get("error", {}).get("message", "Failed to create draft")
                raise Exception(err_msg)

            draft_obj = draft_res.get("data", {}).get("draft") or draft_res.get("data", {})
            artifacts["draft"] = draft_obj
            artifacts["draft_details"] = {
                "to": attendee,
                "subject": draft_subject,
                "body": draft_body,
            }
            completed_steps.append("create_draft")
            artifacts["completed_steps"] = completed_steps

            with factory() as session:
                repo = WorkflowRepository(session)
                repo.update_status(workflow_id, "running", steps=artifacts)
        except Exception as exc:
            try:
                with factory() as session:
                    repo = WorkflowRepository(session)
                    repo.update_status(workflow_id, "failed", steps=artifacts)
            except Exception:
                pass
            err_result = AgentResult(
                agent="workflow",
                status="error",
                summary=f"Workflow failed at step 'create_draft': {exc}. Completed steps: {', '.join(completed_steps)}. Event '{title}' remains created.",
                error={"code": "api_error", "message": str(exc)},
            )
            return {"workflow_id": workflow_id, "workflow_status": "failed", "artifacts": artifacts, "results": [err_result.model_dump()]}

    # STEP 6 & 7: APPROVAL 2 (Send email)
    draft_details = artifacts.get("draft_details", {})
    draft_to = draft_details.get("to", attendee)
    draft_subj = draft_details.get("subject", f"Invitation: {title}")
    draft_body = draft_details.get("body", "Meeting invitation.")

    # The interrupt must be invoked deterministically on every replay
    email_decision = interrupt({
        "type": "approval_request",
        "workflow": "schedule_and_email",
        "step": "send_email",
        "tool": "send_email",
        "args": {
            "to": [draft_to],
            "subject": draft_subj,
            "body": draft_body,
        },
        "summary": f"Send meeting invitation email to {draft_to} for event '{title}'",
    })

    if not email_decision or not email_decision.get("approved"):
        # Rejecting email approval: EVENT REMAINS CREATED, no email sent!
        reason = email_decision.get("reason", "user rejected") if email_decision else "user rejected"
        logger.info("workflow_email_approval_rejected", reason=reason)
        artifacts["rejected_at"] = "send_email"
        artifacts["reason"] = reason
        artifacts["event_created"] = True
        try:
            with factory() as session:
                repo = WorkflowRepository(session)
                repo.update_status(workflow_id, "partially_completed", steps=artifacts)
        except Exception:
            pass

        result = AgentResult(
            agent="workflow",
            status="rejected",
            summary=(
                f"Event '{title}' was successfully created in Google Calendar, "
                f"but sending the invitation email was rejected: {reason}. No email was sent."
            ),
            data={
                "step": "send_email",
                "rejected": True,
                "event_created": True,
                "event": artifacts.get("event"),
            },
        )
        return {
            "workflow_id": workflow_id,
            "workflow_status": "partially_completed",
            "artifacts": artifacts,
            "results": [result.model_dump()],
        }

    # User approved email send; guard actual sending against duplicate execution
    if "send_email" not in completed_steps:
        email_edits = email_decision.get("edits", {})
        final_to = email_edits.get("to", [draft_to])
        final_email_subj = email_edits.get("subject", draft_subj)
        final_email_body = email_edits.get("body", draft_body)

        try:
            send_res = gmail_server.send_email(
                to=final_to,
                subject=final_email_subj,
                body=final_email_body,
            )
            if not send_res.get("ok"):
                err_msg = send_res.get("error", {}).get("message", "Failed to send email")
                raise Exception(err_msg)

            sent_obj = send_res.get("data", {}).get("sent") or send_res.get("data", {})
            artifacts["sent_email"] = sent_obj
            completed_steps.append("send_email")
            artifacts["completed_steps"] = completed_steps

            with factory() as session:
                repo = WorkflowRepository(session)
                repo.update_status(workflow_id, "completed", steps=artifacts)
        except Exception as exc:
            try:
                with factory() as session:
                    repo = WorkflowRepository(session)
                    repo.update_status(workflow_id, "failed", steps=artifacts)
            except Exception:
                pass
            err_result = AgentResult(
                agent="workflow",
                status="error",
                summary=f"Workflow failed at step 'send_email': {exc}. Completed steps: {', '.join(completed_steps)}. Event '{title}' remains created.",
                error={"code": "api_error", "message": str(exc)},
            )
            return {"workflow_id": workflow_id, "workflow_status": "failed", "artifacts": artifacts, "results": [err_result.model_dump()]}

    # STEP 8: Summarize happy path completion
    event = artifacts.get("event", {})
    summary = (
        f"Workflow completed: Calendar event '{title}' scheduled for {start_iso}, "
        f"and invitation email sent to {attendee}."
    )
    result = AgentResult(
        agent="workflow",
        status="ok",
        summary=summary,
        data={
            "workflow_id": workflow_id,
            "event": event,
            "sent_email": artifacts.get("sent_email"),
            "completed_steps": completed_steps,
        },
    )

    return {
        "workflow_id": workflow_id,
        "workflow_status": "completed",
        "artifacts": artifacts,
        "results": [result.model_dump()],
    }
