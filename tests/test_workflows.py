"""Tests for multi-step workflows and WorkflowRepository per SPEC.md section 9, 18, and 19.

Covers:
- Happy path: both approvals granted, event created, email sent, workflow completed
- Reject event: approval 1 rejected -> no event created, workflow ends, summary reports it
- Reject email: approval 1 granted, approval 2 rejected -> event remains created, no email sent
- Mid-step failure: reports completed steps vs failed step without silent partial state
- WorkflowRepository tracking
"""
import sqlite3
from datetime import datetime
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.service import ChatService
from app.db.base import get_session
from app.db.repositories import WorkflowRepository
from app.testing.fakes import FakeGmailClient, FakeCalendarClient
from app.mcp_servers import gmail_server, calendar_server


@pytest.fixture(autouse=True)
def setup_fakes():
    fake_gmail = FakeGmailClient()
    fake_cal = FakeCalendarClient()
    gmail_server.set_gmail_client(fake_gmail)
    calendar_server.set_calendar_client(fake_cal)
    yield {
        "gmail": fake_gmail,
        "calendar": fake_cal,
    }
    gmail_server.set_gmail_client(None)
    calendar_server.set_calendar_client(None)


def test_workflow_happy_path(setup_fakes):
    """Happy path: User approves event creation then approves email sending."""
    fake_cal = setup_fakes["calendar"]
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "wf_happy_path_1"

    # Step 1: Send multi-step workflow query
    query = "Schedule a meeting with client@example.com for Project Kickoff and email them the details"
    res1 = service.send(thread_id, query)

    # First interrupt: Approval for event creation
    assert res1["status"] == "awaiting_approval"
    pending1 = res1["pending_action"]
    assert pending1["step"] == "create_event"
    assert len(fake_cal.events) == 0
    assert len(fake_gmail.sent_messages) == 0

    # User approves event creation
    res2 = service.resume(thread_id, {"approved": True})

    # Second interrupt: Approval for email sending
    assert res2["status"] == "awaiting_approval"
    pending2 = res2["pending_action"]
    assert pending2["step"] == "send_email"
    # Event is created, but email is NOT sent yet!
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 0

    # User approves email sending
    res3 = service.resume(thread_id, {"approved": True})

    # Workflow completed successfully
    assert res3["status"] == "completed"
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 1
    sent_msg = fake_gmail.sent_messages[0]
    assert "client@example.com" in sent_msg["to"]
    assert "kickoff" in res3["response"].lower()

    # Verify WorkflowRepository persistence
    with get_session() as session:
        repo = WorkflowRepository(session)
        runs = repo.list_workflows("schedule_and_email")
        assert len(runs) >= 1
        assert any(r.status == "completed" for r in runs)


def test_workflow_reject_at_event_creates_nothing(setup_fakes):
    """Rejecting the event approval: no event is created, workflow ends, summary says so."""
    fake_cal = setup_fakes["calendar"]
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "wf_reject_event_1"

    # Step 1: Trigger workflow
    query = "Schedule a meeting with vendor@example.com for Contract Review and email them"
    res1 = service.send(thread_id, query)
    assert res1["status"] == "awaiting_approval"
    assert res1["pending_action"]["step"] == "create_event"

    # Step 2: User rejects event creation
    res2 = service.resume(thread_id, {"approved": False, "reason": "Conflict with another meeting"})

    assert res2["status"] == "completed"
    # Crucial acceptance check: NO event created, NO email sent
    assert len(fake_cal.events) == 0
    assert len(fake_gmail.sent_messages) == 0
    response_text = res2["response"].lower()
    assert "no calendar event was created" in response_text or "rejected" in response_text

    # Verify WorkflowRepository shows rejected
    with get_session() as session:
        repo = WorkflowRepository(session)
        runs = repo.list_workflows("schedule_and_email")
        assert any(r.status == "rejected" for r in runs)


def test_workflow_reject_at_email_keeps_event(setup_fakes):
    """Rejecting the email approval: event remains created, no email is sent, summary says so."""
    fake_cal = setup_fakes["calendar"]
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "wf_reject_email_1"

    # Step 1: Trigger workflow
    query = "Schedule a meeting with sam@example.com for Sprint Planning and email him"
    res1 = service.send(thread_id, query)
    assert res1["status"] == "awaiting_approval"

    # Step 2: User approves event creation
    res2 = service.resume(thread_id, {"approved": True})
    assert res2["status"] == "awaiting_approval"
    assert res2["pending_action"]["step"] == "send_email"
    # Event is already created
    assert len(fake_cal.events) == 1

    # Step 3: User rejects email send
    res3 = service.resume(thread_id, {"approved": False, "reason": "I will email him personally"})

    assert res3["status"] == "completed"
    # Crucial acceptance check: Event REMAINS created! Email is NOT sent!
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 0

    response_text = res3["response"].lower()
    assert "created" in response_text
    assert "no email was sent" in response_text

    # Verify WorkflowRepository shows partially_completed
    with get_session() as session:
        repo = WorkflowRepository(session)
        runs = repo.list_workflows("schedule_and_email")
        assert any(r.status == "partially_completed" for r in runs)


def test_workflow_mid_step_failure_reports_partial_state(setup_fakes):
    """Mid-step error stops workflow and reports completed steps vs failed step."""
    fake_cal = setup_fakes["calendar"]
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "wf_failure_test"

    # Step 1: Start workflow
    res1 = service.send(thread_id, "Schedule a meeting with dev@example.com for Architecture Review and email him")
    assert res1["status"] == "awaiting_approval"

    # Make Gmail draft creation fail unexpectedly
    fake_gmail.simulate_failure = Exception("503 Gmail draft service unavailable")

    # Step 2: User approves event creation
    res2 = service.resume(thread_id, {"approved": True})

    assert res2["status"] == "completed"
    # Event was created before failure occurred
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 0

    # Summary must report completed steps and failed step (Rule 263)
    response_text = res2["response"]
    assert "failed" in response_text.lower()
    assert "completed steps" in response_text.lower() or "remains created" in response_text.lower()

    # Verify WorkflowRepository records failure
    with get_session() as session:
        repo = WorkflowRepository(session)
        runs = repo.list_workflows("schedule_and_email")
        assert any(r.status == "failed" for r in runs)


def test_workflow_restart_after_first_approval(tmp_path, setup_fakes):
    """Workflow paused at second approval survives process restart via SQLite checkpointer."""
    fake_cal = setup_fakes["calendar"]
    fake_gmail = setup_fakes["gmail"]
    db_file = tmp_path / "checkpoints_wf_restart.db"
    thread_id = "thread_wf_restart_step2"

    # --- PROCESS 1: Start workflow, approve event, pause at send_email ---
    conn1 = sqlite3.connect(str(db_file), check_same_thread=False)
    saver1 = SqliteSaver(conn1)
    service1 = ChatService(checkpointer=saver1)

    res1 = service1.send(thread_id, "Schedule a meeting with restart@example.com for Q4 Planning and email him")
    assert res1["status"] == "awaiting_approval"
    assert res1["pending_action"]["step"] == "create_event"

    # Approve event creation -> executes create_event and create_draft -> pauses at send_email
    res2 = service1.resume(thread_id, {"approved": True})
    assert res2["status"] == "awaiting_approval"
    assert res2["pending_action"]["step"] == "send_email"
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 0

    # Simulate killing process 1
    conn1.close()
    del service1

    # --- PROCESS 2: Re-open SQLite checkpointer, answer approval 2 ---
    conn2 = sqlite3.connect(str(db_file), check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    service2 = ChatService(checkpointer=saver2)

    res3 = service2.resume(thread_id, {"approved": True})
    assert res3["status"] == "completed"
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 1

    conn2.close()


def test_workflow_resume_does_not_duplicate_event_or_email(setup_fakes):
    """Resuming execution on second approval does not re-create or duplicate event or email."""
    fake_cal = setup_fakes["calendar"]
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "wf_no_duplicate_test"

    # Step 1: Start workflow
    res1 = service.send(thread_id, "Schedule a meeting with unique@example.com for Single Event and email him")
    assert res1["status"] == "awaiting_approval"

    # Step 2: Approve event creation
    res2 = service.resume(thread_id, {"approved": True})
    assert res2["status"] == "awaiting_approval"
    assert len(fake_cal.events) == 1
    initial_event_id = list(fake_cal.events.keys())[0]

    # Step 3: Approve email send (resumes node)
    res3 = service.resume(thread_id, {"approved": True})
    assert res3["status"] == "completed"

    # Verify no duplication occurred
    assert len(fake_cal.events) == 1
    assert list(fake_cal.events.keys())[0] == initial_event_id
    assert len(fake_gmail.sent_messages) == 1


def test_workflow_crash_between_artifact_and_workflow_record(monkeypatch, setup_fakes):
    """If a crash happens after event creation during repository tracking, error is reported cleanly."""
    fake_cal = setup_fakes["calendar"]
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "wf_crash_repo_test"

    res1 = service.send(thread_id, "Schedule a meeting with crash@example.com for Crash Test and email him")
    assert res1["status"] == "awaiting_approval"

    # Monkeypatch WorkflowRepository.update_status to fail right after event is created
    original_update = WorkflowRepository.update_status

    def fail_after_event(self, workflow_id, status, steps=None):
        if steps and "event" in steps:
            raise Exception("Simulated DB connection crash during workflow record update")
        return original_update(self, workflow_id, status, steps=steps)

    monkeypatch.setattr(WorkflowRepository, "update_status", fail_after_event)

    # Approve event creation -> crash occurs during step update
    res2 = service.resume(thread_id, {"approved": True})

    assert res2["status"] == "completed"
    # Event was created in calendar before DB recording crashed
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 0

    response_text = res2["response"]
    assert "error" in response_text.lower() or "failed" in response_text.lower()
    assert "simulated db connection crash" in response_text.lower()

