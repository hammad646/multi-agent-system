"""Tests for human approval guardrails and process restart persistence per SPEC.md section 18 and 19.

Covers:
- Send never runs without approval (interrupted before execution)
- Rejection reports what did not happen
- Edited arguments are applied before execution
- Interrupted approval survives process restart via SQLite checkpointer
"""
import os
from pathlib import Path
import sqlite3
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.service import ChatService
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


def test_send_never_runs_without_approval(setup_fakes):
    """Hard Rule 3: send_email must trigger approval interrupt; no email sent before approval."""
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())

    # Send command requesting an email dispatch
    res = service.send("thread_approval_1", "Send an email to target@example.com about Deal saying Contract attached")

    # 1. State must pause at approval request
    assert res["status"] == "awaiting_approval"
    pending = res["pending_action"]
    assert pending["tool"] == "send_email"
    assert "target@example.com" in str(pending["args"])

    # 2. Hard Rule 3 verification: zero emails dispatched
    assert len(fake_gmail.sent_messages) == 0


def test_approval_rejection_reports_what_did_not_happen(setup_fakes):
    """Rejecting an approval returns status='rejected' and explicitly states what did NOT happen."""
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "thread_approval_reject"

    # Step 1: Trigger approval
    res1 = service.send(thread_id, "Send an email to cancel@example.com about Notice saying Final notice")
    assert res1["status"] == "awaiting_approval"

    # Step 2: User rejects the approval
    res2 = service.resume(thread_id, {"approved": False, "reason": "User declined"})

    assert res2["status"] == "completed"
    response_text = res2["response"].lower()
    # Must report that the action was cancelled and no email was sent
    assert "no email was sent" in response_text or "cancelled" in response_text or "rejected" in response_text
    assert len(fake_gmail.sent_messages) == 0


def test_approval_with_edited_args(setup_fakes):
    """Approving with edited arguments applies the modified arguments to the executed tool."""
    fake_gmail = setup_fakes["gmail"]
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "thread_approval_edit"

    # Step 1: Trigger approval
    res1 = service.send(thread_id, "Send an email to client@example.com about Proposal saying Original body text")
    assert res1["status"] == "awaiting_approval"

    # Step 2: User approves with edits
    edits = {"body": "Revised and improved body text.", "subject": "Updated Proposal"}
    res2 = service.resume(thread_id, {"approved": True, "edits": edits})

    assert res2["status"] == "completed"
    assert len(fake_gmail.sent_messages) == 1
    sent_msg = fake_gmail.sent_messages[0]
    assert sent_msg["subject"] == "Updated Proposal"
    assert sent_msg["body"] == "Revised and improved body text."


def test_interrupted_approval_resumes_after_restart(tmp_path, setup_fakes):
    """Simulate a process restart: interrupt written to SQLite checkpointer resumes in a new process instance."""
    fake_gmail = setup_fakes["gmail"]
    db_file = tmp_path / "checkpoints_test.db"

    thread_id = "thread_restart_test_123"

    # --- PROCESS 1: Service starts, user sends command, graph interrupts at approval prompt ---
    conn1 = sqlite3.connect(str(db_file), check_same_thread=False)
    saver1 = SqliteSaver(conn1)
    service1 = ChatService(checkpointer=saver1)

    res1 = service1.send(thread_id, "Send an email to ceo@example.com about Quarterly Update saying Revenue up 20%")
    assert res1["status"] == "awaiting_approval"
    assert len(fake_gmail.sent_messages) == 0

    # Simulate killing / exiting Process 1
    conn1.close()
    del service1

    # --- PROCESS 2: Fresh process starts up, connects to same SQLite checkpointer, and resumes ---
    conn2 = sqlite3.connect(str(db_file), check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    service2 = ChatService(checkpointer=saver2)

    # Resume the pending approval
    res2 = service2.resume(thread_id, {"approved": True})

    assert res2["status"] == "completed"
    assert len(fake_gmail.sent_messages) == 1
    sent = fake_gmail.sent_messages[0]
    assert sent["to"] == ["ceo@example.com"]
    assert sent["subject"] == "Quarterly Update"

    conn2.close()
