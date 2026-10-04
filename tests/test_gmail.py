"""Tests for Gmail FastMCP server and EmailAgent per SPEC.md section 18 and 19.

Covers:
- draft only (never sends)
- MIME structure and headers
- recipient validation and MAX_EMAIL_RECIPIENTS cap
- replying to email with threading headers
- email agent prompt behavior (missing recipient -> needs_clarification)
- trash / delete email
"""
import pytest
from app.config import settings
from app.agents.email_agent import EmailAgent
from app.mcp_servers import gmail_server
from app.mcp_servers.gmail_server import (
    create_mime_message,
    validate_recipients,
    set_gmail_client,
)
from app.testing.fakes import FakeGmailClient


@pytest.fixture(autouse=True)
def setup_fake_gmail():
    """Ensure a clean FakeGmailClient is injected for each test."""
    fake = FakeGmailClient()
    set_gmail_client(fake)
    yield fake
    set_gmail_client(None)


def test_create_draft_never_sends(setup_fake_gmail: FakeGmailClient):
    """Hard Rule 4: Creating a draft must never send an email."""
    fake = setup_fake_gmail

    res = gmail_server.create_draft(
        to="alice@example.com",
        subject="Project Alpha Review",
        body="Here are the notes for the review.",
    )

    assert res["ok"] is True
    assert "draft" in res["data"]
    assert len(fake.drafts) == 1
    # Ensure no emails were sent
    assert len(fake.sent_messages) == 0


def test_email_agent_draft_command_never_sends(setup_fake_gmail: FakeGmailClient):
    """Agent receiving a 'draft' or 'write' command creates draft only and never sends."""
    fake = setup_fake_gmail
    agent = EmailAgent()

    result = agent.invoke("Please draft an email to bob@example.com about Sprint Kickoff saying The sprint starts Monday.")

    assert result.status == "ok"
    assert "draft" in result.data
    assert result.data["to"] == ["bob@example.com"]
    assert result.data["subject"] == "Sprint Kickoff"
    assert len(fake.drafts) == 1
    assert len(fake.sent_messages) == 0


def test_mime_structure_and_headers():
    """Verify MIME EmailMessage generation adheres to stdlib email specs."""
    msg = create_mime_message(
        to=["alice@example.com", "bob@example.com"],
        subject="Meeting Notes",
        body="Hello everyone, attached are the notes.",
        cc=["carol@example.com"],
        bcc=["dave@example.com"],
        in_reply_to="<parent_msg_123@example.com>",
        references="<parent_msg_123@example.com>",
    )

    assert msg["To"] == "alice@example.com, bob@example.com"
    assert msg["Subject"] == "Meeting Notes"
    assert msg["Cc"] == "carol@example.com"
    assert msg["Bcc"] == "dave@example.com"
    assert msg["In-Reply-To"] == "<parent_msg_123@example.com>"
    assert msg["References"] == "<parent_msg_123@example.com>"
    assert "Hello everyone, attached are the notes." in msg.get_content()


def test_recipient_validation():
    """Verify invalid email syntax is rejected with a validation error."""
    # Invalid email formats
    with pytest.raises(ValueError, match="Invalid email address format"):
        validate_recipients("not-an-email")

    with pytest.raises(ValueError, match="Invalid email address format"):
        validate_recipients(["valid@example.com", "invalid-domain@com"])

    with pytest.raises(ValueError, match="At least one 'to' recipient is required"):
        validate_recipients([])


def test_recipient_cap():
    """Verify recipient cap (MAX_EMAIL_RECIPIENTS) is strictly enforced."""
    cap = settings.MAX_EMAIL_RECIPIENTS
    recipients = [f"user{i}@example.com" for i in range(cap + 1)]

    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        validate_recipients(to=recipients)


def test_reply_to_email(setup_fake_gmail: FakeGmailClient):
    """Replying to an email sets threading headers and Re: subject."""
    fake = setup_fake_gmail

    # Populate a message in fake client
    orig_msg_id = "msg_original_001"
    fake.messages[orig_msg_id] = {
        "id": orig_msg_id,
        "from": "manager@example.com",
        "to": "dev@example.com",
        "subject": "Roadmap Q3",
        "body": "Can you review the roadmap?",
        "threadId": "thread_q3",
    }

    res = gmail_server.reply_to_email(
        message_id=orig_msg_id,
        body="Looks great, no blockers on my side.",
    )

    assert res["ok"] is True
    reply_data = res["data"]["reply"]
    assert reply_data["subject"] == "Re: Roadmap Q3"
    assert reply_data["in_reply_to"] == orig_msg_id
    assert reply_data["to"] == "manager@example.com"
    assert len(fake.sent_messages) == 1


def test_delete_email_trash(setup_fake_gmail: FakeGmailClient):
    """Deleting an email moves it to trash."""
    fake = setup_fake_gmail
    msg_id = "msg_to_delete"
    fake.messages[msg_id] = {
        "id": msg_id,
        "to": "test@example.com",
        "subject": "Spam offer",
        "body": "Buy now",
    }

    res = gmail_server.delete_email(message_id=msg_id)
    assert res["ok"] is True
    assert msg_id in fake.trashed_messages


def test_email_agent_missing_recipient():
    """Missing recipient must return status='needs_clarification' per SPEC."""
    agent = EmailAgent()

    result = agent.invoke("Write an email about the conference schedule")
    assert result.status == "needs_clarification"
    assert "recipient" in result.clarification.lower() or "who" in result.clarification.lower()


def test_email_agent_send_email_displays_data(setup_fake_gmail: FakeGmailClient):
    """EmailAgent sending an email displays final To/Subject/Body in data."""
    fake = setup_fake_gmail
    agent = EmailAgent()

    result = agent.invoke("Send an email to partner@example.com about Partnership saying We are excited to collaborate.")
    assert result.status == "ok"
    assert result.data["to"] == ["partner@example.com"]
    assert result.data["subject"] == "Partnership"
    assert result.data["body"] == "We are excited to collaborate."
    assert len(fake.sent_messages) == 1
