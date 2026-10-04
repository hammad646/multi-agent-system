"""Tests for FastAPI endpoints and static web UI per SPEC.md section 16 and 18.

Covers:
- GET /health
- GET / (serves static app/web/index.html)
- POST /chat -> awaiting_approval -> POST /approve (human-in-the-loop flow)
- POST /chat -> awaiting_approval -> POST /reject
- API_KEY protection (401 without X-API-Key, 200 with valid key)
- GET /scheduled-emails and DELETE /scheduled-emails/{id}
- GET /workflows/{id}
- GET /documents and POST /documents/ingest
"""
from datetime import datetime, timezone
import io
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.api.server import create_app
from app.api.routes import set_chat_service
from app.config import settings
from app.db.base import get_session
from app.db.models import ScheduledEmailModel, utc_now
from app.db.repositories import ScheduledEmailRepository, WorkflowRepository, DocumentRepository
from app.mcp_servers import gmail_server, calendar_server
from app.service import ChatService
from app.testing.fakes import FakeGmailClient, FakeCalendarClient


@pytest.fixture(autouse=True)
def setup_api_fakes():
    """Setup fake clients and in-memory chat service for API testing."""
    fake_gmail = FakeGmailClient()
    fake_cal = FakeCalendarClient()
    gmail_server.set_gmail_client(fake_gmail)
    calendar_server.set_calendar_client(fake_cal)

    # Use MemorySaver checkpointer for clean test isolation
    test_svc = ChatService(checkpointer=MemorySaver())
    set_chat_service(test_svc)

    yield {
        "gmail": fake_gmail,
        "calendar": fake_cal,
        "service": test_svc,
    }

    set_chat_service(None)
    gmail_server.set_gmail_client(None)
    calendar_server.set_calendar_client(None)


def test_api_health():
    """GET /health reports ok status and mock mode."""
    client = TestClient(create_app())
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "mock_mode" in data
    assert "timezone" in data


def test_api_serves_web_ui():
    """GET / returns the static index.html file with isolated dedicated workspaces."""
    client = TestClient(create_app())
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Multi-Agent Assistant" in res.text
    assert "current-thread-id" in res.text

    # Isolated workspaces
    assert 'id="workspace-dashboard"' in res.text
    assert 'id="workspace-chat"' in res.text
    assert 'id="workspace-documents"' in res.text
    assert 'id="workspace-github"' in res.text
    assert 'id="workspace-calendar"' in res.text
    assert 'id="workspace-email"' in res.text
    assert 'id="workspace-scheduled"' in res.text
    assert "switchPage(" in res.text


def test_api_chat_and_approval_flow(setup_api_fakes):
    """Full HTTP cycle: /chat -> awaiting_approval -> /approve -> done."""
    fake_cal = setup_api_fakes["calendar"]
    fake_gmail = setup_api_fakes["gmail"]
    client = TestClient(create_app())
    thread_id = "test_api_thread_1"

    # Step 1: User sends request via POST /chat
    payload = {
        "thread_id": thread_id,
        "message": "Schedule a meeting with client@example.com for Sprint Sync and email them",
    }
    res1 = client.post("/chat", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "awaiting_approval"
    assert data1["pending_action"]["step"] == "create_event"

    # Step 2: Approve event creation via POST /approve
    res2 = client.post("/approve", json={"thread_id": thread_id, "edits": {}})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "awaiting_approval"
    assert data2["pending_action"]["step"] == "send_email"
    assert len(fake_cal.events) == 1

    # Step 3: Approve email send via POST /approve
    res3 = client.post("/approve", json={"thread_id": thread_id, "edits": {}})
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["status"] == "completed"
    assert len(fake_cal.events) == 1
    assert len(fake_gmail.sent_messages) == 1


def test_api_chat_and_rejection_flow(setup_api_fakes):
    """HTTP cycle with rejection: /chat -> awaiting_approval -> /reject."""
    fake_cal = setup_api_fakes["calendar"]
    client = TestClient(create_app())
    thread_id = "test_api_reject_thread"

    # Send request
    res1 = client.post(
        "/chat",
        json={
            "thread_id": thread_id,
            "message": "Schedule a meeting with dev@example.com for Code Review and email him",
        },
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "awaiting_approval"

    # Reject event creation via POST /reject
    res2 = client.post(
        "/reject",
        json={"thread_id": thread_id, "reason": "Not needed anymore"},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "completed"
    # Rejection creates nothing
    assert len(fake_cal.events) == 0


def test_api_approve_with_edits(setup_api_fakes):
    """POST /approve with edited args passes the edits into the execution."""
    fake_cal = setup_api_fakes["calendar"]
    fake_gmail = setup_api_fakes["gmail"]
    client = TestClient(create_app())
    thread_id = "test_api_edits_thread"

    # Send request
    res1 = client.post(
        "/chat",
        json={
            "thread_id": thread_id,
            "message": "Schedule a meeting with lead@example.com for Roadmap and email him",
        },
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "awaiting_approval"

    # User modifies title to 'Q1 Roadmap Final'
    res2 = client.post(
        "/approve",
        json={"thread_id": thread_id, "edits": {"title": "Q1 Roadmap Final"}},
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "awaiting_approval"
    # Verify edited title was used when event was created
    assert len(fake_cal.events) == 1
    event = list(fake_cal.events.values())[0]
    assert event["summary"] == "Q1 Roadmap Final"



def test_api_key_protection(monkeypatch):
    """When API_KEY is set in settings, endpoints reject unauthorized calls with 401."""
    monkeypatch.setattr(settings, "API_KEY", "super-secret-token")
    client = TestClient(create_app())

    # Request without X-API-Key should be rejected
    res_unauth = client.get("/health")
    assert res_unauth.status_code == 401

    # Request with wrong X-API-Key should be rejected
    res_wrong = client.get("/health", headers={"X-API-Key": "bad-token"})
    assert res_wrong.status_code == 401

    # Request with valid X-API-Key should succeed
    res_ok = client.get("/health", headers={"X-API-Key": "super-secret-token"})
    assert res_ok.status_code == 200

    # /chat without key rejected
    res_chat_unauth = client.post(
        "/chat",
        json={"thread_id": "t1", "message": "hello"},
    )
    assert res_chat_unauth.status_code == 401


def test_api_scheduled_emails_crud():
    """GET /scheduled-emails and DELETE /scheduled-emails/{id}."""
    client = TestClient(create_app())

    # Create a scheduled email in DB
    with get_session() as session:
        repo = ScheduledEmailRepository(session)
        target_time = datetime.now(timezone.utc)
        model = ScheduledEmailModel(
            user_id="test_user",
            recipients='["test@example.com"]',
            subject="Test Scheduled Email",
            body="Hello from API test",
            scheduled_at=target_time,
            timezone="UTC",
            status="pending",
            created_at=utc_now(),
        )
        created = repo.create(model)
        email_id = created.id

    # List scheduled emails
    res_list = client.get("/scheduled-emails")
    assert res_list.status_code == 200
    emails = res_list.json()
    assert any(e["id"] == email_id for e in emails)

    # Cancel scheduled email
    res_del = client.delete(f"/scheduled-emails/{email_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "cancelled"

    # Delete non-existent email should return 404
    res_del_404 = client.delete("/scheduled-emails/non_existent_123")
    assert res_del_404.status_code == 404


def test_api_workflows_query():
    """GET /workflows/{id} returns workflow state or 404."""
    import uuid
    client = TestClient(create_app())
    wf_id = f"wf_api_{uuid.uuid4().hex[:8]}"

    # Create workflow in DB
    with get_session() as session:
        repo = WorkflowRepository(session)
        repo.create(workflow_id=wf_id, workflow_type="schedule_and_email", steps={"step": "init"})

    # Query existing workflow
    res = client.get(f"/workflows/{wf_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == wf_id
    assert data["workflow_type"] == "schedule_and_email"

    # Query non-existent workflow
    res_404 = client.get("/workflows/non_existent_wf")
    assert res_404.status_code == 404


def test_api_documents_list_and_ingest(monkeypatch, tmp_path):
    """GET /documents lists ingested docs; POST /documents/ingest accepts PDF."""
    client = TestClient(create_app())

    # Initial document list
    res_docs = client.get("/documents")
    assert res_docs.status_code == 200
    assert isinstance(res_docs.json(), list)

    # Test PDF ingest rejection for non-PDF
    bad_file = ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")
    res_bad = client.post("/documents/ingest", files={"file": bad_file})
    assert res_bad.status_code == 400

    # Test PDF upload
    # Create a minimal valid 1-page PDF using PyMuPDF or pure bytes
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Architecture overview: Antigravity Assistant.")
    pdf_bytes = doc.write()
    doc.close()

    pdf_upload = ("architecture_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")
    res_ingest = client.post("/documents/ingest", files={"file": pdf_upload})
    assert res_ingest.status_code == 200
    ingest_data = res_ingest.json()
    assert ingest_data["filename"] == "architecture_doc.pdf"
    assert ingest_data["pages"] == 1

    # Verify listed in GET /documents
    res_after = client.get("/documents")
    assert res_after.status_code == 200
    assert any(d["filename"] == "architecture_doc.pdf" for d in res_after.json())

    # Test document deletion
    doc_id = [d["document_id"] for d in res_after.json() if d["filename"] == "architecture_doc.pdf"][0]
    res_del = client.delete(f"/documents/{doc_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"


def test_api_threads_endpoints(setup_api_fakes):
    """GET /threads and GET /threads/{thread_id} return persistent conversation history."""
    client = TestClient(create_app())
    thread_id = "test_recent_thread_abc"

    # Send a message
    res = client.post("/chat", json={"thread_id": thread_id, "message": "Hello AI assistant!"})
    assert res.status_code == 200

    # Query threads list
    res_threads = client.get("/threads")
    assert res_threads.status_code == 200
    threads = res_threads.json()
    assert isinstance(threads, list)
    matching = [t for t in threads if t["thread_id"] == thread_id]
    assert len(matching) == 1
    assert "Hello AI assistant" in matching[0]["title"]
    assert matching[0]["message_count"] >= 2
    assert matching[0]["status"] == "completed"

    # Query thread history
    res_hist = client.get(f"/threads/{thread_id}")
    assert res_hist.status_code == 200
    hist = res_hist.json()
    assert hist["thread_id"] == thread_id
    assert len(hist["messages"]) >= 2
    assert hist["messages"][0]["role"] == "user"
    assert hist["messages"][0]["content"] == "Hello AI assistant!"
    assert hist["messages"][1]["role"] == "assistant"

    # Delete thread
    res_del = client.delete(f"/threads/{thread_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"


def test_api_system_status():
    """GET /system/status reports real connection statuses and accurate stats."""
    client = TestClient(create_app())
    res = client.get("/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "connections" in data
    assert "gemini" in data["connections"]
    assert "github" in data["connections"]
    assert "calendar" in data["connections"]
    assert "gmail" in data["connections"]
    assert "rag" in data["connections"]
    assert "checkpointer" in data["connections"]
    assert "stats" in data
    assert "documents" in data["stats"]
    assert "scheduled_emails" in data["stats"]
    assert "recent_threads" in data["stats"]


def test_api_read_helpers(setup_api_fakes):
    """Verify read helper endpoints for GitHub, Calendar, and Email."""
    client = TestClient(create_app())

    # GitHub repos
    res_gh = client.get("/api/github/repos?query=octocat")
    assert res_gh.status_code == 200
    assert "repositories" in res_gh.json().get("data", {}) or "repositories" in res_gh.json()

    # Calendar events
    res_cal = client.get("/api/calendar/events")
    assert res_cal.status_code == 200
    assert "events" in res_cal.json()

    # Email messages
    res_email = client.get("/api/email/messages")
    assert res_email.status_code == 200
    assert "messages" in res_email.json()

