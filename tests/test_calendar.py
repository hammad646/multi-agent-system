"""Tests for Phase 4: Google Calendar MCP Server, Agent, Ambiguity, and Guardrails."""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pytest

from app.config import settings
from app.mcp_servers import calendar_server
from app.mcp_servers.calendar_server import (
    list_events,
    get_event,
    find_conflicts,
    find_free_slots,
    create_event,
    update_event,
    delete_event,
    set_calendar_client,
)
from app.agents.calendar_agent import CalendarAgent
from app.testing.fakes import FakeCalendarClient


@pytest.fixture(autouse=True)
def setup_calendar_mock():
    """Inject fresh FakeCalendarClient for tests."""
    client = FakeCalendarClient()
    set_calendar_client(client)
    yield client
    set_calendar_client(None)


# --- 1. Acceptance Checks ---

def test_list_tomorrows_events():
    """Verify listing tomorrow's events retrieves scheduled events."""
    tz = ZoneInfo(settings.TIMEZONE)
    now = datetime.now(tz)
    tomorrow_start = (now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    tomorrow_end = tomorrow_start + timedelta(minutes=30)

    # Pre-populate tomorrow event
    create_event("Product Review", tomorrow_start.isoformat(), tomorrow_end.isoformat())

    agent = CalendarAgent()
    res = agent.invoke("What are tomorrow's events?")

    assert res.status == "ok"
    assert "Tomorrow's schedule" in res.summary
    assert "Product Review" in res.summary
    assert len(res.data["events"]) >= 1


def test_create_test_event():
    """Verify creating a calendar event with valid timezone-aware datetimes."""
    tz = ZoneInfo(settings.TIMEZONE)
    start_iso = datetime.now(tz).replace(hour=11, minute=0, second=0, microsecond=0).isoformat()
    end_iso = datetime.now(tz).replace(hour=11, minute=30, second=0, microsecond=0).isoformat()

    res = create_event("Architecture Discussion", start_iso, end_iso, description="Review SPEC")
    assert res["ok"] is True
    assert res["data"]["summary"] == "Architecture Discussion"
    assert res["data"]["start"]["dateTime"] == start_iso


def test_duplicate_not_created_twice():
    """Verify creating an identical event (same title + start) returns duplicate detection."""
    tz = ZoneInfo(settings.TIMEZONE)
    start_iso = datetime.now(tz).replace(hour=15, minute=0, second=0, microsecond=0).isoformat()
    end_iso = datetime.now(tz).replace(hour=15, minute=30, second=0, microsecond=0).isoformat()

    # First creation
    res1 = create_event("Sprint Planning", start_iso, end_iso)
    assert res1["ok"] is True
    assert res1["data"]["duplicate"] is False

    # Second creation with identical title and start time
    res2 = create_event("Sprint Planning", start_iso, end_iso)
    assert res2["ok"] is True
    assert res2["data"]["duplicate"] is True
    assert "already exists" in res2["data"]["message"]


def test_friday_at_4_asks_confirmation():
    """Verify ambiguous datetime query returns needs_clarification with precise question."""
    agent = CalendarAgent()
    res = agent.invoke("Schedule a meeting on Friday at 4")

    assert res.status == "needs_clarification"
    assert res.clarification is not None
    assert "4:00 PM" in res.clarification
    assert settings.TIMEZONE in res.clarification
    assert "Do you mean" in res.clarification
    assert "proposed_event" in res.data
    assert res.data["proposed_event"]["start_iso"] is not None
    assert res.data["proposed_event"]["end_iso"] is not None


def test_naive_datetime_rejected():
    """Verify naive datetime string triggers validation_error per Guardrail 7."""
    naive_start = "2026-10-04T10:00:00"  # No timezone offset!
    naive_end = "2026-10-04T10:30:00"

    res = create_event("Invalid Event", naive_start, naive_end)
    assert res["ok"] is False
    assert res["error"]["code"] == "validation_error"
    assert "Naive datetimes are rejected" in res["error"]["message"]


def test_conflict_detection():
    """Verify overlapping time intervals are detected as conflicts."""
    tz = ZoneInfo(settings.TIMEZONE)
    start1 = datetime.now(tz).replace(hour=16, minute=0, second=0, microsecond=0).isoformat()
    end1 = datetime.now(tz).replace(hour=17, minute=0, second=0, microsecond=0).isoformat()

    create_event("Existing Meeting", start1, end1)

    # Overlapping interval 16:30 to 17:30
    start_overlap = datetime.now(tz).replace(hour=16, minute=30, second=0, microsecond=0).isoformat()
    end_overlap = datetime.now(tz).replace(hour=17, minute=30, second=0, microsecond=0).isoformat()

    res = find_conflicts(start_overlap, end_overlap)
    assert res["ok"] is True
    assert res["data"]["has_conflicts"] is True
    assert len(res["data"]["conflicts"]) == 1


# --- 2. Hard Rule 1: No Stdout in MCP Server ---

def test_calendar_server_writes_zero_bytes_to_stdout(capsys):
    """Verify Calendar MCP tools write 0 bytes to sys.stdout."""
    tz = ZoneInfo(settings.TIMEZONE)
    start = datetime.now(tz).isoformat()
    end = (datetime.now(tz) + timedelta(hours=1)).isoformat()

    list_events()
    find_free_slots(start, end)
    find_conflicts(start, end)

    captured = capsys.readouterr()
    assert captured.out == "", f"Hard Rule 1 violated! Stdout captured: {captured.out!r}"


# --- 3. Regression Tests: Calendar Confirmation Loop & HITL Guardrails ---

def test_affirmative_routing_after_clarification():
    """Verify affirmative responses ('yes', 'sure', 'proceed') route to 'calendar' following clarification."""
    from langchain_core.messages import HumanMessage, AIMessage
    from app.router import rule_based_route

    history = [
        HumanMessage(content="Schedule a meeting on Friday at 4"),
        AIMessage(content="Do you mean Friday, October 09 at 4:00 PM Asia/Karachi (duration: 30 minutes)?"),
    ]

    for token in ["yes", "yep", "yeah", "sure", "confirm", "proceed", "go ahead"]:
        d = rule_based_route(token, history=history)
        assert d.intent == "calendar", f"Token {token!r} failed to route to calendar with history"


def test_calendar_confirmation_and_approval_flow(setup_calendar_mock):
    """Verify multi-turn flow: clarification -> 'yes' -> HITL approval -> event creation -> no duplication on repeated 'yes'."""
    from langgraph.checkpoint.memory import MemorySaver
    from app.service import ChatService

    fake_cal = setup_calendar_mock
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "test_calendar_confirm_thread_1"

    # Turn 1: Ambiguous request prompts clarification
    res1 = service.send(thread_id, "Schedule a meeting on Friday at 4")
    assert res1["status"] == "completed"
    assert "Do you mean" in res1["response"]
    assert len(fake_cal.events) == 0

    # Turn 2: Affirmative confirmation triggers guarded create_event and interrupts for HITL approval
    res2 = service.send(thread_id, "yes")
    assert res2["status"] == "awaiting_approval"
    pending = res2["pending_action"]
    assert pending["tool"] == "create_event"
    assert "Meeting" in pending["args"]["title"]
    # Hard Rule 3: No event created in external system prior to explicit approval
    assert len(fake_cal.events) == 0

    # Turn 3: User approves event creation
    res3 = service.resume(thread_id, {"approved": True})
    assert res3["status"] == "completed"
    assert "created" in res3["response"].lower() or "scheduled" in res3["response"].lower()
    assert len(fake_cal.events) == 1

    # Turn 4: Repeated "yes" does NOT duplicate the event or enter an infinite loop
    res4 = service.send(thread_id, "yes")
    assert res4["status"] == "completed"
    assert "no pending calendar action" in res4["response"].lower()
    assert len(fake_cal.events) == 1

    # Turn 5: Normal chitchat functions without recycling stale clarification
    res5 = service.send(thread_id, "hello")
    assert res5["status"] == "completed"
    assert "do you mean" not in res5["response"].lower()
    assert len(fake_cal.events) == 1


def test_calendar_confirmation_rejection_flow(setup_calendar_mock):
    """Verify rejection during HITL approval cancels event creation and reports status."""
    from langgraph.checkpoint.memory import MemorySaver
    from app.service import ChatService

    fake_cal = setup_calendar_mock
    service = ChatService(checkpointer=MemorySaver())
    thread_id = "test_calendar_reject_thread_2"

    # Turn 1: Ambiguous request
    service.send(thread_id, "Schedule a sync on Friday at 4")

    # Turn 2: Confirm
    res2 = service.send(thread_id, "sure")
    assert res2["status"] == "awaiting_approval"
    assert len(fake_cal.events) == 0

    # Turn 3: User rejects approval
    res3 = service.resume(thread_id, {"approved": False, "reason": "No longer needed"})
    assert res3["status"] == "completed"
    assert "cancelled" in res3["response"].lower() or "rejected" in res3["response"].lower()
    assert len(fake_cal.events) == 0


# --- 4. Regression Tests: Dynamic Date Resolution (Tomorrow vs Weekday vs Time-Only vs Contamination) ---

def test_tomorrow_at_4_pm_dynamic_resolution(monkeypatch):
    """Verify 'tomorrow at 4 PM' resolves dynamically to now + 1 day in settings.TIMEZONE, not Friday."""
    tz = ZoneInfo(settings.TIMEZONE)
    # Freeze now to Saturday, Oct 3, 2026 10:00 AM
    fixed_now = datetime(2026, 10, 3, 10, 0, 0, tzinfo=tz)

    from app.agents import calendar_agent
    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(calendar_agent, "datetime", MockDatetime)

    agent = CalendarAgent()
    query = 'Create a calendar event tomorrow at 4 PM called "AI Assistant Live Test" for 30 minutes.'
    res = agent.invoke(query)

    assert res.status == "needs_clarification"
    assert "proposed_event" in res.data
    prop = res.data["proposed_event"]
    assert prop["title"] == "AI Assistant Live Test"
    assert prop["start_iso"] == "2026-10-04T16:00:00+05:00"
    assert prop["end_iso"] == "2026-10-04T16:30:00+05:00"
    assert "Sunday, October 04" in res.clarification
    assert "Friday" not in res.clarification


def test_friday_at_4_pm_resolution(monkeypatch):
    """Verify 'Friday at 4 PM' resolves to next Friday according to SPEC semantics."""
    tz = ZoneInfo(settings.TIMEZONE)
    fixed_now = datetime(2026, 10, 3, 10, 0, 0, tzinfo=tz)  # Saturday Oct 3, 2026

    from app.agents import calendar_agent
    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(calendar_agent, "datetime", MockDatetime)

    agent = CalendarAgent()
    query = "Schedule a meeting on Friday at 4 PM"
    res = agent.invoke(query)

    assert res.status == "needs_clarification"
    assert "proposed_event" in res.data
    prop = res.data["proposed_event"]
    assert prop["start_iso"] == "2026-10-09T16:00:00+05:00"
    assert "Friday, October 09" in res.clarification


def test_time_only_4_pm_asks_for_date_without_inventing_friday():
    """Verify time-only '4 PM' asks for clarification on date and does NOT invent Friday."""
    agent = CalendarAgent()
    query = "Schedule a meeting at 4 PM"
    res = agent.invoke(query)

    assert res.status == "needs_clarification"
    assert "What date would you like to schedule the meeting for at 4:00 PM" in res.clarification
    assert "Friday" not in res.clarification
    # Must NOT invent a proposed_event with a fake Friday date
    assert "proposed_event" not in res.data


def test_stale_friday_date_does_not_contaminate_new_tomorrow_request(monkeypatch):
    """Verify a previously proposed Friday event in state does not contaminate a new 'tomorrow' request."""
    tz = ZoneInfo(settings.TIMEZONE)
    fixed_now = datetime(2026, 10, 3, 10, 0, 0, tzinfo=tz)

    from app.agents import calendar_agent
    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(calendar_agent, "datetime", MockDatetime)

    agent = CalendarAgent()
    stale_state = {
        "artifacts": {
            "proposed_event": {
                "title": "Stale Friday Event",
                "start_iso": "2026-10-09T16:00:00+05:00",
                "end_iso": "2026-10-09T16:30:00+05:00",
                "timezone": settings.TIMEZONE,
            }
        }
    }

    new_query = 'Create a calendar event tomorrow at 4 PM called "AI Assistant Live Test" for 30 minutes.'
    res = agent.invoke(new_query, state=stale_state)

    assert res.status == "needs_clarification"
    assert "proposed_event" in res.data
    prop = res.data["proposed_event"]
    assert prop["title"] == "AI Assistant Live Test"
    assert prop["start_iso"] == "2026-10-04T16:00:00+05:00"
    assert "2026-10-09" not in prop["start_iso"]


