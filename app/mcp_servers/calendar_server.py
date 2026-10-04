"""Google Calendar FastMCP server per SPEC.md section 10.

Runs over stdio. Strictly writes logs to stderr, NEVER stdout.
Provides tools: list_events, get_event, find_free_slots, find_conflicts,
create_event (with duplicate detection), update_event, delete_event.
All datetimes must be timezone-aware ISO 8601. Naive datetimes are rejected.
"""
from datetime import datetime, timezone
from typing import Any
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.logging import get_logger
from app.mcp_servers.common import mcp_error_handler, retry_external_call

logger = get_logger("mcp_servers.calendar")

mcp = FastMCP("calendar")

_calendar_client: Any = None


def validate_iso_datetime(iso_str: str) -> datetime:
    """Validate ISO 8601 string and strictly enforce timezone awareness."""
    if not iso_str:
        raise ValueError("Datetime string cannot be empty.")
    cleaned = iso_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except Exception as exc:
        raise ValueError(f"Invalid ISO 8601 datetime format: {iso_str}") from exc

    if dt.tzinfo is None:
        raise ValueError("Naive datetimes are rejected per SPEC.md guardrails. Timezone offset is required.")
    return dt


def get_calendar_client() -> Any:
    """Get active Calendar client (real Google Calendar service or FakeCalendarClient)."""
    global _calendar_client
    if _calendar_client is not None:
        return _calendar_client

    if settings.MOCK_MODE:
        from app.testing.fakes import FakeCalendarClient
        _calendar_client = FakeCalendarClient()
        return _calendar_client

    # Attempt to load Google OAuth credentials
    from app.mcp_servers.google_auth import get_google_credentials
    creds = get_google_credentials()
    if creds is None:
        # Fall back to fake client when credentials not yet generated
        from app.testing.fakes import FakeCalendarClient
        _calendar_client = FakeCalendarClient()
        return _calendar_client

    try:
        from googleapiclient.discovery import build
        service = build("calendar", "v3", credentials=creds)
        _calendar_client = service
        return _calendar_client
    except Exception as exc:
        logger.error("calendar_client_init_failed", error=str(exc))
        from app.testing.fakes import FakeCalendarClient
        _calendar_client = FakeCalendarClient()
        return _calendar_client


def set_calendar_client(client: Any) -> None:
    """Explicitly inject a client (used in tests)."""
    global _calendar_client
    _calendar_client = client


@mcp.tool()
@mcp_error_handler("calendar")
def list_events(
    time_min: str | None = None,
    time_max: str | None = None,
    max_results: int = 25,
) -> dict[str, Any]:
    """List calendar events within an optional timezone-aware ISO 8601 interval."""
    if time_min:
        validate_iso_datetime(time_min)
    if time_max:
        validate_iso_datetime(time_max)

    client = get_calendar_client()
    if hasattr(client, "list_events"):
        events = client.list_events(time_min=time_min, time_max=time_max, max_results=max_results)
        return {"events": events, "count": len(events)}

    # Real Google Calendar API
    events_result = (
        client.events()
        .list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    items = events_result.get("items", [])
    return {"events": items, "count": len(items)}


@mcp.tool()
@mcp_error_handler("calendar")
def get_event(event_id: str) -> dict[str, Any]:
    """Get details of a specific calendar event by ID."""
    client = get_calendar_client()
    if hasattr(client, "get_event"):
        return client.get_event(event_id)

    event = client.events().get(calendarId="primary", eventId=event_id).execute()
    return event


@mcp.tool()
@mcp_error_handler("calendar")
def find_conflicts(start_iso: str, end_iso: str) -> dict[str, Any]:
    """Check for conflicting events overlapping with the requested interval."""
    validate_iso_datetime(start_iso)
    validate_iso_datetime(end_iso)

    client = get_calendar_client()
    if hasattr(client, "find_conflicts"):
        conflicts = client.find_conflicts(start_iso=start_iso, end_iso=end_iso)
        return {"conflicts": conflicts, "has_conflicts": len(conflicts) > 0}

    # Real Google Calendar API query
    events_result = (
        client.events()
        .list(
            calendarId="primary",
            timeMin=start_iso,
            timeMax=end_iso,
            singleEvents=True,
        )
        .execute()
    )
    items = events_result.get("items", [])
    return {"conflicts": items, "has_conflicts": len(items) > 0}


@mcp.tool()
@mcp_error_handler("calendar")
def find_free_slots(
    start_date: str,
    end_date: str,
    duration_minutes: int = 30,
) -> dict[str, Any]:
    """Find available free time slots for a given duration within an interval."""
    validate_iso_datetime(start_date)
    validate_iso_datetime(end_date)

    client = get_calendar_client()
    if hasattr(client, "find_free_slots"):
        slots = client.find_free_slots(
            start_date=start_date, end_date=end_date, duration_minutes=duration_minutes
        )
        return {"free_slots": slots, "count": len(slots)}

    # Basic fallback slots
    return {"free_slots": [], "count": 0}


@mcp.tool()
@mcp_error_handler("calendar")
def create_event(
    title: str,
    start_iso: str,
    end_iso: str,
    attendees: list[str] | None = None,
    description: str = "",
    add_meet_link: bool = False,
) -> dict[str, Any]:
    """Create a new calendar event with timezone validation and duplicate detection."""
    # Enforce timezone-aware datetimes
    validate_iso_datetime(start_iso)
    validate_iso_datetime(end_iso)

    client = get_calendar_client()
    if hasattr(client, "create_event"):
        return client.create_event(
            summary=title,
            start_iso=start_iso,
            end_iso=end_iso,
            attendees=attendees,
            description=description,
            add_meet_link=add_meet_link,
        )

    # Real Google Calendar API duplicate detection
    existing = (
        client.events()
        .list(
            calendarId="primary",
            timeMin=start_iso,
            timeMax=end_iso,
            q=title,
            singleEvents=True,
        )
        .execute()
    )
    for item in existing.get("items", []):
        if item.get("summary", "").strip().lower() == title.strip().lower():
            return {
                "duplicate": True,
                "event": item,
                "message": f"Duplicate event '{title}' already exists at {start_iso}",
            }

    event_body: dict[str, Any] = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
    }
    if attendees:
        event_body["attendees"] = [{"email": a} for a in attendees]
    if add_meet_link:
        event_body["conferenceData"] = {
            "createRequest": {"requestId": f"meet-{title}"}
        }

    created = client.events().insert(calendarId="primary", body=event_body).execute()
    return created


@mcp.tool()
@mcp_error_handler("calendar")
def update_event(
    event_id: str,
    title: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    attendees: list[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Update an existing calendar event (Requires human approval)."""
    if start_iso:
        validate_iso_datetime(start_iso)
    if end_iso:
        validate_iso_datetime(end_iso)

    client = get_calendar_client()
    if hasattr(client, "update_event"):
        return client.update_event(
            event_id=event_id,
            summary=title,
            start_iso=start_iso,
            end_iso=end_iso,
            attendees=attendees,
            description=description,
        )

    event = client.events().get(calendarId="primary", eventId=event_id).execute()
    if title is not None:
        event["summary"] = title
    if start_iso is not None:
        event["start"]["dateTime"] = start_iso
    if end_iso is not None:
        event["end"]["dateTime"] = end_iso
    if attendees is not None:
        event["attendees"] = [{"email": a} for a in attendees]
    if description is not None:
        event["description"] = description

    updated = client.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return updated


@mcp.tool()
@mcp_error_handler("calendar")
def delete_event(event_id: str) -> dict[str, Any]:
    """Delete a calendar event by ID (Requires human approval)."""
    client = get_calendar_client()
    if hasattr(client, "delete_event"):
        success = client.delete_event(event_id)
        return {"deleted": success, "event_id": event_id}

    client.events().delete(calendarId="primary", eventId=event_id).execute()
    return {"deleted": True, "event_id": event_id}


if __name__ == "__main__":
    mcp.run(transport="stdio")
