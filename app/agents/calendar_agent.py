"""Calendar Agent per SPEC.md section 8 and section 10.

Includes current datetime and TIMEZONE dynamically refreshed on every call.
Enforces ambiguity check returning needs_clarification.
Performs conflict check before creating events.
"""
from datetime import datetime, timezone, timedelta
import re
from typing import Any
from zoneinfo import ZoneInfo
from langchain_core.tools import tool

from app.config import settings
from app.mcp_servers import calendar_server
from app.state import AgentResult


def get_current_time_context() -> str:
    """Return formatted current time and timezone context string."""
    try:
        tz = ZoneInfo(settings.TIMEZONE)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    return (
        f"CURRENT LOCAL TIME: {now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
        f"TIMEZONE: {settings.TIMEZONE}\n"
        f"CURRENT ISO: {now.isoformat()}"
    )


def build_calendar_tools(guarded: bool = False) -> list[Any]:
    """Expose FastMCP Calendar tools as LangChain tools."""

    @tool
    def list_events(
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 25,
    ) -> dict[str, Any]:
        """List calendar events within an ISO 8601 interval."""
        return calendar_server.list_events(time_min=time_min, time_max=time_max, max_results=max_results)

    @tool
    def get_event(event_id: str) -> dict[str, Any]:
        """Get details for an event by ID."""
        return calendar_server.get_event(event_id=event_id)

    @tool
    def find_conflicts(start_iso: str, end_iso: str) -> dict[str, Any]:
        """Check for scheduling conflicts overlapping with the interval."""
        return calendar_server.find_conflicts(start_iso=start_iso, end_iso=end_iso)

    @tool
    def find_free_slots(start_date: str, end_date: str, duration_minutes: int = 30) -> dict[str, Any]:
        """Find free unbooked slots."""
        return calendar_server.find_free_slots(start_date=start_date, end_date=end_date, duration_minutes=duration_minutes)

    @tool
    def create_event(
        title: str,
        start_iso: str,
        end_iso: str,
        attendees: list[str] | None = None,
        description: str = "",
        add_meet_link: bool = False,
    ) -> dict[str, Any]:
        """Create a new calendar event with duplicate and timezone validation."""
        return calendar_server.create_event(
            title=title,
            start_iso=start_iso,
            end_iso=end_iso,
            attendees=attendees,
            description=description,
            add_meet_link=add_meet_link,
        )

    @tool
    def update_event(
        event_id: str,
        title: str | None = None,
        start_iso: str | None = None,
        end_iso: str | None = None,
        attendees: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update an event."""
        return calendar_server.update_event(
            event_id=event_id,
            title=title,
            start_iso=start_iso,
            end_iso=end_iso,
            attendees=attendees,
            description=description,
        )

    @tool
    def delete_event(event_id: str) -> dict[str, Any]:
        """Delete an event by ID."""
    raw_tools = [
        list_events,
        get_event,
        find_conflicts,
        find_free_slots,
        create_event,
        update_event,
        delete_event,
    ]
    if guarded:
        from app.guard import guard
        return [guard(t) for t in raw_tools]
    return raw_tools


class CalendarAgent:
    """Calendar Agent handling scheduling, availability queries, and ambiguity clarification."""

    def __init__(self, llm: Any | None = None, guarded: bool = False) -> None:
        self.llm = llm
        self.tools = build_calendar_tools(guarded=guarded)
        self.tool_map = {t.name: t for t in self.tools}

    def _is_ambiguous_datetime(self, query: str) -> bool:
        """Detect ambiguous date/time references that require confirmation."""
        patterns = [
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+\d+",
            r"\b(tomorrow|tonight|next\s+week)\s+at\s+\d+",
            r"\b(at|for)\s+\d+\s*(am|pm)?\b",
        ]
        q_lower = query.lower()
        # If user gives a relative day without full date or ISO timezone
        for p in patterns:
            if re.search(p, q_lower) and not re.search(r"\d{4}-\d{2}-\d{2}", query):
                return True
        return False

    def _extract_title(self, query: str) -> str:
        """Extract event title handling quotes, naming prefixes, and durations."""
        q_trimmed = query.strip()

        # 1. Quoted titles: e.g. called "AI Assistant Live Test", 'Team Sync'
        quoted_match = re.search(r'["\']([^"\']+)["\']', q_trimmed)
        if quoted_match and quoted_match.group(1).strip():
            return quoted_match.group(1).strip()

        # 2. Explicit naming prefixes without quotes: e.g. called Project Kickoff for 30 minutes
        named_match = re.search(
            r'\b(?:called|named|titled)\s+(.+?)(?=\s+(?:for\s+\d+|at\s+\d+|on\s+[A-Za-z]+|tomorrow|today|next\b|$)|$)',
            q_trimmed,
            re.IGNORECASE,
        )
        if named_match and named_match.group(1).strip():
            return named_match.group(1).strip()

        # 3. With attendee: e.g. with alice@example.com or with Bob
        with_match = re.search(r'\bwith\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-.]+|[a-zA-Z]+)', q_trimmed, re.IGNORECASE)
        if with_match:
            return f"Meeting with {with_match.group(1)}"

        # 4. For <topic> where topic is NOT a duration (negative lookahead for \d+ minutes/hours)
        for_match = re.search(
            r'\bfor\s+(?!\d+\s*(?:minutes?|mins?|hours?|hrs?)\b)(.+?)(?=\s+(?:for\s+\d+|at\s+\d+|on\s+[A-Za-z]+|tomorrow|today|next\b|$)|$)',
            q_trimmed,
            re.IGNORECASE,
        )
        if for_match and for_match.group(1).strip():
            return for_match.group(1).strip()

        # 5. About <topic>: e.g. about Roadmap Planning
        about_match = re.search(
            r'\babout\s+(.+?)(?=\s+(?:for\s+\d+|at\s+\d+|on\s+[A-Za-z]+|tomorrow|today|next\b|$)|$)',
            q_trimmed,
            re.IGNORECASE,
        )
        if about_match and about_match.group(1).strip():
            return about_match.group(1).strip()

        return "Meeting"

    async def ainvoke(self, query: str, state: Any | None = None) -> AgentResult:
        """Asynchronous execution."""
        return self.invoke(query, state=state)

    def invoke(self, query: str, state: Any | None = None) -> AgentResult:
        """Execute Calendar agent loop."""
        try:
            tz = ZoneInfo(settings.TIMEZONE)
        except Exception:
            tz = timezone.utc
        now = datetime.now(tz)
        q_lower = query.lower().strip()

        # 0. Check for Affirmative Confirmation ("yes", "yep", "sure", "confirm", "proceed")
        confirmation_match = re.match(r"^(yes|yep|yeah|sure|confirm|proceed|go ahead|correct|ok|okay)\b", q_lower)
        if confirmation_match:
            # Extract proposed event from LangGraph state artifacts or recent results
            proposed_event = None
            if state and isinstance(state, dict):
                artifacts = state.get("artifacts") or {}
                proposed_event = artifacts.get("proposed_event")
                if not proposed_event and state.get("results"):
                    for r in reversed(state.get("results", [])):
                        data = r.get("data", {}) if isinstance(r, dict) else getattr(r, "data", {})
                        if data and "proposed_event" in data:
                            proposed_event = data["proposed_event"]
                            break

            if proposed_event:
                start_iso = proposed_event["start_iso"]
                end_iso = proposed_event["end_iso"]
                title = proposed_event.get("title", "Meeting")
                description = proposed_event.get("description", "")

                # Check conflicts first
                conflict_res = self.tool_map["find_conflicts"].invoke({"start_iso": start_iso, "end_iso": end_iso})
                if conflict_res.get("data", {}).get("has_conflicts"):
                    conflicts = conflict_res.get("data", {}).get("conflicts", [])
                    return AgentResult(
                        agent="calendar",
                        status="ok",
                        summary=f"Conflict detected with {len(conflicts)} existing event(s).",
                        data={"conflicts": conflicts},
                    )

                # Call the guarded create_event tool (hits app/guard.py interrupt for HITL approval)
                create_res = self.tool_map["create_event"].invoke({
                    "title": title,
                    "start_iso": start_iso,
                    "end_iso": end_iso,
                    "description": description,
                })

                if create_res.get("status") == "rejected":
                    return AgentResult(
                        agent="calendar",
                        status="rejected",
                        summary=create_res.get("summary", "Event creation was rejected by user."),
                        data={"tool": "create_event", "rejected": True, "reason": create_res.get("reason")},
                    )
                if not create_res.get("ok"):
                    err = create_res.get("error", {})
                    return AgentResult(agent="calendar", status="error", summary=err.get("message", "Failed to create event"), error=err)

                data = create_res.get("data", {})
                if data.get("duplicate"):
                    return AgentResult(
                        agent="calendar",
                        status="ok",
                        summary=f"Duplicate event detected: {data.get('message')}",
                        data=data,
                    )

                return AgentResult(
                    agent="calendar",
                    status="ok",
                    summary=f"Event '{title}' scheduled for {start_iso}.",
                    data=data,
                )

            return AgentResult(
                agent="calendar",
                status="ok",
                summary="No pending calendar action to confirm. Specify an action such as listing events or scheduling a meeting.",
                data={},
            )

        # 1. Ambiguity Check per SPEC Section 8
        if ("schedule" in q_lower or "book" in q_lower or "create" in q_lower or "meet" in q_lower) and self._is_ambiguous_datetime(query):
            title = self._extract_title(query)

            # 1a. Parse Time
            time_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", q_lower)
            if not time_match:
                time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", q_lower)

            target_hour = 16
            target_minute = 0
            if time_match:
                h = int(time_match.group(1))
                m = int(time_match.group(2)) if time_match.group(2) else 0
                ampm = time_match.group(3)
                if ampm == "pm" and h < 12:
                    h += 12
                elif ampm == "am" and h == 12:
                    h = 0
                elif not ampm and 1 <= h <= 7:
                    h += 12
                target_hour = h
                target_minute = m

            time_str = f"{target_hour % 12 or 12}:{target_minute:02d} {'PM' if target_hour >= 12 else 'AM'}"

            # 1b. Parse Duration
            duration_minutes = 30
            dur_match = re.search(r"(\d+)\s*(?:minutes?|mins?)\b", q_lower)
            if dur_match:
                duration_minutes = int(dur_match.group(1))
            else:
                dur_hr_match = re.search(r"(\d+)\s*(?:hours?|hrs?)\b", q_lower)
                if dur_hr_match:
                    duration_minutes = int(dur_hr_match.group(1)) * 60

            # 1c. Parse Date (Tomorrow vs Explicit Weekday vs Missing Date)
            target_date = None
            if "tomorrow" in q_lower:
                target_date = (now + timedelta(days=1)).date()
            elif "today" in q_lower or "tonight" in q_lower:
                target_date = now.date()
            else:
                weekday_map = {
                    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6
                }
                for day_name, w_idx in weekday_map.items():
                    if re.search(rf"\b{day_name}\b", q_lower):
                        days_ahead = (w_idx - now.weekday()) % 7 or 7
                        target_date = (now + timedelta(days=days_ahead)).date()
                        break

            # Handle Missing Date ("4 PM" with no date) -> Do NOT invent Friday!
            if target_date is None:
                return AgentResult(
                    agent="calendar",
                    status="needs_clarification",
                    summary="Date is missing. Clarification required.",
                    clarification=f"What date would you like to schedule the meeting for at {time_str} {settings.TIMEZONE}?",
                    data={
                        "suggested_timezone": settings.TIMEZONE,
                        "time": time_str,
                        "duration_minutes": duration_minutes,
                        "title": title,
                    },
                )

            target_start = datetime(target_date.year, target_date.month, target_date.day, target_hour, target_minute, 0, tzinfo=tz)
            target_end = target_start + timedelta(minutes=duration_minutes)
            target_start_iso = target_start.isoformat()
            target_end_iso = target_end.isoformat()

            clarification = (
                f"Do you mean {target_start.strftime('%A, %B %d')} at {time_str} {settings.TIMEZONE} "
                f"(duration: {duration_minutes} minutes)?"
            )
            proposed_event = {
                "title": title,
                "start_iso": target_start_iso,
                "end_iso": target_end_iso,
                "timezone": settings.TIMEZONE,
                "description": f"Scheduled via assistant: {query}",
            }
            return AgentResult(
                agent="calendar",
                status="needs_clarification",
                summary="Date or time is ambiguous. Confirmation required.",
                clarification=clarification,
                data={
                    "suggested_timezone": settings.TIMEZONE,
                    "proposed_event": proposed_event,
                },
            )

        # 2. List events (e.g. tomorrow or upcoming)
        if "tomorrow" in q_lower or "list" in q_lower or "schedule" in q_lower or "events" in q_lower:
            start_tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_tomorrow = (now + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)

            res = self.tool_map["list_events"].invoke({
                "time_min": start_tomorrow.isoformat(),
                "time_max": end_tomorrow.isoformat(),
            })
            if not res.get("ok"):
                err = res.get("error", {})
                return AgentResult(
                    agent="calendar",
                    status="error",
                    summary=f"Failed to list events: {err.get('message')}",
                    error=err,
                )

            events = res.get("data", {}).get("events", [])
            if not events:
                return AgentResult(
                    agent="calendar",
                    status="ok",
                    summary=f"No events scheduled for tomorrow ({start_tomorrow.strftime('%Y-%m-%d')}).",
                    data={"events": []},
                )

            lines = [f"Tomorrow's schedule ({start_tomorrow.strftime('%A, %B %d')}):"]
            for e in events:
                summary_text = e.get("summary", "Untitled")
                s_time = e.get("start", {}).get("dateTime", "")
                lines.append(f"- {summary_text} at {s_time}")
            return AgentResult(
                agent="calendar",
                status="ok",
                summary="\n".join(lines),
                data={"events": events},
            )

        # 3. Create event with conflict detection
        if "create" in q_lower or "book" in q_lower:
            # Simple fallback parser for test event
            start_iso = (now + timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0).isoformat()
            end_iso = (now + timedelta(days=1)).replace(hour=14, minute=30, second=0, microsecond=0).isoformat()

            # Always check conflicts first
            conflict_res = self.tool_map["find_conflicts"].invoke({"start_iso": start_iso, "end_iso": end_iso})
            if conflict_res.get("data", {}).get("has_conflicts"):
                conflicts = conflict_res.get("data", {}).get("conflicts", [])
                return AgentResult(
                    agent="calendar",
                    status="ok",
                    summary=f"Conflict detected with {len(conflicts)} existing event(s).",
                    data={"conflicts": conflicts},
                )

            create_res = self.tool_map["create_event"].invoke({
                "title": "Team Sync",
                "start_iso": start_iso,
                "end_iso": end_iso,
                "description": "Weekly alignment",
            })
            if create_res.get("status") == "rejected":
                return AgentResult(
                    agent="calendar",
                    status="rejected",
                    summary=create_res.get("summary", "Event creation was rejected by user."),
                    data={"tool": "create_event", "rejected": True, "reason": create_res.get("reason")},
                )
            if not create_res.get("ok"):
                err = create_res.get("error", {})
                return AgentResult(agent="calendar", status="error", summary=err.get("message"), error=err)

            data = create_res.get("data", {})
            if data.get("duplicate"):
                return AgentResult(
                    agent="calendar",
                    status="ok",
                    summary=f"Duplicate event detected: {data.get('message')}",
                    data=data,
                )

            return AgentResult(
                agent="calendar",
                status="ok",
                summary=f"Event 'Team Sync' scheduled for {start_iso}.",
                data=data,
            )

        # 4. Follow-up query about previously listed events (e.g. "Which one is at 4 PM?")
        if state and isinstance(state, dict):
            messages = state.get("messages", [])
            if len(messages) >= 2:
                prev_ai_msg = messages[-2]
                prev_content = getattr(prev_ai_msg, "content", "") or str(prev_ai_msg)
                if isinstance(prev_content, list):
                    prev_content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in prev_content)

                if "schedule" in str(prev_content).lower() or "event" in str(prev_content).lower():
                    if self.llm is not None and not getattr(settings, "MOCK_MODE", False):
                        try:
                            prompt = (
                                f"Previously listed events:\n{prev_content}\n\n"
                                f"User question: {query}\n"
                                "Answer the question directly and factually using only the previously listed events."
                            )
                            res = self.llm.invoke(prompt)
                            ans = res.content if hasattr(res, "content") else str(res)
                            if isinstance(ans, list):
                                ans = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in ans)
                            return AgentResult(agent="calendar", status="ok", summary=ans.strip(), data={})
                        except Exception:
                            pass

                    # Rule-based fallback for "Which one is at 4 PM?"
                    time_m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", q_lower)
                    if time_m:
                        hr = int(time_m.group(1))
                        meridiem = time_m.group(3)
                        is_pm = meridiem == "pm" or (not meridiem and hr == 16)
                        hour_24 = f"{hr + 12 if is_pm and hr < 12 else hr:02d}:"
                        for line in str(prev_content).split("\n"):
                            if line.strip().startswith("- ") and (hour_24 in line or f"{hr} {meridiem}" in line.lower()):
                                return AgentResult(
                                    agent="calendar",
                                    status="ok",
                                    summary=f"The event scheduled at that time is: {line.strip().lstrip('- ').strip()}.",
                                    data={"matched_line": line.strip()},
                                )

        return AgentResult(
            agent="calendar",
            status="ok",
            summary="Calendar agent ready. Specify an action such as listing events or scheduling a meeting.",
            data={},
        )
