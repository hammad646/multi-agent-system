"""Final response formatting node per SPEC.md section 6, 7, and 8.

Translates AgentResult or rejection notifications into user-facing output messages.
Ensures that rejected actions explicitly explain what did NOT happen.
"""
from typing import Any
from langchain_core.messages import AIMessage

from app.logging import get_logger
from app.state import AgentState, AgentResult

logger = get_logger("responder")


def format_agent_response(result: AgentResult | dict[str, Any] | None) -> str:
    """Format AgentResult into final user-facing response string."""
    if result is None:
        return "Task completed."

    # Handle AgentResult object or dictionary
    if isinstance(result, AgentResult):
        status = result.status
        summary = result.summary
        clarification = result.clarification
        error = result.error
        data = result.data
    elif isinstance(result, dict):
        status = result.get("status", "ok")
        summary = result.get("summary", "")
        clarification = result.get("clarification")
        error = result.get("error")
        data = result.get("data", {})
    else:
        return str(result)

    # 1. Rejected actions (SPEC section 7: must say exactly what did NOT happen)
    if status == "rejected":
        action_name = data.get("tool", "") if data else ""
        if "email" in action_name or "send" in summary.lower():
            return f"Action cancelled: {summary}. No email was sent."
        if "event" in action_name or "calendar" in summary.lower():
            return f"Action cancelled: {summary}. No calendar event was created or modified."
        return f"Action cancelled: {summary}."

    # 2. Needs Clarification
    if status == "needs_clarification":
        return clarification or summary

    # 3. Error
    if status == "error":
        err_msg = error.get("message") if isinstance(error, dict) else str(error)
        err_code = error.get("code") if isinstance(error, dict) else ""
        if err_code == "auth_error" or "auth_error" in (err_msg or ""):
            return (
                f"Authentication error: {err_msg}. "
                "Please run 'python -m app.mcp_servers.google_auth' to authenticate."
            )
        return summary or f"Error: {err_msg}"

    # 4. Success / OK
    return summary


def responder_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node that compiles the final response message."""
    results = state.get("results", [])
    latest_result = results[-1] if results else None

    # Check if there was a chitchat or direct intent
    intent = state.get("intent")
    if intent == "chitchat":
        content = (
            "Hello! I am your AI assistant. I can help you search your ingested PDF documents, "
            "inspect GitHub repositories and issues, manage your Google Calendar events, "
            "and draft or schedule emails."
        )
    elif latest_result:
        content = format_agent_response(latest_result)
    else:
        content = "Request processed."

    response_message = AIMessage(content=content)
    logger.info("responder_generated_response", content_length=len(content))

    return {
        "messages": [response_message],
    }
