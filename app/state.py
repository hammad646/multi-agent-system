"""Core graph state and data models per SPEC.md section 5."""
from typing import TypedDict, Annotated, Literal, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

Intent = Literal["rag", "github", "calendar", "email", "workflow", "chitchat"]


class RouteDecision(BaseModel):
    """Router classification output."""

    intent: Intent
    workflow: Literal["schedule_and_email"] | None = None
    reasoning: str = Field(description="one short sentence")


class PendingAction(BaseModel):
    """Action requiring human approval before execution."""

    id: str
    tool: str
    args: dict[str, Any]
    summary: str
    risk: Literal["low", "medium", "high"]


class AgentResult(BaseModel):
    """Standardized output returned by all worker agents."""

    agent: str
    status: Literal["ok", "needs_clarification", "needs_approval", "rejected", "error"]
    summary: str
    data: dict[str, Any] = {}
    clarification: str | None = None
    error: dict | None = None  # {"code": str, "message": str, "retryable": bool}


class AgentState(TypedDict):
    """Complete LangGraph shared state."""

    messages: Annotated[list, add_messages]
    user_id: str
    request_id: str
    intent: str | None
    workflow: str | None
    workflow_id: str | None
    workflow_status: str | None
    requires_approval: bool
    pending_action: dict | None
    artifacts: dict
    results: list[dict]
    errors: list[str]
    workspace: str | None
    agent_type: str | None
