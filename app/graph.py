"""LangGraph assistant StateGraph compilation per SPEC.md section 6, 7, and 8.

Connects:
- router: classifies intent (rag | github | calendar | email | workflow | chitchat)
- specialized agent nodes: rag, github, calendar, email, workflow
- responder: compiles final user-facing response from AgentResult
- checkpointer: SQLite persistence for state and interrupt/approval recovery.
"""
from pathlib import Path
import sqlite3
from typing import Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config import settings
from app.logging import get_logger
from app.state import AgentState, AgentResult
from app.router import route
from app.responder import responder_node
from app.agents.rag_agent import RagAgent
from app.agents.github_agent import GitHubAgent
from app.agents.calendar_agent import CalendarAgent
from app.agents.email_agent import EmailAgent

logger = get_logger("graph")


def extract_latest_query(state: AgentState) -> str:
    """Extract latest user message content from state."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return str(msg.content)
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"])
    return ""


def get_llm() -> Any | None:
    """Return configured Gemini LLM instance if credentials are valid and not in mock mode."""
    if not getattr(settings, "MOCK_MODE", False) and getattr(settings, "GEMINI_API_KEY", ""):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warn("graph_llm_init_failed", error=str(exc))
    return None


def router_node(state: AgentState) -> dict[str, Any]:
    """Router node classifying user intent."""
    query = extract_latest_query(state)
    history = state.get("messages", [])[:-1] if state.get("messages") else []
    workspace = state.get("workspace")
    agent_type = state.get("agent_type")
    llm = get_llm()
    decision = route(query=query, history=history, workspace=workspace, agent_type=agent_type, llm=llm)
    return {
        "intent": decision.intent,
        "workflow": decision.workflow,
    }


def route_intent(state: AgentState) -> str:
    """Conditional routing function based on state intent."""
    intent = state.get("intent", "chitchat")
    if intent in {"rag", "github", "calendar", "email", "workflow"}:
        return intent
    return "chitchat"


def rag_node(state: AgentState) -> dict[str, Any]:
    """Execute RAG Agent."""
    query = extract_latest_query(state)
    llm = get_llm()
    agent = RagAgent(llm=llm)
    result = agent.invoke(query, state=state)
    return {"results": [result.model_dump()]}


def github_node(state: AgentState) -> dict[str, Any]:
    """Execute GitHub Agent."""
    query = extract_latest_query(state)
    agent = GitHubAgent()
    result = agent.invoke(query, state=state)
    artifacts = dict(state.get("artifacts") or {})

    # Preserve target user in artifacts if one was queried
    if result.status == "ok" and result.data.get("user"):
        artifacts["github_target_user"] = result.data["user"]

    return {
        "results": [result.model_dump()],
        "artifacts": artifacts,
    }


def calendar_node(state: AgentState) -> dict[str, Any]:
    """Execute Calendar Agent with approval guards enabled."""
    query = extract_latest_query(state)
    llm = get_llm()
    agent = CalendarAgent(guarded=True, llm=llm)
    result = agent.invoke(query, state=state)
    artifacts = dict(state.get("artifacts") or {})

    # If datetime was clarified, preserve proposed_event in state artifacts
    if result.status == "needs_clarification" and "proposed_event" in result.data:
        artifacts["proposed_event"] = result.data["proposed_event"]
    elif result.status in {"ok", "rejected"} and "proposed_event" in artifacts:
        artifacts.pop("proposed_event", None)

    return {
        "results": [result.model_dump()],
        "artifacts": artifacts,
    }


def email_node(state: AgentState) -> dict[str, Any]:
    """Execute Email Agent with approval guards enabled."""
    query = extract_latest_query(state)
    agent = EmailAgent(guarded=True)
    result = agent.invoke(query)
    return {"results": [result.model_dump()]}


def workflow_node(state: AgentState) -> dict[str, Any]:
    """Execute multi-step workflow via WorkflowManager."""
    from app.workflows.manager import execute_workflow
    wf_type = state.get("workflow") or "schedule_and_email"
    return execute_workflow(wf_type, state)


def get_default_checkpointer() -> SqliteSaver:
    """Create default SQLite checkpointer from settings.CHECKPOINT_DB."""
    db_path = Path(settings.CHECKPOINT_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def create_graph(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """Build and compile the multi-agent StateGraph with checkpointer."""
    workflow = StateGraph(AgentState)

    # Add core nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("github", github_node)
    workflow.add_node("calendar", calendar_node)
    workflow.add_node("email", email_node)
    workflow.add_node("workflow", workflow_node)
    workflow.add_node("responder", responder_node)

    # Start -> Router
    workflow.add_edge(START, "router")

    # Router conditional edge
    workflow.add_conditional_edges(
        "router",
        route_intent,
        {
            "rag": "rag",
            "github": "github",
            "calendar": "calendar",
            "email": "email",
            "workflow": "workflow",
            "chitchat": "responder",
        },
    )

    # Agent nodes connect to responder
    workflow.add_edge("rag", "responder")
    workflow.add_edge("github", "responder")
    workflow.add_edge("calendar", "responder")
    workflow.add_edge("email", "responder")
    workflow.add_edge("workflow", "responder")

    # Responder -> END
    workflow.add_edge("responder", END)

    saver = checkpointer if checkpointer is not None else get_default_checkpointer()
    return workflow.compile(checkpointer=saver)
