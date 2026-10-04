"""Application service layer per SPEC.md section 7 and Hard Rule 12.

ChatService is the single orchestrator interface used by CLI, FastAPI, and tests.
CLI and API contain NO business logic; both call ChatService.send / ChatService.resume.
"""
import re
import uuid
from pathlib import Path
import sqlite3
from typing import Any
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config import settings
from app.logging import get_logger, bind_context
from app.graph import create_graph

logger = get_logger("service")


class ChatService:
    """Service layer executing the assistant graph and managing approval resumes."""

    def __init__(
        self,
        graph: Any | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.graph = graph or create_graph(checkpointer=checkpointer)

    def send(
        self,
        thread_id: str,
        text: str,
        user_id: str = "default_user",
        workspace: str | None = None,
        agent_type: str | None = None,
    ) -> dict[str, Any]:
        """Send a new message to the assistant for a given thread.

        Returns:
            {"status": "completed", "response": str, "thread_id": str, ...} OR
            {"status": "awaiting_approval", "pending_action": dict, "thread_id": str, ...}
        """
        request_id = str(uuid.uuid4())
        bind_context(request_id=request_id)
        logger.info("chat_service_send_initiated", thread_id=thread_id, request_id=request_id, workspace=workspace)

        config = {"configurable": {"thread_id": thread_id}}

        # Preserve or infer workspace and agent_type
        existing_ws = None
        existing_at = None
        try:
            cur_state = self.graph.get_state(config)
            if cur_state and cur_state.values:
                existing_ws = cur_state.values.get("workspace")
                existing_at = cur_state.values.get("agent_type")
        except Exception:
            pass

        active_ws = workspace or existing_ws
        if not active_ws:
            if thread_id.startswith(("doc_", "rag_")):
                active_ws = "documents"
            elif thread_id.startswith(("git_", "github_")):
                active_ws = "github"
            elif thread_id.startswith(("cal_", "calendar_")):
                active_ws = "calendar"
            elif thread_id.startswith(("mail_", "email_")):
                active_ws = "email"
            else:
                active_ws = "chat"

        active_at = agent_type or existing_at or ("rag" if active_ws == "documents" else ("supervisor" if active_ws == "chat" else active_ws))

        input_data = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "request_id": request_id,
            "workspace": active_ws,
            "agent_type": active_at,
        }

        res = self.graph.invoke(input_data, config)
        return self._process_graph_result(thread_id, res, config)

    def resume(
        self,
        thread_id: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Resume execution of an interrupted thread with a human approval decision.

        Args:
            thread_id: Conversation thread identifier.
            decision: Dict specifying approval: {"approved": bool, "reason": str, "edits": dict}

        Returns:
            {"status": "completed", ...} or {"status": "awaiting_approval", ...}
        """
        request_id = str(uuid.uuid4())
        bind_context(request_id=request_id)
        logger.info("chat_service_resume_initiated", thread_id=thread_id, approved=decision.get("approved"))

        config = {"configurable": {"thread_id": thread_id}}
        command = Command(resume=decision)

        res = self.graph.invoke(command, config)
        return self._process_graph_result(thread_id, res, config)

    def _process_graph_result(
        self,
        thread_id: str,
        res: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Examine graph outcome for interrupts vs final message."""
        # Check for interrupt in response or graph state
        graph_state = self.graph.get_state(config)
        interrupts = [i for t in graph_state.tasks for i in t.interrupts] if hasattr(graph_state, "tasks") else []
        intent = graph_state.values.get("intent")
        workflow = graph_state.values.get("workflow")
        artifacts = graph_state.values.get("artifacts", {})
        results = graph_state.values.get("results", [])
        errors = graph_state.values.get("errors", [])

        trace_steps = []
        if intent:
            trace_steps.append({
                "step": "router",
                "label": "Router Node",
                "status": "ok",
                "detail": f"Classified intent: {intent}",
            })
        if workflow:
            trace_steps.append({
                "step": "workflow",
                "label": f"Workflow Node ({workflow})",
                "status": "ok",
                "detail": f"Orchestrating workflow {workflow}",
            })
        for r in results:
            agent_name = r.get("agent", "agent") if isinstance(r, dict) else getattr(r, "agent", "agent")
            status_val = r.get("status", "ok") if isinstance(r, dict) else getattr(r, "status", "ok")
            summary_val = r.get("summary", "") if isinstance(r, dict) else getattr(r, "summary", "")
            trace_steps.append({
                "step": "agent",
                "label": f"{agent_name.capitalize()} Node",
                "status": status_val,
                "detail": summary_val,
            })

        if interrupts or res.get("__interrupt__"):
            pending = interrupts[0].value if interrupts else res["__interrupt__"][0].value
            logger.info("chat_service_interrupted_awaiting_approval", thread_id=thread_id, action=pending)
            trace_steps.append({
                "step": "interrupt",
                "label": "HITL Guard Node",
                "status": "awaiting_approval",
                "detail": f"Action {pending.get('tool', pending.get('step', 'action'))} paused for confirmation",
            })
            return {
                "status": "awaiting_approval",
                "thread_id": thread_id,
                "pending_action": pending,
                "summary": pending.get("summary", ""),
                "intent": intent,
                "workspace": graph_state.values.get("workspace"),
                "agent_type": graph_state.values.get("agent_type"),
                "workflow": workflow,
                "artifacts": artifacts,
                "results": results,
                "errors": errors,
                "trace": trace_steps,
                "model": settings.MODEL,
            }

        # Completed execution
        messages = graph_state.values.get("messages", [])
        last_message = messages[-1].content if messages else "No response generated."
        last_result = results[-1] if results else None
        trace_steps.append({
            "step": "finalize",
            "label": "Response Formatter",
            "status": "ok",
            "detail": "Turn completed successfully",
        })

        logger.info("chat_service_turn_completed", thread_id=thread_id)
        return {
            "status": "completed",
            "thread_id": thread_id,
            "response": last_message,
            "agent_result": last_result,
            "intent": intent,
            "workspace": graph_state.values.get("workspace"),
            "agent_type": graph_state.values.get("agent_type"),
            "workflow": workflow,
            "artifacts": artifacts,
            "results": results,
            "errors": errors,
            "trace": trace_steps,
            "model": settings.MODEL,
        }

    def list_threads(
        self,
        workspace: str | None = None,
        agent_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List persistent conversation threads ordered by recency with optional workspace filtering."""
        thread_ids: list[str] = []

        # 1. Try MemorySaver storage if present (e.g. inside unit tests)
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is not None and hasattr(checkpointer, "storage"):
            # MemorySaver holds thread_id as dict keys
            thread_ids = list(checkpointer.storage.keys())[::-1]

        # 2. Try SQLite connection from checkpointer
        elif checkpointer is not None and hasattr(checkpointer, "conn"):
            try:
                cur = checkpointer.conn.cursor()
                cur.execute(
                    "SELECT thread_id, MAX(checkpoint_id) as latest "
                    "FROM checkpoints GROUP BY thread_id ORDER BY latest DESC LIMIT ?",
                    (limit,),
                )
                thread_ids = [r[0] for r in cur.fetchall()]
            except Exception as exc:
                logger.warn("list_threads_sqlite_query_failed", error=str(exc))

        # 3. Fallback to reading settings.CHECKPOINT_DB file directly
        elif Path(settings.CHECKPOINT_DB).exists():
            try:
                conn = sqlite3.connect(settings.CHECKPOINT_DB)
                cur = conn.cursor()
                cur.execute(
                    "SELECT thread_id, MAX(checkpoint_id) as latest "
                    "FROM checkpoints GROUP BY thread_id ORDER BY latest DESC LIMIT ?",
                    (limit,),
                )
                thread_ids = [r[0] for r in cur.fetchall()]
                conn.close()
            except Exception as exc:
                logger.warn("list_threads_file_sqlite_failed", error=str(exc))

        results = []
        for tid in thread_ids:
            try:
                state = self.graph.get_state({"configurable": {"thread_id": tid}})
                if not state or not state.values:
                    continue
                msgs = state.values.get("messages", [])
                if not msgs:
                    continue

                thread_intent = state.values.get("intent", "general")
                thread_ws = state.values.get("workspace")
                thread_at = state.values.get("agent_type")

                # If workspace/agent_type not saved in state, infer from thread_id or intent
                if not thread_ws:
                    if tid.startswith(("doc_", "rag_")) or thread_intent == "rag":
                        thread_ws = "documents"
                        thread_at = "rag"
                    elif tid.startswith(("git_", "github_")) or thread_intent == "github":
                        thread_ws = "github"
                        thread_at = "github"
                    elif tid.startswith(("cal_", "calendar_")) or thread_intent == "calendar":
                        thread_ws = "calendar"
                        thread_at = "calendar"
                    elif tid.startswith(("mail_", "email_")) or thread_intent == "email":
                        thread_ws = "email"
                        thread_at = "email"
                    else:
                        thread_ws = "chat"
                        thread_at = "supervisor"

                # Filter by workspace if requested
                if workspace:
                    ws_norm = workspace.strip().lower()
                    if ws_norm in ("documents", "rag") and thread_ws not in ("documents", "rag"):
                        continue
                    if ws_norm == "github" and thread_ws != "github":
                        continue
                    if ws_norm == "calendar" and thread_ws != "calendar":
                        continue
                    if ws_norm == "email" and thread_ws != "email":
                        continue
                    if ws_norm in ("chat", "supervisor") and thread_ws not in ("chat", "supervisor"):
                        continue

                # Filter by agent_type if requested
                if agent_type:
                    at_norm = agent_type.strip().lower()
                    if at_norm != (thread_at or "").lower():
                        continue

                # Find first human message for title
                first_human = ""
                for m in msgs:
                    if type(m).__name__ == "HumanMessage" or getattr(m, "type", "") == "human":
                        first_human = str(m.content).strip()
                        break
                title = first_human[:50] if first_human else f"Conversation {tid}"

                # Last message snippet
                last_msg = str(msgs[-1].content).strip() if msgs else ""
                last_snippet = (last_msg[:85] + "...") if len(last_msg) > 85 else last_msg

                # Check if awaiting approval
                interrupts = [i for t in state.tasks for i in t.interrupts] if hasattr(state, "tasks") else []
                status = "awaiting_approval" if interrupts else "completed"

                results.append({
                    "thread_id": tid,
                    "title": title,
                    "last_message": last_snippet,
                    "message_count": len(msgs),
                    "status": status,
                    "intent": thread_intent,
                    "workspace": thread_ws,
                    "agent_type": thread_at,
                })
                if len(results) >= limit:
                    break
            except Exception as exc:
                logger.warn("list_threads_item_failed", thread_id=tid, error=str(exc))

        return results

    def get_thread_history(self, thread_id: str) -> dict[str, Any]:
        """Retrieve full conversation history and state for a specific thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        if not state or not state.values:
            return {
                "thread_id": thread_id,
                "messages": [],
                "status": "completed",
                "pending_action": None,
                "summary": "",
                "intent": None,
                "workflow": None,
            }

        interrupts = [i for t in state.tasks for i in t.interrupts] if hasattr(state, "tasks") else []
        pending = interrupts[0].value if interrupts else None
        status = "awaiting_approval" if pending else "completed"

        messages = []
        for m in state.values.get("messages", []):
            role = "user" if type(m).__name__ == "HumanMessage" or getattr(m, "type", "") == "human" else "assistant"
            messages.append({"role": role, "content": str(m.content)})

        intent = state.values.get("intent")
        workflow = state.values.get("workflow")
        artifacts = state.values.get("artifacts", {})
        results = state.values.get("results", [])
        errors = state.values.get("errors", [])

        trace_steps = []
        if intent:
            trace_steps.append({
                "step": "router",
                "label": "Router Node",
                "status": "ok",
                "detail": f"Classified intent: {intent}",
            })
        if workflow:
            trace_steps.append({
                "step": "workflow",
                "label": f"Workflow Node ({workflow})",
                "status": "ok",
                "detail": f"Orchestrating workflow {workflow}",
            })
        for r in results:
            agent_name = r.get("agent", "agent") if isinstance(r, dict) else getattr(r, "agent", "agent")
            status_val = r.get("status", "ok") if isinstance(r, dict) else getattr(r, "status", "ok")
            summary_val = r.get("summary", "") if isinstance(r, dict) else getattr(r, "summary", "")
            trace_steps.append({
                "step": "agent",
                "label": f"{agent_name.capitalize()} Node",
                "status": status_val,
                "detail": summary_val,
            })
        if pending:
            trace_steps.append({
                "step": "interrupt",
                "label": "HITL Guard Node",
                "status": "awaiting_approval",
                "detail": f"Action {pending.get('tool', pending.get('step', 'action'))} paused for confirmation",
            })
        elif messages:
            trace_steps.append({
                "step": "finalize",
                "label": "Response Formatter",
                "status": "ok",
                "detail": "Turn completed successfully",
            })

        return {
            "thread_id": thread_id,
            "messages": messages,
            "status": status,
            "pending_action": pending,
            "summary": pending.get("summary", "") if pending else "",
            "intent": intent,
            "workspace": state.values.get("workspace"),
            "agent_type": state.values.get("agent_type"),
            "workflow": workflow,
            "artifacts": artifacts,
            "results": results,
            "errors": errors,
            "trace": trace_steps,
            "model": settings.MODEL,
        }

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread from the checkpointer."""
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is not None and hasattr(checkpointer, "storage"):
            if thread_id in checkpointer.storage:
                del checkpointer.storage[thread_id]
                return True
        if Path(settings.CHECKPOINT_DB).exists():
            try:
                conn = sqlite3.connect(settings.CHECKPOINT_DB)
                cur = conn.cursor()
                cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
                conn.commit()
                conn.close()
                return True
            except Exception:
                return False
        return False

    def get_system_status(self) -> dict[str, Any]:
        """Return real operational status of all subsystem connections."""
        from app.db.base import get_session
        from app.db.repositories import DocumentRepository, ScheduledEmailRepository

        doc_count = 0
        sched_count = 0
        try:
            with get_session() as session:
                doc_count = len(DocumentRepository(session).list_documents())
                sched_count = len(ScheduledEmailRepository(session).list_all())
        except Exception:
            pass

        thread_count = len(self.list_threads(limit=100))
        token_path = Path(settings.GOOGLE_TOKEN_PATH)
        has_google_token = token_path.exists()

        return {
            "status": "ok",
            "mock_mode": settings.MOCK_MODE,
            "timezone": settings.TIMEZONE,
            "connections": {
                "gemini": {
                    "status": "connected" if bool(settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY) else "not_configured",
                    "model": settings.MODEL,
                },
                "github": {
                    "status": "connected" if bool(settings.GITHUB_PAT) else "not_configured",
                    "write_allowed": settings.GITHUB_ALLOW_WRITE,
                },
                "calendar": {
                    "status": "connected" if has_google_token or settings.MOCK_MODE else "awaiting_auth",
                    "timezone": settings.TIMEZONE,
                },
                "gmail": {
                    "status": "connected" if has_google_token or settings.MOCK_MODE else "awaiting_auth",
                },
                "rag": {
                    "status": "connected" if bool(settings.PINECONE_API_KEY) or settings.MOCK_MODE else "mock_only",
                    "index": settings.PINECONE_INDEX,
                },
                "checkpointer": {
                    "status": "connected",
                    "db": settings.CHECKPOINT_DB,
                },
            },
            "stats": {
                "documents": doc_count,
                "scheduled_emails": sched_count,
                "recent_threads": thread_count,
            },
        }

    def list_github_repos(self, query: str | None = None, limit: int = 50) -> dict[str, Any]:
        """List or search GitHub repositories."""
        from app.mcp_servers.github_server import search_repositories, get_user
        if query and query.strip():
            q = query.strip()
            # If the query is a simple username or handle without spaces/colons (e.g. 'abdullahnadeem215'),
            # try user:{q} first to fetch all repositories of that user
            if re.match(r"^[a-zA-Z0-9_\-]+$", q):
                user_res = search_repositories(f"user:{q}", limit=limit)
                repos = user_res.get("data", {}).get("repositories", [])
                if repos:
                    return user_res
            return search_repositories(q, limit=limit)
        try:
            u = get_user()
            if u.get("ok") and u.get("data", {}).get("login"):
                login = u["data"]["login"]
                return search_repositories(f"user:{login}", limit=limit)
        except Exception:
            pass
        return search_repositories("stars:>1000", limit=15)

    def get_github_repo_details(self, repo: str) -> dict[str, Any]:
        """Get repository overview, issues, PRs, and recent commits."""
        from app.mcp_servers.github_server import get_repository, list_issues, list_pull_requests, list_commits
        repo_data = get_repository(repo)
        issues = list_issues(repo, limit=10)
        prs = list_pull_requests(repo, limit=10)
        commits = list_commits(repo, limit=10)
        return {
            "repository": repo_data.get("data") if repo_data.get("ok") else None,
            "issues": issues.get("data") if issues.get("ok") else [],
            "pull_requests": prs.get("data") if prs.get("ok") else [],
            "commits": commits.get("data") if commits.get("ok") else [],
            "ok": repo_data.get("ok", False),
            "error": repo_data.get("error"),
        }

    def list_calendar_events(self, time_min: str | None = None, time_max: str | None = None, max_results: int = 50) -> dict[str, Any]:
        """List upcoming calendar events."""
        from app.mcp_servers.calendar_server import list_events
        res = list_events(time_min=time_min, time_max=time_max, max_results=max_results)
        return res.get("data", {"events": [], "count": 0}) if res.get("ok") else {"events": [], "count": 0}

    def find_calendar_free_slots(self, start_date: str, end_date: str, duration_minutes: int = 30) -> dict[str, Any]:
        """Find free calendar slots between start_date and end_date."""
        from app.mcp_servers.calendar_server import find_free_slots
        res = find_free_slots(start_date=start_date, end_date=end_date, duration_minutes=duration_minutes)
        return res.get("data", {"free_slots": [], "count": 0}) if res.get("ok") else {"free_slots": [], "count": 0}

    def list_email_messages(self, query: str | None = None, max_results: int = 20) -> dict[str, Any]:
        """List email messages with extracted headers."""
        from app.mcp_servers.gmail_server import search_emails
        res = search_emails(query=query or "", max_results=max_results)
        if not res.get("ok"):
            return {"messages": [], "count": 0}
        raw_msgs = res.get("data", {}).get("messages", [])
        normalized = []
        for m in raw_msgs:
            if isinstance(m, dict):
                norm = dict(m)
                headers = {h.get("name", "").lower(): h.get("value", "") for h in m.get("payload", {}).get("headers", [])}
                if headers:
                    norm.setdefault("subject", headers.get("subject") or "No Subject")
                    norm.setdefault("from", headers.get("from") or "Unknown")
                    norm.setdefault("sender", headers.get("from") or "Unknown")
                    norm.setdefault("to", headers.get("to") or "Me")
                    norm.setdefault("date", headers.get("date"))
                normalized.append(norm)
            else:
                normalized.append(m)
        return {"messages": normalized, "count": len(normalized)}

    def get_email_message(self, message_id: str) -> dict[str, Any] | None:
        """Get single email message content with extracted headers."""
        from app.mcp_servers.gmail_server import read_email
        res = read_email(message_id)
        if not res.get("ok"):
            return None
        m = res.get("data", {}).get("message", res.get("data", {}))
        if isinstance(m, dict):
            norm = dict(m)
            headers = {h.get("name", "").lower(): h.get("value", "") for h in m.get("payload", {}).get("headers", [])}
            if headers:
                norm.setdefault("subject", headers.get("subject") or "No Subject")
                norm.setdefault("from", headers.get("from") or "Unknown")
                norm.setdefault("sender", headers.get("from") or "Unknown")
                norm.setdefault("to", headers.get("to") or "Me")
                norm.setdefault("date", headers.get("date"))
            if not norm.get("body") and norm.get("snippet"):
                norm["body"] = norm["snippet"]
            return norm
        return m

    def delete_email(self, message_id: str) -> dict[str, Any]:
        """Move email message to trash."""
        from app.mcp_servers.gmail_server import delete_email
        return delete_email(message_id)

