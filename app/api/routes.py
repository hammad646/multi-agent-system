"""API route definitions per SPEC.md section 16.

All routes strictly delegate to ChatService, SchedulerService, or DocumentRepository.
CLI, API, and Web UI contain no business logic (Hard Rule 12).
"""
import os
from pathlib import Path
import tempfile
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.db.base import get_session
from app.db.repositories import DocumentRepository, WorkflowRepository
from app.rag.ingest import ingest_pdf
from app.scheduler.service import SchedulerService
from app.service import ChatService
from app.api.auth import verify_api_key

router = APIRouter()

# ChatService dependency injection
_chat_service_instance: ChatService | None = None


def get_chat_service() -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance


def set_chat_service(service: ChatService | None) -> None:
    global _chat_service_instance
    _chat_service_instance = service


# Request Models
class ChatRequest(BaseModel):
    thread_id: str = Field(description="Conversation thread identifier")
    message: str = Field(description="User message text")
    user_id: str = Field(default="default_user", description="User identifier")
    workspace: str | None = Field(default=None, description="Active workspace identifier")
    agent_type: str | None = Field(default=None, description="Specialist agent identifier")


class ApproveRequest(BaseModel):
    thread_id: str = Field(description="Conversation thread identifier")
    edits: dict[str, Any] = Field(default_factory=dict, description="Optional parameter overrides")


class RejectRequest(BaseModel):
    thread_id: str = Field(description="Conversation thread identifier")
    reason: str = Field(default="user rejected", description="Reason for rejection")


# Static Web UI route (GET /)
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_web_ui() -> Any:
    """Serve the static single-file web UI."""
    web_file = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if not web_file.exists():
        return HTMLResponse(
            "<h3>Web UI not found. Ensure app/web/index.html exists.</h3>",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return FileResponse(str(web_file), media_type="text/html")


# Health check
@router.get("/health")
def health_check(auth: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    """Health check endpoint reporting system status."""
    return {
        "status": "ok",
        "mock_mode": settings.MOCK_MODE,
        "timezone": settings.TIMEZONE,
    }


# Chat endpoint
@router.post("/chat")
def chat_endpoint(
    req: ChatRequest,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Send a user message to the assistant and execute the agent graph."""
    try:
        return chat_svc.send(
            thread_id=req.thread_id,
            text=req.message,
            user_id=req.user_id,
            workspace=req.workspace,
            agent_type=req.agent_type,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat execution failed: {exc}",
        ) from exc


# Approval resume endpoint
@router.post("/approve")
def approve_endpoint(
    req: ApproveRequest,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Resume execution with approval decision."""
    try:
        decision = {"approved": True, "edits": req.edits or {}}
        return chat_svc.resume(thread_id=req.thread_id, decision=decision)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Approval resume failed: {exc}",
        ) from exc


# Rejection resume endpoint
@router.post("/reject")
def reject_endpoint(
    req: RejectRequest,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Resume execution with rejection decision."""
    try:
        decision = {"approved": False, "reason": req.reason or "user rejected"}
        return chat_svc.resume(thread_id=req.thread_id, decision=decision)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rejection resume failed: {exc}",
        ) from exc


# Workflow query endpoint
@router.get("/workflows/{workflow_id}")
def get_workflow(
    workflow_id: str,
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get status and step execution progress of a workflow run."""
    with get_session() as session:
        repo = WorkflowRepository(session)
        wf = repo.get(workflow_id)
        if not wf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow run '{workflow_id}' not found",
            )
        return {
            "id": wf.id,
            "workflow_type": wf.workflow_type,
            "status": wf.status,
            "steps": wf.steps,
            "created_at": wf.created_at.isoformat() if wf.created_at else None,
            "updated_at": wf.updated_at.isoformat() if wf.updated_at else None,
        }


# Scheduled email endpoints
@router.get("/scheduled-emails")
def list_scheduled_emails(
    user_id: str | None = None,
    auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """List pending and historical scheduled emails."""
    scheduler_svc = SchedulerService()
    emails = scheduler_svc.list_scheduled_emails(user_id=user_id)
    return [
        {
            "id": e.id,
            "user_id": e.user_id,
            "recipients": e.recipients,
            "cc": e.cc,
            "bcc": e.bcc,
            "subject": e.subject,
            "scheduled_at": e.scheduled_at.isoformat() if e.scheduled_at else None,
            "timezone": e.timezone,
            "status": e.status,
            "attempts": e.attempts,
            "last_error": e.last_error,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in emails
    ]


@router.delete("/scheduled-emails/{email_id}")
def cancel_scheduled_email(
    email_id: str,
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Cancel a pending scheduled email."""
    scheduler_svc = SchedulerService()
    cancelled = scheduler_svc.cancel_scheduled_email(email_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scheduled email '{email_id}' not found or cannot be cancelled",
        )
    return {"status": "cancelled", "id": email_id}


# Document endpoints
@router.get("/documents")
def list_documents(
    auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """List all ingested RAG documents."""
    with get_session() as session:
        repo = DocumentRepository(session)
        docs = repo.list_documents()
        return [
            {
                "document_id": d.document_id,
                "filename": d.filename,
                "title": d.title,
                "pages": d.pages,
                "chunk_count": d.chunk_count,
                "status": d.status,
                "uploaded_at": d.uploaded_at.isoformat() if hasattr(d, "uploaded_at") and d.uploaded_at else None,
            }
            for d in docs
        ]


@router.post("/documents/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Upload and ingest a PDF document into the RAG pipeline."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for ingestion",
        )

    # Save uploaded file to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_f:
        content = await file.read()
        temp_f.write(content)
        temp_path = Path(temp_f.name)

    try:
        with get_session() as session:
            repo = DocumentRepository(session)
            # rename temp file so ingest captures original filename
            dest_path = temp_path.parent / file.filename
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            res = ingest_pdf(dest_path, repo=repo)
            try:
                dest_path.unlink()
            except Exception:
                pass
            return res.model_dump()
    except Exception as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete an ingested document and its text chunks."""
    with get_session() as session:
        repo = DocumentRepository(session)
        deleted = repo.delete_document(document_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found",
            )
        return {"status": "deleted", "document_id": document_id}


# Thread / Recent Chat endpoints (with /chats aliases)
@router.get("/threads")
@router.get("/chats")
def list_threads(
    workspace: str | None = None,
    agent_type: str | None = None,
    limit: int = 50,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """List recent conversation threads from the checkpointer with optional workspace filtering."""
    return chat_svc.list_threads(workspace=workspace, agent_type=agent_type, limit=limit)


@router.get("/threads/{thread_id}")
@router.get("/chats/{thread_id}")
def get_thread(
    thread_id: str,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get message history and current state for a conversation thread."""
    return chat_svc.get_thread_history(thread_id)


@router.delete("/threads/{thread_id}")
@router.delete("/chats/{thread_id}")
def delete_thread(
    thread_id: str,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete a conversation thread from checkpointer storage."""
    deleted = chat_svc.delete_thread(thread_id)
    return {"status": "deleted" if deleted else "not_found", "thread_id": thread_id}


# System Status endpoint
@router.get("/system/status")
def system_status(
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return real connection statuses and statistics without fabricated metrics."""
    return chat_svc.get_system_status()


# Dedicated Page Read Helper Endpoints
@router.get("/api/github/repos")
def github_repos(
    query: str | None = None,
    limit: int = 50,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Search or list GitHub repositories."""
    return chat_svc.list_github_repos(query=query, limit=limit)


@router.get("/api/github/repo")
def github_repo_details(
    repo: str,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get repository metadata, issues, PRs, and recent commits."""
    return chat_svc.get_github_repo_details(repo)


@router.get("/api/calendar/events")
def calendar_events(
    time_min: str | None = None,
    time_max: str | None = None,
    max_results: int = 50,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List calendar events in timezone-aware format."""
    return chat_svc.list_calendar_events(time_min=time_min, time_max=time_max, max_results=max_results)


@router.get("/api/calendar/free-slots")
def calendar_free_slots(
    start_date: str,
    end_date: str,
    duration_minutes: int = 30,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Find free slots on calendar."""
    return chat_svc.find_calendar_free_slots(start_date=start_date, end_date=end_date, duration_minutes=duration_minutes)


@router.get("/api/email/messages")
def email_messages(
    query: str | None = None,
    max_results: int = 20,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List recent email messages."""
    return chat_svc.list_email_messages(query=query, max_results=max_results)


@router.get("/api/email/messages/{message_id}")
def email_message_detail(
    message_id: str,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get full email message content."""
    msg = chat_svc.get_email_message(message_id)
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Email message '{message_id}' not found",
        )
    return msg


@router.delete("/api/email/messages/{message_id}")
def delete_email_message(
    message_id: str,
    chat_svc: ChatService = Depends(get_chat_service),
    auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Move an email to trash."""
    res = chat_svc.delete_email(message_id)
    if not res.get("ok", True):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete email: {res.get('error', 'Unknown error')}",
        )
    return {"status": "deleted", "id": message_id}
