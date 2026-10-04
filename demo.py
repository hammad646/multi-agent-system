"""End-to-end interactive/automated demo script per SPEC.md section 18 and 20.

Demonstrates:
1. RAG query with page citations
2. GitHub issue inspection (read-only)
3. Calendar availability and conflict check
4. Human-in-the-loop approval workflow (Schedule + Email)
5. Scheduled email inspection

Usage:
    python demo.py [--live]
"""
import argparse
import io
import sys
import uuid
import fitz

from app.config import settings
from app.db.base import init_db, get_session
from app.db.repositories import DocumentRepository
from app.rag.ingest import ingest_pdf
from app.service import ChatService
from app.scheduler.service import SchedulerService
from app.testing.fakes import FakeGmailClient, FakeCalendarClient
from app.mcp_servers import gmail_server, calendar_server


def setup_sample_document() -> None:
    """Create and ingest a sample PDF into RAG pipeline if not already present."""
    with get_session() as session:
        repo = DocumentRepository(session)
        docs = repo.list_documents()
        if docs:
            print(f"[RAG] Document catalog already contains {len(docs)} document(s).")
            return

    print("[RAG] Ingesting sample technical specification PDF...")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 60),
        "Antigravity Assistant Architecture Overview\n\n"
        "Section 1: Multi-Agent Architecture\n"
        "The system uses LangGraph StateGraph orchestrating specialized sub-agents:\n"
        "- RAG Agent: Pinecone dense vectors + BM25 reciprocal rank fusion (RRF).\n"
        "- GitHub Agent: FastMCP server providing issue and repository queries.\n"
        "- Google Calendar Agent: Conflict checking, duplicate detection, timezone-aware events.\n"
        "- Gmail Agent: MIME email drafting, scheduled delivery worker, recipient limits.\n\n"
        "Section 2: Human in the Loop (HITL) Policy\n"
        "All critical actions require explicit human approval via LangGraph interrupt().\n"
        "Actions requiring approval: calendar event creation, sending emails, GitHub write actions.\n"
    )
    pdf_bytes = doc.write()
    doc.close()

    temp_path = "sample_spec.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    with get_session() as session:
        repo = DocumentRepository(session)
        res = ingest_pdf(temp_path, repo=repo)
        print(f"[RAG] Ingested '{res.filename}': {res.pages} page(s), {res.chunk_count} chunk(s).")


def run_demo(live: bool = False) -> None:
    """Run end-to-end demo flow."""
    print("=" * 70)
    print("      MULTI-AGENT ASSISTANT (RAG + GITHUB + CALENDAR + GMAIL)")
    print("=" * 70)

    if not live:
        print("[Mode] Running in MOCK_MODE with simulated Gmail & Calendar services.")
        fake_gmail = FakeGmailClient()
        fake_cal = FakeCalendarClient()
        gmail_server.set_gmail_client(fake_gmail)
        calendar_server.set_calendar_client(fake_cal)
    else:
        print("[Mode] Running in LIVE mode with real external credentials.")

    init_db()
    setup_sample_document()

    service = ChatService()
    thread_id = f"demo_{uuid.uuid4().hex[:6]}"
    print(f"\n[Session] Active Thread ID: {thread_id}\n")

    # 1. RAG Query
    print("-" * 70)
    print("1. RAG AGENT QUERY")
    print("-" * 70)
    query_rag = "What actions require human approval in the system according to the document?"
    print(f"User: {query_rag}")
    res1 = service.send(thread_id, query_rag)
    print(f"Assistant: {res1.get('response', '')}\n")

    # 2. GitHub Query
    print("-" * 70)
    print("2. GITHUB AGENT QUERY")
    print("-" * 70)
    query_gh = "Show README for octocat/Hello-World"
    print(f"User: {query_gh}")
    res2 = service.send(thread_id, query_gh)
    print(f"Assistant: {res2.get('response', '')}\n")

    # 3. Calendar Check
    print("-" * 70)
    print("3. CALENDAR AVAILABILITY CHECK")
    print("-" * 70)
    query_cal = "Check my calendar events for tomorrow"
    print(f"User: {query_cal}")
    res3 = service.send(thread_id, query_cal)
    print(f"Assistant: {res3.get('response', '')}\n")

    # 4. Multi-Step Workflow with HITL Approvals
    print("-" * 70)
    print("4. CROSS-DOMAIN WORKFLOW (Schedule + Email with Approvals)")
    print("-" * 70)
    wf_query = "Schedule a meeting with client@example.com for Project Launch and email them"
    print(f"User: {wf_query}")
    wf_res1 = service.send(thread_id, wf_query)

    if wf_res1.get("status") == "awaiting_approval":
        pending = wf_res1.get("pending_action", {})
        print("\n>>> HITL APPROVAL 1 REQUIRED:")
        print(f"    Tool/Step: {pending.get('tool')} ({pending.get('step')})")
        print(f"    Summary:   {pending.get('summary')}")
        print("    [Action] Automatically simulating human APPROVE...")

        # Resume step 1
        wf_res2 = service.resume(thread_id, {"approved": True})
        if wf_res2.get("status") == "awaiting_approval":
            pending2 = wf_res2.get("pending_action", {})
            print("\n>>> HITL APPROVAL 2 REQUIRED:")
            print(f"    Tool/Step: {pending2.get('tool')} ({pending2.get('step')})")
            print(f"    Summary:   {pending2.get('summary')}")
            print("    [Action] Automatically simulating human APPROVE...")

            # Resume step 2
            wf_res3 = service.resume(thread_id, {"approved": True})
            print(f"\nAssistant: {wf_res3.get('response')}\n")

    # 5. Scheduled Emails Queue Inspection
    print("-" * 70)
    print("5. SCHEDULED EMAIL QUEUE INSPECTION")
    print("-" * 70)
    scheduler_svc = SchedulerService()
    jobs = scheduler_svc.list_scheduled_emails()
    print(f"Total scheduled emails in persistent queue: {len(jobs)}")
    for j in jobs[:3]:
        print(f" - [{j.status.upper()}] To: {j.recipients} | Time: {j.scheduled_at} | Subject: {j.subject}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE! All capabilities operational.")
    print("=" * 70)
    print("\nNext steps to run manually:")
    print("1. Web UI:        uvicorn app.api.server:app --reload (Open http://localhost:8000)")
    print("2. Terminal CLI:  python -m app.cli")
    print("3. Background:    python -m app.scheduler.worker")
    print("4. Test Suite:    pytest -q\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Assistant Demo")
    parser.add_argument("--live", action="store_true", help="Run in live mode instead of mock mode")
    args = parser.parse_args()
    run_demo(live=args.live)
