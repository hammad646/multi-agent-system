"""Interactive Command-Line Interface per SPEC.md section 18.

Uses ChatService exclusively for graph execution and approval resume (Hard Rule 12).
No business logic resides in this module.
"""
import argparse
import json
import sys
import uuid
from typing import Any

from app.config import settings
from app.service import ChatService


def handle_approval_prompt(service: ChatService, thread_id: str, pending_action: dict[str, Any]) -> None:
    """Handle interactive approval prompt from user."""
    tool_name = pending_action.get("tool", "unknown_action")
    args = pending_action.get("args", {})
    summary = pending_action.get("summary", f"{tool_name} {args}")

    print("\n" + "=" * 50)
    print("ACTION REQUIRES HUMAN APPROVAL (Hard Rule 3)")
    print(f"Tool:    {tool_name}")
    print(f"Summary: {summary}")
    print(f"Details: {json.dumps(args, indent=2)}")
    print("=" * 50)
    print("Options: [A]pprove  |  [R]eject  |  [E]dit  |  [Q]uit (test restart)")

    choice = input("Your decision [A/R/E/Q]: ").strip().lower()

    if choice in {"a", "approve", "yes", "y"}:
        res = service.resume(thread_id, {"approved": True})
        if res.get("status") == "awaiting_approval":
            handle_approval_prompt(service, thread_id, res.get("pending_action", {}))
        else:
            print(f"\nAssistant > {res.get('response')}")

    elif choice in {"r", "reject", "no", "n"}:
        reason = input("Optional rejection reason: ").strip() or "Rejected by user in CLI"
        res = service.resume(thread_id, {"approved": False, "reason": reason})
        print(f"\nAssistant > {res.get('response')}")

    elif choice in {"e", "edit"}:
        print("Enter argument edits as JSON (e.g. {\"body\": \"new text\"}):")
        raw_edits = input("Edits JSON: ").strip()
        try:
            edits = json.loads(raw_edits)
            res = service.resume(thread_id, {"approved": True, "edits": edits})
            if res.get("status") == "awaiting_approval":
                handle_approval_prompt(service, thread_id, res.get("pending_action", {}))
            else:
                print(f"\nAssistant > {res.get('response')}")
        except Exception as exc:
            print(f"Invalid JSON edits: {exc}. Resuming as rejected.")
            res = service.resume(thread_id, {"approved": False, "reason": f"Invalid edit JSON: {exc}"})
            print(f"\nAssistant > {res.get('response')}")

    else:
        print(f"Exiting without deciding. You can resume this approval later with thread ID: {thread_id}")


def main() -> None:
    """Run CLI application."""
    parser = argparse.ArgumentParser(description="Multi-Agent Assistant CLI")
    parser.add_argument("query", nargs="?", help="Optional one-shot user query")
    parser.add_argument("--thread-id", default=None, help="Thread ID to resume or continue")
    parser.add_argument("--resume", choices=["approve", "reject"], help="Resume a pending approval on thread-id")
    args = parser.parse_args()

    service = ChatService()
    thread_id = args.thread_id or f"thread_{uuid.uuid4().hex[:8]}"

    if settings.MOCK_MODE:
        print("[MOCK MODE ACTIVE] Running offline with mocked external services.")

    # One-shot approval resume flag
    if args.resume:
        decision = {"approved": bool(args.resume == "approve")}
        res = service.resume(thread_id, decision)
        print(f"Assistant > {res.get('response')}")
        return

    # One-shot query
    if args.query:
        res = service.send(thread_id, args.query)
        if res.get("status") == "awaiting_approval":
            handle_approval_prompt(service, thread_id, res.get("pending_action", {}))
        else:
            print(f"Assistant > {res.get('response')}")
        return

    # Interactive REPL
    print("=" * 60)
    print(" Multi-Agent Assistant CLI (Phase 6 MVP)")
    print(f" Thread ID: {thread_id}")
    print(" Type 'exit' or 'quit' to quit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("Goodbye.")
            break

        res = service.send(thread_id, user_input)
        if res.get("status") == "awaiting_approval":
            handle_approval_prompt(service, thread_id, res.get("pending_action", {}))
        else:
            print(f"\nAssistant > {res.get('response')}")


if __name__ == "__main__":
    main()
