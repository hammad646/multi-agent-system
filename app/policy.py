"""Approval policy table and policy lookup per SPEC.md section 7 and Hard Rule 3.

Policy Table:
- Read anything (events, email, GitHub, RAG): auto
- Create email draft: auto
- Create calendar event: configurable via APPROVE_EVENT_CREATE (default true)
- Update calendar event: approve (always)
- Delete calendar event: approve (always)
- Send email / reply sending: approve (always)
- Schedule email: approve (always)
- Delete email: approve (always)
- GitHub write actions: approve (always)
"""
from app.config import settings

# Actions that strictly ALWAYS require human approval via interrupt().
# Enforcement in code: NO flags to disable these (Hard Rule 3).
STRICT_APPROVAL_ACTIONS = {
    "send_email",
    "send_draft",
    "reply_to_email",
    "schedule_email",
    "delete_email",
    "update_event",
    "delete_event",
    "create_issue",
    "create_pull_request_comment",
    "create_comment",
}


def get_policy(action_name: str) -> str:
    """Return approval policy for a given tool/action name.

    Returns:
        "approve" if approval is required, "auto" otherwise.
    """
    # 1. Strict approval actions (Hard Rule 3)
    if action_name in STRICT_APPROVAL_ACTIONS:
        return "approve"

    # 2. Configurable calendar event creation
    if action_name == "create_event":
        return "approve" if settings.APPROVE_EVENT_CREATE else "auto"

    # 3. Read, search, draft creation, and info queries run automatically
    return "auto"
