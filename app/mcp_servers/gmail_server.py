"""Google Gmail FastMCP server per SPEC.md section 10.

Runs over stdio. Strictly writes logs to stderr, NEVER stdout.
Provides tools: search_emails, read_email, create_draft, list_drafts,
send_draft, send_email, reply_to_email, delete_email (trash), list_labels.
MIME via stdlib email; recipient validation; cap MAX_EMAIL_RECIPIENTS.
Never logs email bodies unless LOG_EMAIL_BODIES=true.
"""
import base64
from email.message import EmailMessage
import re
from typing import Any
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.logging import get_logger
from app.mcp_servers.common import mcp_error_handler, retry_external_call

logger = get_logger("mcp_servers.gmail")

mcp = FastMCP("gmail")

_gmail_client: Any = None

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_recipients(
    to: str | list[str],
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Validate recipient email formats and enforce MAX_EMAIL_RECIPIENTS cap."""
    to_list = [to] if isinstance(to, str) else list(to)
    cc_list = list(cc or [])
    bcc_list = list(bcc or [])

    if not to_list:
        raise ValueError("At least one 'to' recipient is required.")

    all_recipients = to_list + cc_list + bcc_list
    max_recipients = settings.MAX_EMAIL_RECIPIENTS

    if len(all_recipients) > max_recipients:
        raise ValueError(
            f"Recipient count {len(all_recipients)} exceeds maximum allowed of {max_recipients}."
        )

    for r in all_recipients:
        cleaned = r.strip()
        if not EMAIL_REGEX.match(cleaned):
            raise ValueError(f"Invalid email address format: '{r}'")

    return to_list, cc_list, bcc_list


def create_mime_message(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    in_reply_to: str | None = None,
    references: str | None = None,
) -> EmailMessage:
    """Create a standard MIME EmailMessage using stdlib email."""
    msg = EmailMessage()
    msg.set_content(body)
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = ", ".join(cc)
    if bcc:
        msg["Bcc"] = ", ".join(bcc)
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    return msg


def encode_mime_message(msg: EmailMessage) -> str:
    """Base64url encode an EmailMessage for Gmail API."""
    raw_bytes = msg.as_bytes()
    return base64.urlsafe_b64encode(raw_bytes).decode("ascii")


def get_gmail_client() -> Any:
    """Get active Gmail client (real Google Gmail service or FakeGmailClient)."""
    global _gmail_client
    if _gmail_client is not None:
        return _gmail_client

    if settings.MOCK_MODE:
        from app.testing.fakes import FakeGmailClient
        _gmail_client = FakeGmailClient()
        return _gmail_client

    # Attempt to load Google OAuth credentials
    from app.mcp_servers.google_auth import get_google_credentials
    creds = get_google_credentials()
    if creds is None:
        from app.testing.fakes import FakeGmailClient
        _gmail_client = FakeGmailClient()
        return _gmail_client

    try:
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=creds)
        _gmail_client = service
        return _gmail_client
    except Exception as exc:
        logger.error("gmail_client_init_failed", error=str(exc))
        from app.testing.fakes import FakeGmailClient
        _gmail_client = FakeGmailClient()
        return _gmail_client


def set_gmail_client(client: Any) -> None:
    """Explicitly inject a client (used in tests)."""
    global _gmail_client
    _gmail_client = client


@mcp.tool()
@mcp_error_handler("gmail")
def search_emails(query: str = "", max_results: int = 10) -> dict[str, Any]:
    """Search messages in Gmail matching an optional query string."""
    client = get_gmail_client()
    if hasattr(client, "search_emails"):
        msgs = client.search_emails(query=query, max_results=max_results)
        return {"messages": msgs, "count": len(msgs)}

    # Real Google Gmail API
    res = client.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    message_ids = res.get("messages", [])
    results = []
    for item in message_ids:
        msg = client.users().messages().get(userId="me", id=item["id"], format="metadata").execute()
        results.append(msg)
    return {"messages": results, "count": len(results)}


@mcp.tool()
@mcp_error_handler("gmail")
def read_email(message_id: str) -> dict[str, Any]:
    """Read a specific email message by its ID."""
    if not message_id:
        raise ValueError("message_id is required")

    client = get_gmail_client()
    if hasattr(client, "read_email"):
        msg = client.read_email(message_id=message_id)
        return {"message": msg}

    msg = client.users().messages().get(userId="me", id=message_id, format="full").execute()
    return {"message": msg}


@mcp.tool()
@mcp_error_handler("gmail")
def create_draft(
    to: str | list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> dict[str, Any]:
    """Create a draft email without sending it."""
    to_list, cc_list, bcc_list = validate_recipients(to, cc, bcc)

    client = get_gmail_client()
    log_data: dict[str, Any] = {
        "to": to_list,
        "subject": subject,
        "cc": cc_list,
        "bcc": bcc_list,
    }
    if settings.LOG_EMAIL_BODIES:
        log_data["body"] = body
    logger.info("create_draft_initiated", **log_data)

    if hasattr(client, "create_draft"):
        draft = client.create_draft(to=to_list, subject=subject, body=body, cc=cc_list, bcc=bcc_list)
        return {"draft": draft, "message": f"Draft created for {to_list}"}

    mime_msg = create_mime_message(to_list, subject, body, cc_list, bcc_list)
    raw = encode_mime_message(mime_msg)
    res = client.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
    return {"draft": res, "message": f"Draft created for {to_list}"}


@mcp.tool()
@mcp_error_handler("gmail")
def list_drafts(max_results: int = 10) -> dict[str, Any]:
    """List drafts in Gmail."""
    client = get_gmail_client()
    if hasattr(client, "list_drafts"):
        drafts = client.list_drafts(max_results=max_results)
        return {"drafts": drafts, "count": len(drafts)}

    res = client.users().drafts().list(userId="me", maxResults=max_results).execute()
    drafts = res.get("drafts", [])
    return {"drafts": drafts, "count": len(drafts)}


@mcp.tool()
@mcp_error_handler("gmail")
def send_draft(draft_id: str) -> dict[str, Any]:
    """Send an existing draft email by draft_id."""
    if not draft_id:
        raise ValueError("draft_id is required")

    client = get_gmail_client()
    logger.info("send_draft_initiated", draft_id=draft_id)

    if hasattr(client, "send_draft"):
        sent = client.send_draft(draft_id=draft_id)
        return {"sent": sent, "message": f"Draft {draft_id} sent successfully."}

    res = client.users().drafts().send(userId="me", body={"id": draft_id}).execute()
    return {"sent": res, "message": f"Draft {draft_id} sent successfully."}


@mcp.tool()
@mcp_error_handler("gmail")
def send_email(
    to: str | list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> dict[str, Any]:
    """Send an email immediately."""
    to_list, cc_list, bcc_list = validate_recipients(to, cc, bcc)

    client = get_gmail_client()
    log_data: dict[str, Any] = {
        "to": to_list,
        "subject": subject,
        "cc": cc_list,
        "bcc": bcc_list,
    }
    if settings.LOG_EMAIL_BODIES:
        log_data["body"] = body
    logger.info("send_email_initiated", **log_data)

    if hasattr(client, "send_email"):
        sent = client.send_email(to=to_list, subject=subject, body=body, cc=cc_list, bcc=bcc_list)
        return {"sent": sent, "message": f"Email sent successfully to {to_list}."}

    mime_msg = create_mime_message(to_list, subject, body, cc_list, bcc_list)
    raw = encode_mime_message(mime_msg)
    res = client.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"sent": res, "message": f"Email sent successfully to {to_list}."}


@mcp.tool()
@mcp_error_handler("gmail")
def reply_to_email(
    message_id: str,
    body: str,
    reply_all: bool = False,
) -> dict[str, Any]:
    """Reply to an existing email message."""
    if not message_id:
        raise ValueError("message_id is required")

    client = get_gmail_client()
    logger.info("reply_to_email_initiated", message_id=message_id, reply_all=reply_all)

    if hasattr(client, "reply_to_email"):
        sent = client.reply_to_email(message_id=message_id, body=body, reply_all=reply_all)
        return {"reply": sent, "message": f"Reply sent to email {message_id}."}

    # Fetch original email to construct threading headers
    orig = client.users().messages().get(userId="me", id=message_id, format="metadata").execute()
    headers = {h["name"].lower(): h["value"] for h in orig.get("payload", {}).get("headers", [])}

    orig_subject = headers.get("subject", "")
    reply_subject = orig_subject if orig_subject.lower().startswith("re:") else f"Re: {orig_subject}"
    reply_to = headers.get("reply-to") or headers.get("from", "")
    thread_id = orig.get("threadId")

    to_list, _, _ = validate_recipients([reply_to])
    mime_msg = create_mime_message(
        to=to_list,
        subject=reply_subject,
        body=body,
        in_reply_to=headers.get("message-id"),
        references=headers.get("message-id"),
    )
    raw = encode_mime_message(mime_msg)
    body_payload: dict[str, Any] = {"raw": raw}
    if thread_id:
        body_payload["threadId"] = thread_id

    res = client.users().messages().send(userId="me", body=body_payload).execute()
    return {"reply": res, "message": f"Reply sent to email {message_id}."}


@mcp.tool()
@mcp_error_handler("gmail")
def delete_email(message_id: str) -> dict[str, Any]:
    """Move an email message to trash."""
    if not message_id:
        raise ValueError("message_id is required")

    client = get_gmail_client()
    logger.info("delete_email_initiated", message_id=message_id)

    if hasattr(client, "delete_email"):
        res = client.delete_email(message_id=message_id)
        return {"result": res, "message": f"Email {message_id} moved to trash."}

    res = client.users().messages().trash(userId="me", id=message_id).execute()
    return {"result": res, "message": f"Email {message_id} moved to trash."}


@mcp.tool()
@mcp_error_handler("gmail")
def list_labels() -> dict[str, Any]:
    """List labels in the user's Gmail mailbox."""
    client = get_gmail_client()
    if hasattr(client, "list_labels"):
        labels = client.list_labels()
        return {"labels": labels, "count": len(labels)}

    res = client.users().labels().list(userId="me").execute()
    labels = res.get("labels", [])
    return {"labels": labels, "count": len(labels)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
