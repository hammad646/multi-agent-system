"""Shared Google OAuth authentication utilities per SPEC.md section 10.

Manages Desktop OAuth flow, token persistence (token.json), auto-refresh,
and least-privilege scopes for Calendar and Gmail.
"""
from pathlib import Path
import sys
from typing import Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.config import settings
from app.errors import create_error_response
from app.logging import get_logger

logger = get_logger("google_auth")

# Least-privilege scopes per SPEC.md section 10
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]

TESTING_MODE_WARNING = (
    "WARNING: While the Google Cloud OAuth app is in 'Testing' mode, refresh tokens "
    "typically expire after ~7 days with 'invalid_grant'. When this occurs, re-run "
    "'python -m app.mcp_servers.google_auth' to re-authenticate."
)


def get_google_credentials() -> Any:
    """Retrieve or refresh valid Google OAuth credentials.

    Returns:
        google.oauth2.credentials.Credentials | FakeGoogleCredentials | None
    """
    token_path = Path(settings.GOOGLE_TOKEN_PATH)
    creds = None

    if token_path.is_file():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:
            logger.error("google_token_load_failed", error=str(exc))
            creds = None

    # If valid credentials exist
    if creds and creds.valid:
        return creds

    # If credentials expired and have a refresh token, attempt auto-refresh
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            return creds
        except Exception as exc:
            err_msg = str(exc)
            logger.error("google_token_refresh_failed", error=err_msg)
            if "invalid_grant" in err_msg.lower():
                logger.error("google_token_expired_7_days", warning=TESTING_MODE_WARNING)
            return None

    return None


def run_auth_flow() -> None:
    """Run one-time interactive Desktop OAuth consent flow via CLI."""
    cred_file = Path(settings.GOOGLE_OAUTH_CREDENTIALS)
    token_path = Path(settings.GOOGLE_TOKEN_PATH)

    if not cred_file.is_file():
        print(f"Error: Google OAuth credentials file not found at: {cred_file}", file=sys.stderr)
        print("Please download your OAuth client credentials from Google Cloud Console as gcp-oauth.keys.json.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Google Desktop OAuth flow using {cred_file}...", file=sys.stderr)
    print(TESTING_MODE_WARNING, file=sys.stderr)

    flow = InstalledAppFlow.from_client_secrets_file(str(cred_file), SCOPES)
    creds = flow.run_local_server(port=0)

    token_path.write_text(creds.to_json(), encoding="utf-8")
    print(f"Success! OAuth token saved to {token_path}", file=sys.stderr)


if __name__ == "__main__":
    run_auth_flow()
