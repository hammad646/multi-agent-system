"""API key authentication dependency per SPEC.md section 16 and Hard Rule 16."""
from fastapi import Header, HTTPException, status
from app.config import settings


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> str | None:
    """Verify X-API-Key header if API_KEY is configured in settings.

    If settings.API_KEY is not set (empty string), authentication is bypassed.
    If settings.API_KEY is set, requests without matching X-API-Key receive 401 Unauthorized.
    """
    expected_key = getattr(settings, "API_KEY", "")
    if expected_key:
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header",
            )
    return x_api_key
