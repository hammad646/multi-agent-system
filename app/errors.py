"""Error taxonomy and structured error formatting per SPEC.md section 5."""
from typing import Literal, Any
from pydantic import BaseModel

ErrorCode = Literal[
    "auth_error",
    "permission_error",
    "validation_error",
    "not_found",
    "rate_limited",
    "timeout",
    "api_error",
    "unexpected",
]

VALID_ERROR_CODES: set[str] = {
    "auth_error",
    "permission_error",
    "validation_error",
    "not_found",
    "rate_limited",
    "timeout",
    "api_error",
    "unexpected",
}


class ErrorDetail(BaseModel):
    """Structured error details."""

    code: ErrorCode
    message: str
    retryable: bool = False


class ToolErrorResponse(BaseModel):
    """Standardized tool failure response structure."""

    ok: Literal[False] = False
    error: ErrorDetail


def create_error_response(
    code: ErrorCode, message: str, retryable: bool = False
) -> dict[str, Any]:
    """Create a structured error dictionary matching SPEC.md section 5."""
    if code not in VALID_ERROR_CODES:
        code = "unexpected"
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }


class AssistantException(Exception):
    """Base exception carrying structured error taxonomy attributes."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        retryable: bool = False,
        *args: Any,
    ) -> None:
        super().__init__(message, *args)
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to structured error response dict."""
        return create_error_response(
            code=self.code, message=self.message, retryable=self.retryable
        )
