"""Common utilities, decorators, and error handlers for FastMCP servers.

Strictly logs to stderr/file only. NEVER writes to stdout.
Enforces timeouts, tenacity retries, and structured error taxonomy per SPEC.md section 5.
"""
from functools import wraps
import sys
import time
from typing import Any, Callable, TypeVar
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.errors import create_error_response, ErrorCode
from app.logging import get_logger

logger = get_logger("mcp_servers.common")

F = TypeVar("F", bound=Callable[..., Any])


def mcp_error_handler(server_name: str) -> Callable[[F], F]:
    """Decorator catching exceptions and returning structured error taxonomy dictionary.

    Guarantees no raw exceptions leak into stdio or LLM context.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                # If function returns a dict that already has ok=False, return as is
                if isinstance(result, dict) and result.get("ok") is False:
                    return result
                return {"ok": True, "data": result}
            except TimeoutError as exc:
                logger.error(
                    "mcp_call_timeout",
                    server=server_name,
                    tool=func.__name__,
                    error=str(exc),
                )
                return create_error_response(
                    code="timeout",
                    message=f"Operation timed out in {server_name}.{func.__name__}",
                    retryable=True,
                )
            except PermissionError as exc:
                logger.error(
                    "mcp_call_permission_denied",
                    server=server_name,
                    tool=func.__name__,
                    error=str(exc),
                )
                return create_error_response(
                    code="permission_error",
                    message=str(exc),
                    retryable=False,
                )
            except ValueError as exc:
                logger.error(
                    "mcp_call_validation_error",
                    server=server_name,
                    tool=func.__name__,
                    error=str(exc),
                )
                return create_error_response(
                    code="validation_error",
                    message=str(exc),
                    retryable=False,
                )
            except Exception as exc:
                # Inspect exception type and message for known API errors
                err_str = str(exc).lower()
                code: ErrorCode = "unexpected"
                retryable = False

                if "404" in err_str or "not found" in err_str:
                    code = "not_found"
                elif "401" in err_str or "unauthorized" in err_str or "bad credentials" in err_str:
                    code = "auth_error"
                elif "403" in err_str or "rate limit" in err_str or "429" in err_str:
                    code = "rate_limited"
                    retryable = True
                elif "timeout" in err_str:
                    code = "timeout"
                    retryable = True
                elif "50" in err_str or "server error" in err_str:
                    code = "api_error"
                    retryable = True

                logger.error(
                    "mcp_call_failed",
                    server=server_name,
                    tool=func.__name__,
                    error_code=code,
                    error=str(exc),
                    duration_ms=int((time.perf_counter() - start) * 1000),
                )
                return create_error_response(
                    code=code,
                    message=f"Error in {server_name}.{func.__name__}: {str(exc)}",
                    retryable=retryable,
                )

        return wrapper  # type: ignore

    return decorator


def retry_external_call(
    max_attempts: int = 3,
    min_wait: float = 0.5,
    max_wait: float = 3.0,
) -> Any:
    """Tenacity retry decorator for external network calls honoring exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, min=min_wait, max=max_wait),
        reraise=True,
    )
