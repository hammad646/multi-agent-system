"""Guard implementation for human-in-the-loop tool approval per SPEC.md section 7.

Uses LangGraph's interrupt() to pause graph execution and request approval
before performing sensitive actions.
Supports approval, rejection, and parameter edits.
"""
from typing import Any
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt

from app.logging import get_logger
from app.policy import get_policy

logger = get_logger("guard")


def guard(tool: Any, policy: str | None = None) -> Any:
    """Wrap a tool with an interrupt-based approval guard if required by policy.

    Args:
        tool: The LangChain tool to guard.
        policy: Optional override ("approve" or "auto"). If None, resolves from policy table.

    Returns:
        The original tool if policy is "auto", or a guarded StructuredTool if "approve".
    """
    tool_name = getattr(tool, "name", str(tool))
    effective_policy = policy if policy is not None else get_policy(tool_name)

    if effective_policy == "auto":
        return tool

    def _is_in_pregel() -> bool:
        try:
            from langgraph.config import get_config
            conf = get_config().get("configurable", {})
            return "__pregel_scratchpad" in conf
        except Exception:
            return False

    def _sync_run(**kwargs: Any) -> Any:
        if not _is_in_pregel():
            return tool.invoke(kwargs)

        logger.info("guard_approval_requested", tool=tool_name, args=kwargs)
        decision = interrupt({
            "type": "approval_request",
            "tool": tool_name,
            "args": kwargs,
            "summary": f"{tool_name} {kwargs}",
        })

        if not decision or not decision.get("approved"):
            reason = decision.get("reason", "user rejected") if decision else "user rejected"
            logger.info("guard_action_rejected", tool=tool_name, reason=reason)
            return {
                "status": "rejected",
                "tool": tool_name,
                "reason": reason,
                "summary": f"Action '{tool_name}' was rejected by user: {reason}",
            }

        edits = decision.get("edits", {})
        if edits:
            logger.info("guard_action_edits_applied", tool=tool_name, edits=edits)
            kwargs.update(edits)

        logger.info("guard_action_approved_executing", tool=tool_name, args=kwargs)
        return tool.invoke(kwargs)

    async def _async_run(**kwargs: Any) -> Any:
        if not _is_in_pregel():
            if hasattr(tool, "ainvoke"):
                return await tool.ainvoke(kwargs)
            return tool.invoke(kwargs)

        logger.info("guard_approval_requested_async", tool=tool_name, args=kwargs)
        decision = interrupt({
            "type": "approval_request",
            "tool": tool_name,
            "args": kwargs,
            "summary": f"{tool_name} {kwargs}",
        })

        if not decision or not decision.get("approved"):
            reason = decision.get("reason", "user rejected") if decision else "user rejected"
            logger.info("guard_action_rejected", tool=tool_name, reason=reason)
            return {
                "status": "rejected",
                "tool": tool_name,
                "reason": reason,
                "summary": f"Action '{tool_name}' was rejected by user: {reason}",
            }

        edits = decision.get("edits", {})
        if edits:
            logger.info("guard_action_edits_applied", tool=tool_name, edits=edits)
            kwargs.update(edits)

        logger.info("guard_action_approved_executing", tool=tool_name, args=kwargs)
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(kwargs)
        return tool.invoke(kwargs)

    return StructuredTool.from_function(
        func=_sync_run,
        coroutine=_async_run,
        name=tool_name,
        description=getattr(tool, "description", f"Guarded action for {tool_name}"),
        args_schema=getattr(tool, "args_schema", None),
    )
