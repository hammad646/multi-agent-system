"""Workflow Manager per SPEC.md section 9.

Maps RouteDecision.workflow to a subgraph or workflow executor.
Adding a new workflow requires one new file plus one registry line in this file.
"""
from typing import Callable, Any
from app.logging import get_logger
from app.state import AgentState, AgentResult
from app.workflows.schedule_and_email import execute_schedule_and_email_workflow

logger = get_logger("workflows.manager")

# Workflow Registry mapping workflow name to executor
WORKFLOW_REGISTRY: dict[str, Callable[[AgentState], dict[str, Any]]] = {
    "schedule_and_email": execute_schedule_and_email_workflow,
}


def register_workflow(name: str, executor: Callable[[AgentState], dict[str, Any]]) -> None:
    """Register a new workflow handler."""
    WORKFLOW_REGISTRY[name] = executor
    logger.info("workflow_registered", workflow=name)


def execute_workflow(workflow_name: str, state: AgentState) -> dict[str, Any]:
    """Execute the workflow corresponding to workflow_name."""
    if workflow_name not in WORKFLOW_REGISTRY:
        err_msg = f"Unknown workflow '{workflow_name}'. Available: {list(WORKFLOW_REGISTRY.keys())}"
        logger.error("workflow_not_found", workflow=workflow_name)
        result = AgentResult(
            agent="workflow",
            status="error",
            summary=err_msg,
            error={"code": "not_found", "message": err_msg},
        )
        return {
            "workflow_status": "error",
            "results": [result.model_dump()],
        }

    handler = WORKFLOW_REGISTRY[workflow_name]
    logger.info("executing_workflow", workflow=workflow_name)
    return handler(state)
