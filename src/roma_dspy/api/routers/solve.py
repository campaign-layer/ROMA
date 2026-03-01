"""Compatibility solve endpoint for synchronous ROMA callers."""

import asyncio
import os

from fastapi import APIRouter, HTTPException, Request

from roma_dspy.api.schemas import SolveRequest, SolveResponse

router = APIRouter()


def _extract_final_text(execution) -> str | None:
    final_result = getattr(execution, "final_result", None) or {}
    if isinstance(final_result, dict):
        result = final_result.get("result")
        if result is not None:
            return str(result)
    return None


@router.post("/solve", response_model=SolveResponse)
async def solve(
    request: Request,
    solve_request: SolveRequest,
) -> SolveResponse:
    """
    Run a ROMA solve request with a short synchronous wait window.

    If the execution finishes within the configured wait budget, return the
    final text directly. Otherwise return an execution_id so the caller can
    poll `/api/v1/executions/{id}`.
    """
    app_state = request.app.state.app_state

    if not app_state.execution_service:
        raise HTTPException(
            status_code=503,
            detail="ExecutionService not available (storage may be disabled)",
        )

    execution_id = await app_state.execution_service.start_execution(
        goal=solve_request.goal,
        max_depth=solve_request.max_depth,
        config_profile=solve_request.config_profile,
        config_overrides=solve_request.config_overrides,
        metadata=solve_request.metadata,
    )

    sync_wait_ms = int(os.getenv("ROMA_SYNC_WAIT_MS", "12000"))
    deadline = asyncio.get_running_loop().time() + max(sync_wait_ms, 0) / 1000

    while asyncio.get_running_loop().time() < deadline:
        execution = await app_state.storage.get_execution(execution_id)
        if execution and execution.status in {"completed", "failed", "cancelled", "canceled"}:
            final_text = _extract_final_text(execution)
            result = getattr(execution, "final_result", None) or None
            wrapped_result = {"final_answer": final_text, **result} if final_text and isinstance(result, dict) else result
            return SolveResponse(
                execution_id=execution_id,
                status=execution.status,
                output=final_text,
                answer=final_text,
                result=wrapped_result,
            )

        await asyncio.sleep(0.4)

    execution = await app_state.storage.get_execution(execution_id)
    return SolveResponse(
        execution_id=execution_id,
        status=execution.status if execution else "running",
        output=None,
        answer=None,
        result=None,
    )
