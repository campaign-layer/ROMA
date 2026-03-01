"""Compatibility solve endpoint for synchronous ROMA callers."""

from __future__ import annotations

import uuid
from typing import Any

import dspy
from fastapi import APIRouter
from loguru import logger

from roma_dspy.api.schemas import SolveRequest, SolveResponse
from roma_dspy.config.manager import ConfigManager

router = APIRouter()


class MaitrixGuidanceSignature(dspy.Signature):
    """Generate compact planning guidance for another agent.

    Return plain text with these exact sections:
    Plan:
    Risks:
    SuggestedAction:

    Keep it concise and action-oriented. Do not answer the task directly.
    """

    request: str = dspy.InputField(description="Task or query that needs guidance")
    context: str = dspy.InputField(description="Relevant prior context")
    actions: str = dspy.InputField(description="Optional tool or action hints")
    guidance: str = dspy.OutputField(description="Short planning guidance")


def _build_lm(llm_config: Any) -> dspy.LM:
    """Build a DSPy LM from ROMA LLM config."""
    lm_kwargs: dict[str, Any] = {
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
        "timeout": llm_config.timeout,
        "num_retries": llm_config.num_retries,
        "cache": llm_config.cache,
    }

    if llm_config.api_key:
        lm_kwargs["api_key"] = llm_config.api_key
    if llm_config.base_url:
        lm_kwargs["base_url"] = llm_config.base_url
    if llm_config.rollout_id is not None:
        lm_kwargs["rollout_id"] = llm_config.rollout_id
    if llm_config.extra_body:
        lm_kwargs["extra_body"] = llm_config.extra_body

    return dspy.LM(llm_config.model, **lm_kwargs)


async def _generate_guidance(solve_request: SolveRequest) -> str:
    """Run a lightweight planner call for mAItrix guidance."""
    config = ConfigManager().load_config(profile=solve_request.config_profile)
    planner_config = config.agents.planner.llm

    predictor = dspy.Predict(MaitrixGuidanceSignature)
    predictor.lm = _build_lm(planner_config)

    request_text = solve_request.goal.strip()
    context_text = solve_request.context.strip() if solve_request.context else "None"
    actions_text = (
        ", ".join(solve_request.actions)
        if solve_request.actions
        else "No tool hints provided"
    )

    prediction = await predictor.acall(
        request=request_text,
        context=context_text,
        actions=actions_text,
    )
    guidance = str(getattr(prediction, "guidance", "")).strip()

    if guidance:
        return guidance

    logger.warning("Planner returned empty guidance; using fallback template")
    return (
        "Plan:\n"
        f"- Clarify the task scope: {request_text}\n"
        "- Break the work into the fewest necessary steps.\n"
        "Risks:\n"
        "- Missing context or tool assumptions may derail execution.\n"
        "SuggestedAction:\n"
        "- Start with the highest-confidence next step and verify the result."
    )


@router.post("/solve", response_model=SolveResponse)
async def solve(solve_request: SolveRequest) -> SolveResponse:
    """Run a synchronous guidance request for mAItrix callers."""
    guidance = await _generate_guidance(solve_request)

    return SolveResponse(
        execution_id=f"direct-{uuid.uuid4()}",
        status="completed",
        output=guidance,
        answer=guidance,
        result={
            "final_answer": guidance,
            "status": "completed",
        },
    )
