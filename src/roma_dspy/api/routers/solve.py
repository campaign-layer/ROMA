"""Compatibility solve endpoint for synchronous ROMA callers."""

from __future__ import annotations

import os
import uuid

import httpx
from fastapi import APIRouter
from loguru import logger

from roma_dspy.api.schemas import SolveRequest, SolveResponse

router = APIRouter()


def _build_fallback_guidance(request_text: str) -> str:
    """Return deterministic guidance when the model call is unavailable."""
    return (
        "Plan:\n"
        f"- Clarify the task scope: {request_text}\n"
        "- Break the work into the fewest necessary steps.\n"
        "Risks:\n"
        "- Missing context or tool assumptions may derail execution.\n"
        "SuggestedAction:\n"
        "- Start with the highest-confidence next step and verify the result."
    )


def _build_prompt(solve_request: SolveRequest) -> str:
    """Create a compact planning prompt for the sidecar model."""
    actions_text = (
        ", ".join(solve_request.actions)
        if solve_request.actions
        else "No tool hints provided"
    )
    context_text = solve_request.context.strip() if solve_request.context else "None"

    return (
        "You are ROMA, a planning sidecar for another agent.\n"
        "Return short guidance only, not a full end-user answer.\n"
        "Use these exact sections and keep the total output concise:\n"
        "Plan:\n"
        "Risks:\n"
        "SuggestedAction:\n\n"
        f"Task:\n{solve_request.goal.strip()}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Action hints:\n{actions_text}\n"
    )


async def _generate_guidance(solve_request: SolveRequest) -> str:
    """Run a lightweight planner call for mAItrix guidance."""
    request_text = solve_request.goal.strip()
    prompt = _build_prompt(solve_request)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("ROMA_GUIDANCE_MODEL", "gemini-2.5-flash")

    if not api_key:
        logger.warning("No Gemini API key configured for ROMA; using fallback guidance")
        return _build_fallback_guidance(request_text)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning(f"Gemini guidance request failed: {exc}")
        return _build_fallback_guidance(request_text)

    candidates = data.get("candidates") or []
    if not candidates:
        logger.warning("Gemini returned no candidates; using fallback guidance")
        return _build_fallback_guidance(request_text)

    parts = candidates[0].get("content", {}).get("parts", [])
    guidance = "\n".join(
        part.get("text", "").strip() for part in parts if part.get("text")
    ).strip()
    if guidance:
        return guidance

    logger.warning("Gemini returned empty guidance; using fallback template")
    return _build_fallback_guidance(request_text)


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
