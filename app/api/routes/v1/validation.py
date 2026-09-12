"""
GET /api/v1/validation/summary — validation/evaluation endpoint.

IMPORTANT: This endpoint runs a CONTROLLED validation — it is NOT live inference.
It runs a fresh synthetic scenario with hidden ground truth and scores
the estimator's posterior against that hidden state.

DO NOT call this endpoint during production inference.
Ground truth in this endpoint is from the synthetic simulator ONLY.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.validation import ValidationSummary, run_validation

router = APIRouter(prefix="/api/v1/validation", tags=["validation"])


@router.get("/summary", response_model=ValidationSummary)
def validation_summary(
    scenario: str = "B",
    seed: int = 42,
    total_laps: int = 50,
) -> ValidationSummary:
    """
    Run a controlled hidden-state validation and return aggregate metrics.

    The estimator sees only kinematic observables. Ground truth (the simulator's
    hidden rival SoC) is used only for scoring — never for inference.

    ground_truth_available is always True for synthetic scenarios.
    """
    valid_scenarios = ("A", "B", "C", "D", "E")
    if scenario not in valid_scenarios:
        raise HTTPException(400, f"scenario must be one of {valid_scenarios}")
    if total_laps < 5 or total_laps > 100:
        raise HTTPException(400, "total_laps must be 5–100")

    try:
        return run_validation(scenario=scenario, seed=seed, total_laps=total_laps)
    except Exception as exc:
        raise HTTPException(500, f"Validation run failed: {exc}") from exc
