"""
POST /api/v1/decision — the one authoritative ChronoPace decision snapshot.

Runs the deterministic pipeline (app.decision.run_decision) and returns a
`DecisionSnapshot`: decision, energy, rival, opportunity, monte_carlo, compliance,
confidence, reason_codes, trace — plus an optional LLM `narrative` that is
explanation only and never changes a number.

Stateless: the provider is rebuilt from (source, scenario, seed) on every call and
laps 1..N are replayed, so the same request always returns the same snapshot
(bar `meta.generated_at`). No module-level mutable race state.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.data.providers import FASTF1_AVAILABLE, build_provider
from app.decision import DecisionConfig, DecisionSnapshot, run_decision

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["v1"])


class FastF1Spec(BaseModel):
    year: int
    event: str | int
    session: str = "R"
    our_driver: str
    rival_driver: str
    laps: Optional[list[int]] = None


class DecisionRequest(BaseModel):
    source: str = Field("synthetic", description="'synthetic' or 'fastf1'")
    scenario: str = Field("B", description="synthetic scenario A-E")
    seed: int = 42
    total_laps: int = 50
    lap: Optional[int] = Field(None, description="Target lap (default: provider's last lap)")
    with_narrative: bool = Field(False, description="Also fill snapshot.narrative via the LLM/fallback")
    fastf1: Optional[FastF1Spec] = None


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "ChronoPace",
        "version": "1.0.0",
        "fastf1_available": FASTF1_AVAILABLE,
    }


@router.post("/decision", response_model=DecisionSnapshot)
def decision(req: DecisionRequest) -> DecisionSnapshot:
    cfg = DecisionConfig(seed=req.seed)
    try:
        if req.source == "fastf1":
            if req.fastf1 is None:
                raise HTTPException(422, "source='fastf1' requires a 'fastf1' block")
            provider = build_provider(
                "fastf1", seed=req.seed, fastf1_kwargs=req.fastf1.model_dump()
            )
        else:
            provider = build_provider(
                "synthetic", scenario=req.scenario, seed=req.seed, total_laps=req.total_laps
            )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        return run_decision(
            provider, lap=req.lap, config=cfg, with_narrative=req.with_narrative
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("decision pipeline failed")
        raise HTTPException(500, f"decision pipeline error: {exc}") from exc


@router.get("/scenario/{scenario}", response_model=DecisionSnapshot)
def scenario_decision(
    scenario: str, seed: int = 42, lap: Optional[int] = None, narrative: bool = False
) -> DecisionSnapshot:
    """Convenience GET for the demo scenarios."""
    return decision(
        DecisionRequest(
            source="synthetic", scenario=scenario, seed=seed, lap=lap, with_narrative=narrative
        )
    )
