"""
/api/v1/demo/* — the interactive demonstration + Rival Estimator validation layer.

Same deterministic pipeline as POST /api/v1/decision. These endpoints ADD:
  * manual input overrides (our SoC, hidden rival SoC, gap, noise, ...),
  * a `rival_validation` block scoring the estimator vs the simulator's hidden
    ground truth (DEMO ONLY — never in POST /api/v1/decision),
  * a sequential `rival-trace` for animating the posterior updating.

The frontend still only sends inputs and renders backend outputs.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.routes.v1.decision import FastF1Spec
from app.data.providers import FASTF1_AVAILABLE, SyntheticProvider, build_provider
from app.decision import DecisionConfig, DecisionSnapshot
from app.decision.engine import assemble_snapshot, run_pipeline
from app.demo import (
    DEMO_PRESETS,
    InputOverrides,
    build_rival_validation,
    resolve_preset,
    rival_estimator_trace,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/demo", tags=["demo"])


# ---------------------------------------------------------------------------
# request / response models
# ---------------------------------------------------------------------------
class DemoDecisionRequest(BaseModel):
    source: str = Field("synthetic", description="'synthetic' or 'fastf1'")
    scenario: str = Field("B", description="synthetic scenario A-E")
    seed: int = 42
    total_laps: int = 50
    lap: Optional[int] = None
    with_narrative: bool = False
    overrides: Optional[InputOverrides] = None
    telemetry_glitch: bool = Field(
        False, description="Deterministically corrupt the feed (data-quality / low-confidence demo)"
    )
    fastf1: Optional[FastF1Spec] = None


class DemoDecisionResponse(BaseModel):
    snapshot: DecisionSnapshot
    rival_validation: dict = Field(
        ..., description="ESTIMATED vs GROUND TRUTH — demo only; ground truth is never telemetry"
    )
    inputs_resolved: dict = Field(..., description="The effective simulator inputs after overrides")


class RivalTraceRequest(BaseModel):
    source: str = "synthetic"
    scenario: str = "B"
    seed: int = 42
    total_laps: int = 50
    up_to_lap: Optional[int] = None
    overrides: Optional[InputOverrides] = None
    fastf1: Optional[FastF1Spec] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _provider_for(req, ov: Optional[InputOverrides]):
    if req.source == "fastf1":
        if req.fastf1 is None:
            raise HTTPException(422, "source='fastf1' requires a 'fastf1' block")
        if not FASTF1_AVAILABLE:
            raise HTTPException(400, "the `fastf1` package is not installed on this server")
        return build_provider("fastf1", seed=req.seed, fastf1_kwargs=req.fastf1.model_dump())
    return build_provider(
        "synthetic", scenario=req.scenario, seed=req.seed, total_laps=req.total_laps,
        preset_overrides=(ov.preset_overrides() if ov else None),
        rival_obs_dropout=(ov.rival_obs_dropout or 0.0) if ov else 0.0,
        telemetry_glitch=getattr(req, "telemetry_glitch", False),
    )


def _run_demo_decision(req: DemoDecisionRequest) -> DemoDecisionResponse:
    cfg = DecisionConfig(seed=req.seed)
    ov = req.overrides
    try:
        provider = _provider_for(req, ov)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc

    try:
        ctx = run_pipeline(provider, lap=req.lap, config=cfg)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("demo decision pipeline failed")
        raise HTTPException(500, f"decision pipeline error: {exc}") from exc

    snapshot = assemble_snapshot(ctx, provider.describe())
    if req.with_narrative:
        from app.narrative.narrator import narrate

        snapshot.narrative = narrate(snapshot)

    resolved = {}
    if isinstance(provider, SyntheticProvider):
        from telemetry_simulator import SCENARIO_PRESETS

        resolved = {**SCENARIO_PRESETS[provider.scenario], **provider.preset_overrides}
        resolved["_note"] = "rival_initial_soc_mj here is the HIDDEN ground truth, not telemetry"

    return DemoDecisionResponse(
        snapshot=snapshot,
        rival_validation=build_rival_validation(ctx, provider),
        inputs_resolved=resolved,
    )


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@router.get("/presets")
def list_presets() -> dict:
    return {
        "presets": [
            {
                "key": p.key, "label": p.label, "description": p.description,
                "scenario": p.scenario, "lap": p.lap, "seed": p.seed,
                "overrides": p.overrides, "telemetry_glitch": p.telemetry_glitch,
                "rival_obs_dropout": p.rival_obs_dropout,
                "expectation": p.expectation, "pair_group": p.pair_group,
            }
            for p in DEMO_PRESETS.values()
        ]
    }


@router.post("/decision", response_model=DemoDecisionResponse)
def demo_decision(req: DemoDecisionRequest) -> DemoDecisionResponse:
    return _run_demo_decision(req)


@router.post("/preset/{key}", response_model=DemoDecisionResponse)
def demo_preset(key: str, seed: Optional[int] = None) -> DemoDecisionResponse:
    try:
        p = resolve_preset(key)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    req = DemoDecisionRequest(
        source="synthetic", scenario=p.scenario, seed=seed if seed is not None else p.seed,
        total_laps=p.total_laps, lap=p.lap,
        overrides=InputOverrides(**p.overrides) if p.overrides else None,
        telemetry_glitch=p.telemetry_glitch,
    )
    resp = _run_demo_decision(req)
    resp.inputs_resolved["_preset"] = key
    resp.inputs_resolved["_expectation"] = p.expectation
    return resp


@router.post("/rival-trace")
def demo_rival_trace(req: RivalTraceRequest) -> dict:
    ov = req.overrides
    try:
        return rival_estimator_trace(
            source=req.source, scenario=req.scenario, seed=req.seed,
            total_laps=req.total_laps, up_to_lap=req.up_to_lap,
            preset_overrides=(ov.preset_overrides() if ov else None),
            rival_obs_dropout=(ov.rival_obs_dropout or 0.0) if ov else 0.0,
            fastf1_kwargs=(req.fastf1.model_dump() if req.fastf1 else None),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
