"""Overtake analysis model."""

from typing import Optional

from pydantic import BaseModel, Field


class OvertakeAnalysis(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Overall overtake opportunity score [0,1]")
    probability: float = Field(..., ge=0.0, le=1.0, description="Estimated overtake success probability [0,1]")
    gap_s: Optional[float] = Field(None, ge=0.0, description="Gap to car ahead (seconds)")
    closing_speed_mps: float = Field(..., description="Closing speed on car ahead (m/s)")
    slipstream_factor: float = Field(..., ge=0.0, le=1.0)
    braking_zone_score: float = Field(..., ge=0.0, le=1.0)
    corner_exit_score: float = Field(..., ge=0.0, le=1.0)
    energy_advantage_score: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: list[str] = Field(
        default_factory=list, description="Human-readable factors driving the score"
    )
    closing_speed_score: float = Field(..., ge=0.0, le=1.0)
    gap_score: float = Field(..., ge=0.0, le=1.0)
