"""Strategy recommendation model — full output shape for the dashboard."""

from typing import Optional

from pydantic import BaseModel, Field


class StrategyRecommendation(BaseModel):
    """
    Full strategy recommendation. Every field is pre-computed by the backend;
    the frontend renders it directly without recalculating anything.
    """

    timestamp: str
    lap: int
    recommended_mode: str = Field(..., description="DeploymentMode value string")
    confidence: float = Field(..., ge=0.0, le=1.0)

    overtake: dict = Field(
        ...,
        description="score, probability, gap, closing_speed, slipstream, braking_zone",
    )
    energy: dict = Field(
        ...,
        description="soc, remaining, deployment_headroom, projected_reserve",
    )
    risk: dict = Field(..., description="overtake, energy, overall risk scores [0,1]")
    regulatory: dict = Field(..., description="legal (bool), violations (list)")
    decision: dict = Field(..., description="utility, reason_codes")

    explanation: str = Field(..., description="Human-readable explanation from structured facts")
    utility: float = Field(..., ge=0.0, le=1.0, description="Overall utility score [0,1]")
    reason_codes: list[str]
