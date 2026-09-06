"""Race telemetry state model. Units documented in every Field description."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TelemetryState(BaseModel):
    """
    Full race telemetry snapshot. Internally prefer SI where appropriate.
    Speed in km/h for UI consistency; energy in MJ; gaps in seconds; distances in meters.
    """

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    lap: int = Field(..., ge=1, description="Current lap number")
    total_laps: int = Field(..., ge=1, description="Total race laps")

    position: int = Field(..., ge=1, description="Race position (1 = leader)")
    gap_to_car_ahead_s: Optional[float] = Field(
        None, ge=0.0, description="Gap to car ahead (seconds). None if P1 or no car ahead."
    )
    gap_to_car_behind_s: Optional[float] = Field(
        None, ge=0.0, description="Gap to car behind (seconds)"
    )

    speed_kmh: float = Field(..., ge=0.0, description="Current speed (km/h)")
    closing_speed_mps: float = Field(
        ..., description="Closing speed on car ahead (m/s). Positive = closing."
    )

    throttle: float = Field(..., ge=0.0, le=1.0, description="Throttle position [0,1]")
    brake: float = Field(..., ge=0.0, le=1.0, description="Brake pressure [0,1]")
    braking_point: bool = Field(..., description="True if at braking point for next corner")

    sector: int = Field(..., ge=1, le=3, description="Current sector (1, 2, or 3)")
    corner_id: Optional[int] = Field(None, description="Current corner identifier (null on straights)")

    straight_distance_m: float = Field(..., ge=0.0, description="Current straight length (meters)")
    distance_to_next_corner_m: float = Field(
        ..., ge=0.0, description="Distance to next corner braking point (meters)"
    )

    slipstream_factor: float = Field(
        ..., ge=0.0, le=1.0, description="Slipstream benefit [0,1]. 1.0 = maximum tow."
    )

    soc_mj: float = Field(..., ge=0.0, le=9.0, description="Battery SoC (MJ)")
    soc_pct: float = Field(..., ge=0.0, le=100.0, description="Battery SoC as percentage")
    energy_deployment_mj: float = Field(
        ..., ge=0.0, description="MGU-K energy deployed this lap so far (MJ). Tracked against the per-lap deployment cap."
    )
    energy_harvest_mj: float = Field(
        ..., ge=0.0, description="MGU-K energy harvested this lap so far (MJ)"
    )
    energy_remaining_mj: float = Field(
        ..., ge=0.0, description="Usable energy remaining (MJ)"
    )
    energy_budget_mj: float = Field(
        ..., ge=0.0, description="Remaining deployment budget before the per-lap cap (MJ)"
    )

    tyre_age_laps: int = Field(..., ge=0, description="Tyre age in laps")
    tyre_compound: str = Field(..., description="Tyre compound: SOFT, MEDIUM, or HARD")

    drs_available: bool = Field(..., description="True if DRS is available (within 1s gap)")
    overtake_opportunity: bool = Field(
        ..., description="True if basic overtake conditions are met"
    )

    track_position: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized track position [0,1]"
    )

    lap_start_soc_mj: Optional[float] = Field(
        None, ge=0.0, le=9.0, description="SoC at lap start (enables delta-SoC swing check)"
    )
    overtake_qualified_last_lap: bool = Field(
        False, description="True if within gap threshold on immediately preceding lap"
    )
