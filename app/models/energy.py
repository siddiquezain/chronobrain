"""Energy state model."""

from pydantic import BaseModel, Field


class EnergyState(BaseModel):
    soc_mj: float = Field(..., ge=0.0, le=9.0, description="State of charge (MJ)")
    soc_pct: float = Field(..., ge=0.0, le=100.0, description="SoC as percentage")
    remaining_mj: float = Field(..., ge=0.0, description="Usable energy remaining (MJ)")
    deployed_this_lap_mj: float = Field(..., ge=0.0, description="Energy deployed this lap (MJ)")
    harvested_this_lap_mj: float = Field(..., ge=0.0, description="Energy harvested this lap (MJ)")
    deployment_headroom_mj: float = Field(
        ..., description="Remaining budget before the per-lap deployment cap (MJ)"
    )
    projected_reserve_mj: float = Field(
        ..., description="Projected energy at end of lap (MJ)"
    )
    projected_end_of_race_mj: float = Field(
        ..., description="Projected energy at race end (MJ)"
    )
    can_afford_aggressive: bool = Field(
        ..., description="True if sufficient reserve for an aggressive deployment"
    )
