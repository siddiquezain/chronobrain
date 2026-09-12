"""Race state and update models."""

from typing import Optional

from pydantic import BaseModel, Field


class RaceState(BaseModel):
    timestamp: str
    lap: int
    total_laps: int
    position: int
    gap_to_car_ahead_s: Optional[float]
    gap_to_car_behind_s: Optional[float]
    speed_kmh: float
    soc_mj: float
    soc_pct: float
    tyre_compound: str
    tyre_age_laps: int
    sector: int
    overtake_mode_eligible: bool


class RaceStateUpdate(BaseModel):
    """Partial update accepted via POST /api/race/update."""

    lap: Optional[int] = None
    position: Optional[int] = None
    gap_to_car_ahead_s: Optional[float] = None
    gap_to_car_behind_s: Optional[float] = None
    speed_kmh: Optional[float] = None
    soc_mj: Optional[float] = None
    soc_pct: Optional[float] = None
    tyre_compound: Optional[str] = None
    tyre_age_laps: Optional[int] = None
    sector: Optional[int] = None
    overtake_mode_eligible: Optional[bool] = None
    overtake_qualified_last_lap: Optional[bool] = None
    lap_energy_deployed_mj: Optional[float] = None
