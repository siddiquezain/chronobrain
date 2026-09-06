"""
samples.py — the two normalized telemetry contracts every source is reduced to.

`TelemetrySample` is the sub-lap primitive (one car, one instant). It is what
`fastf1_service.py` emits and what `normalizer.py` condenses. Only three fields
are required (timestamp, lap, speed); everything a given source cannot supply
stays `None` rather than being faked.

`NormalizedLap` is the per-lap object the ChronoPace decision engine consumes.
A provider yields these; the engine never sees a `TelemetrySample`, a DataFrame,
or a `fastf1` object. `to_telemetry_input()` / `to_rival_observation()` adapt a
`NormalizedLap` to the Stack A single-lap contracts (`rule_gate.TelemetryInput`,
`telemetry_simulator.RivalObservation`) without the engine importing this module's
internals.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DataMode = Literal["SYNTHETIC", "REPLAY", "LIVE"]


class TelemetrySample(BaseModel):
    """
    One car, one instant. Motion channels normalized to SI-ish units:
    speed km/h, throttle/brake in [0, 1], distance in metres.

    F1 does not publish ERS state of charge, so `soc_mj` is populated only by
    the synthetic simulator or by a downstream energy model — never read from a
    real timing feed. Optional fields left `None` are honoured downstream as
    "unavailable", not "zero".
    """

    timestamp_s: float = Field(..., description="Seconds from session/replay start")
    lap: int = Field(..., ge=1, description="Lap this sample belongs to")
    speed_kmh: float = Field(..., ge=0.0)

    distance_m: Optional[float] = Field(None, ge=0.0, description="Distance along the lap")
    throttle: Optional[float] = Field(None, ge=0.0, le=1.0)
    brake: Optional[float] = Field(None, ge=0.0, le=1.0)
    rpm: Optional[float] = Field(None, ge=0.0)
    gear: Optional[int] = Field(None, ge=-1, le=8)
    drs: Optional[bool] = Field(None, description="DRS actually open at this sample")

    position: Optional[int] = Field(None, ge=1, description="Track position where available")
    sector: Optional[int] = Field(None, ge=1, le=3)

    # Modelled, never measured for real F1 — see module docstring.
    soc_mj: Optional[float] = Field(None, ge=0.0)


class NormalizedLap(BaseModel):
    """
    One lap of ChronoPace-normalized state. The single contract between any
    telemetry source and the decision engine.

    Fields split into: race context, our-car state (some modelled for REPLAY),
    rival kinematic observables (optional — the particle filter simply gets fewer
    observations when a source cannot supply them), and provenance flags a
    judge / the frontend can surface.
    """

    # --- race context ---
    lap: int = Field(..., ge=1)
    total_laps: int = Field(..., ge=1)
    data_mode: DataMode = Field(..., description="How this lap was produced")

    # --- our car ---
    our_speed_kmh: float = Field(..., ge=0.0)
    our_soc_mj: Optional[float] = Field(None, ge=0.0, description="None if unknown")
    our_lap_start_soc_mj: Optional[float] = Field(None, ge=0.0)
    our_lap_energy_deployed_mj: float = Field(0.0, ge=0.0)
    gap_to_car_ahead_s: Optional[float] = Field(None, ge=0.0)
    gap_to_car_behind_s: Optional[float] = Field(
        None, ge=0.0, description="Rearward gap — strategic 'defending' signal only, never legality"
    )
    position: Optional[int] = Field(None, ge=1)
    sector: Optional[int] = Field(None, ge=1, le=3)
    drs_available: Optional[bool] = None

    # --- rival kinematic observables (feed rival_estimator particle filter) ---
    rival_terminal_speed_kmh: Optional[float] = Field(None, ge=0.0)
    rival_clipping_point_fraction: Optional[float] = Field(None, ge=0.0, le=1.0)
    rival_corner_exit_accel_g: Optional[float] = Field(None, ge=0.0)
    rival_sector_delta_s: Optional[float] = None

    # --- provenance ---
    energy_is_modeled: bool = Field(
        False,
        description="True when our_soc_mj / deployed came from a model, not the feed "
        "(always True for REPLAY — F1 does not publish ERS SoC).",
    )
    raw_sample_count: int = Field(0, ge=0, description="Sub-lap samples backing this lap")
    source_detail: str = Field("", description="e.g. '2024 Monza R, VER vs LEC'")

    @property
    def has_rival_observation(self) -> bool:
        return (
            self.rival_terminal_speed_kmh is not None
            and self.rival_clipping_point_fraction is not None
            and self.rival_corner_exit_accel_g is not None
            and self.rival_sector_delta_s is not None
        )
