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

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

DataMode = Literal["SYNTHETIC", "REPLAY", "LIVE"]

# A lap's fitness as *clean racing evidence*. Pit in-laps, out-laps and
# otherwise-flagged laps (FastF1 `IsAccurate == False`, or a lap time far off the
# rolling baseline) are real telemetry but NOT representative racing performance:
# they must not feed the rival-energy particle filter, must not create an overtake
# window, and must not pass the Data Quality Gate as perfect evidence.
LapStatus = Literal["racing", "pit", "out_lap", "invalid_for_energy_inference"]


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


class StrategicRivalInfo(BaseModel):
    """Which opponent the dynamic selector judged most relevant for THIS lap, and
    why. The four `rival_*` observables below belong to `driver`. Absent (`None`)
    for the synthetic path and the fixed two-car replay — behaviour is then exactly
    as before this field existed."""

    driver: str
    role: str = Field(
        "NONE",
        description="ATTACK_TARGET | DEFENDING_THREAT | POSITION_BATTLE | STRATEGICALLY_RELEVANT | NONE",
    )
    position: Optional[int] = Field(None, ge=1)
    gap_s: Optional[float] = Field(None, ge=0.0, description="Unsigned gap magnitude")
    ahead: Optional[bool] = Field(None, description="True = rival ahead of us, False = behind")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)

    # --- tick-level provenance (additive; set only when the replay had telemetry
    #     at tick cadence — the lap value above is then a SUMMARY of the ticks) ---
    tick_level: bool = Field(
        False, description="True when this lap's rival was derived from telemetry-tick selection"
    )
    changes_this_lap: int = Field(0, ge=0, description="Number of strategic-rival switches during this lap")
    tick_share: Dict[str, float] = Field(
        default_factory=dict, description="Fraction of the lap each opponent was the strategic rival"
    )


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
    our_soc_capacity_mj: float = Field(
        9.0, gt=0.0, description="SoC ceiling for the scale our_soc_mj is on (MODEL_ASSUMPTION)"
    )
    our_lap_start_soc_mj: Optional[float] = Field(None, ge=0.0)
    our_lap_energy_deployed_mj: float = Field(0.0, ge=0.0, description="Modeled MGU-K deployment this lap (MJ)")
    our_lap_energy_recovered_mj: Optional[float] = Field(
        None, ge=0.0, description="Modeled ERS recovery this lap (MJ) — brake regen + coast harvest"
    )
    our_lap_net_swing_mj: Optional[float] = Field(
        None, description="Modeled net SoC change this lap (recovered - deployed vs a nominal lap)"
    )
    our_mean_throttle: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="REAL — mean throttle fraction over the lap"
    )
    our_mean_brake: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="REAL — mean brake fraction over the lap"
    )
    gap_to_car_ahead_s: Optional[float] = Field(None, ge=0.0)
    gap_to_car_behind_s: Optional[float] = Field(
        None, ge=0.0, description="Rearward gap — strategic 'defending' signal only, never legality"
    )
    position: Optional[int] = Field(None, ge=1)
    sector: Optional[int] = Field(None, ge=1, le=3)
    drs_available: Optional[bool] = None
    lap_status: LapStatus = Field(
        "racing", description="Our car's lap: 'racing' or a pit/out/invalid lap that is not clean evidence"
    )

    # --- rival kinematic observables (feed rival_estimator particle filter) ---
    strategic_rival: Optional[StrategicRivalInfo] = Field(
        None,
        description="Dynamic-selection metadata: whose observables these are and why. "
        "None => fixed rival (synthetic / two-car replay), behaviour unchanged.",
    )
    rival_lap_status: LapStatus = Field(
        "racing", description="Rival's lap: when not 'racing' the four observables below are None"
    )
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
