"""
telemetry_simulator.py — Deterministic seeded telemetry generator for ChronoPace.

Produces TelemetryInput and RivalObservation sequences for the Core Demo
Scenarios (A, B, C) and test fixtures. Every consumer of this module is
telemetry-source-agnostic — the same Pydantic models can be populated from
real FastF1 data later without changing any downstream code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    """
    Single-lap telemetry snapshot from our own car.
    All energy values in MJ, gaps in seconds, speed in km/h.
    """

    lap_number: int = Field(..., ge=1, description="Current lap number")
    current_soc_mj: float = Field(..., ge=0.0, le=9.0, description="Current battery SoC in MJ")
    lap_start_soc_mj: Optional[float] = Field(
        None,
        ge=0.0,
        le=9.0,
        description="SoC at start of this lap (enables delta-SoC swing check; None = skip check)",
    )
    lap_energy_deployed_mj: float = Field(
        ...,
        ge=0.0,
        description="MGU-K energy deployed so far this lap (MJ). Tracked against the per-lap deployment cap.",
    )
    gap_to_car_ahead_s: Optional[float] = Field(
        None,
        ge=0.0,
        description="Gap to car ahead at the detection point (seconds). None if no car ahead.",
    )
    gap_to_car_behind_s: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Gap to car behind (seconds). None when rearward telemetry is not modelled. "
            "Used only as a strategic 'defending' signal downstream — never for legality."
        ),
    )
    overtake_qualified_last_lap: bool = Field(
        False,
        description=(
            "True only if this car was within the 1.0s gap threshold on the immediately preceding "
            "lap. Use-it-or-lose-it: the calling orchestrator must not carry True forward more "
            "than one lap."
        ),
    )
    speed_kmh: float = Field(..., ge=0.0, description="Current speed in km/h")
    total_laps: int = Field(..., ge=1, description="Total laps in the race")


class RivalObservation(BaseModel):
    """
    Kinematic observables for a rival car, derived from FIA timing/GPS data.
    All four signals are publicly available to every team on track.
    """

    terminal_speed_kmh: float = Field(..., ge=0.0, description="Peak straight-line speed (km/h)")
    clipping_point_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction along straight where speed trace flattens (0=early, 1=late)",
    )
    corner_exit_accel_g: float = Field(
        ...,
        ge=0.0,
        description="Longitudinal acceleration on corner exit (g). Higher deployment → harder exit.",
    )
    sector_delta_s: float = Field(
        ...,
        description=(
            "Rival's sector time vs. their own rolling baseline (seconds). "
            "Negative = faster than baseline."
        ),
    )


# Scenario presets — initial conditions that produce the expected demo outcomes
SCENARIO_PRESETS: dict[str, dict] = {
    # Normal race — balanced, no strong overtake
    "A": {
        "initial_soc_mj": 5.5,
        "rival_initial_soc_mj": 5.0,
        "gap_to_car_ahead_s": 2.5,
        "rival_terminal_speed_kmh": 315.0,
    },
    # Strong overtake window — high SoC, small gap, high closing speed
    "B": {
        "initial_soc_mj": 7.2,
        "rival_initial_soc_mj": 2.5,
        "gap_to_car_ahead_s": 0.6,
        "rival_terminal_speed_kmh": 308.0,
    },
    # Low energy — opportunity exists but SoC is depleted
    "C": {
        "initial_soc_mj": 1.8,
        "rival_initial_soc_mj": 2.0,
        "gap_to_car_ahead_s": 0.7,
        "rival_terminal_speed_kmh": 310.0,
    },
    # Defensive — car behind approaching quickly
    "D": {
        "initial_soc_mj": 5.0,
        "rival_initial_soc_mj": 7.5,
        "gap_to_car_ahead_s": 3.0,
        "gap_to_car_behind_s": 0.5,
        "rival_terminal_speed_kmh": 325.0,
    },
    # Illegal candidate — deployment already over the modelled per-lap cap
    "E": {
        "initial_soc_mj": 4.0,
        "rival_initial_soc_mj": 4.0,
        "gap_to_car_ahead_s": 0.8,
        "rival_terminal_speed_kmh": 318.0,
        # forces an over-cap lap (above even the +0.5 MJ bonus cap) so the
        # regulatory gate rejects every one of the five modes
        "lap_energy_deployed_mj": 9.7,
    },
}


class TelemetrySimulator:
    """
    Deterministic, seeded generator for TelemetryInput and RivalObservation sequences.

    The only planned telemetry source for the hackathon demo.
    Replace with a real FastF1 adapter by pointing the same Pydantic models at real
    data — no downstream code changes required.
    """

    # Preset keys a caller may override for the interactive demo. Everything else
    # about the scenario stays fixed.
    OVERRIDABLE = {
        "initial_soc_mj",
        "rival_initial_soc_mj",
        "gap_to_car_ahead_s",
        "gap_to_car_behind_s",
        "rival_terminal_speed_kmh",
        "lap_energy_deployed_mj",
        "noise_scale",
    }

    def __init__(
        self,
        scenario: str = "B",
        seed: int = 42,
        total_laps: int = 50,
        preset_overrides: dict | None = None,
    ):
        if scenario not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIO_PRESETS)}")
        self.scenario = scenario
        self.total_laps = total_laps
        self._rng = np.random.default_rng(seed)

        overrides = {
            k: v for k, v in (preset_overrides or {}).items()
            if k in self.OVERRIDABLE and v is not None
        }
        self._preset = {**SCENARIO_PRESETS[scenario], **overrides}
        # `noise_scale` multiplies the RIVAL observation noise only (the signal the
        # estimator sees). 1.0 = default; >1 = noisier telemetry -> a wider posterior.
        self._noise_scale = max(0.0, float(self._preset.get("noise_scale", 1.0)))

        self._lap = 1
        self._soc_mj = self._preset["initial_soc_mj"]
        self._lap_start_soc_mj = self._soc_mj
        self._lap_energy_deployed_mj = 0.0
        self._overtake_qualified_last_lap = False
        self._rival_soc_mj = self._preset["rival_initial_soc_mj"]

        # HIDDEN GROUND TRUTH — the true rival SoC at the end of each lap. Appended
        # by next_lap(). Exists ONLY so a demo/validation layer can score the
        # estimator; it is never placed in TelemetryInput / RivalObservation and
        # never reaches the decision engine.
        self.rival_soc_ground_truth: list[float] = []

    def _harvest_this_lap(self) -> float:
        # ponytail: illustrative harvest model, not validated against real PU data
        return float(np.clip(1.5 + self._rng.normal(0, 0.15), 0.5, 2.5))

    def _deploy_this_lap(self) -> float:
        # ponytail: illustrative deployment model
        return float(np.clip(1.8 + self._rng.normal(0, 0.2), 0.5, 3.5))

    def next_lap(self) -> tuple[TelemetryInput, RivalObservation]:
        """Advance one lap and return (TelemetryInput, RivalObservation)."""
        harvested = self._harvest_this_lap()  # draws kept for RNG-stream stability
        deployed = self._deploy_this_lap()
        if "lap_energy_deployed_mj" in self._preset:
            deployed = float(self._preset["lap_energy_deployed_mj"])
        self._lap_energy_deployed_mj = deployed

        # SoC is mean-reverting toward the scenario's characteristic level rather
        # than monotonically draining: a scenario-based source should keep each
        # scenario "in character" for the whole stint, not run every car flat by
        # mid-race. (harvested/deployed above still feed the per-lap-cap gate field.)
        target_soc = float(self._preset["initial_soc_mj"])
        drift = 0.18 * (target_soc - self._soc_mj) + (harvested - deployed) * 0.5
        self._soc_mj = float(np.clip(self._soc_mj + drift, 0.0, 9.0))

        gap_base = self._preset.get("gap_to_car_ahead_s", 2.0)
        gap = float(np.clip(gap_base + self._rng.normal(0, 0.1), 0.1, 5.0))

        qualified = gap <= 1.0
        last_qualified = self._overtake_qualified_last_lap
        self._overtake_qualified_last_lap = qualified

        # Rival SoC mean-reverts toward its scenario level (like our own car) rather
        # than draining monotonically to zero by mid-race — a scenario-based source
        # should keep the rival's hidden state in a meaningful band all stint, so
        # the rival estimator has something real to track.
        rival_target = float(self._preset["rival_initial_soc_mj"])
        self._rival_soc_mj = float(np.clip(
            self._rival_soc_mj
            + 0.15 * (rival_target - self._rival_soc_mj)
            + self._rng.normal(-0.05, 0.28 * self._noise_scale),
            0.0, 9.0,
        ))
        self.rival_soc_ground_truth.append(round(self._rival_soc_mj, 4))
        rival_obs = self._build_rival_observation()

        gap_behind = self._preset.get("gap_to_car_behind_s")
        telemetry = TelemetryInput(
            lap_number=self._lap,
            current_soc_mj=round(self._soc_mj, 3),
            lap_start_soc_mj=round(self._lap_start_soc_mj, 3),
            lap_energy_deployed_mj=round(self._lap_energy_deployed_mj, 3),
            gap_to_car_ahead_s=round(gap, 3),
            gap_to_car_behind_s=(round(float(gap_behind), 3) if gap_behind is not None else None),
            overtake_qualified_last_lap=last_qualified,
            speed_kmh=float(np.clip(280.0 + self._rng.normal(0, 10), 200, 360)),
            total_laps=self.total_laps,
        )

        self._lap_start_soc_mj = self._soc_mj
        self._lap += 1
        return telemetry, rival_obs

    def _build_rival_observation(self) -> RivalObservation:
        soc_fraction = self._rival_soc_mj / 9.0
        base_speed = self._preset.get("rival_terminal_speed_kmh", 315.0)
        ns = self._noise_scale
        terminal_speed = float(
            np.clip(base_speed + soc_fraction * 15.0 + self._rng.normal(0, 5.0 * ns), 260.0, 360.0)
        )
        clipping = float(
            np.clip(0.3 + soc_fraction * 0.5 + self._rng.normal(0, 0.08 * ns), 0.0, 1.0)
        )
        accel_g = float(
            np.clip(1.0 + soc_fraction * 0.3 + self._rng.normal(0, 0.15 * ns), 0.3, 2.0)
        )
        sector_delta = float(soc_fraction * (-0.4) + self._rng.normal(0, 0.12 * ns))
        return RivalObservation(
            terminal_speed_kmh=round(terminal_speed, 2),
            clipping_point_fraction=round(clipping, 4),
            corner_exit_accel_g=round(accel_g, 4),
            sector_delta_s=round(sector_delta, 4),
        )

    def generate_sequence(
        self, n_laps: int
    ) -> list[tuple[TelemetryInput, RivalObservation]]:
        """Generate a sequence of (TelemetryInput, RivalObservation) for n laps."""
        return [self.next_lap() for _ in range(n_laps)]
