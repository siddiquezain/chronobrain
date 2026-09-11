"""Shared request models for the interactive demo."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InputOverrides(BaseModel):
    """
    Per-field overrides applied over a synthetic scenario preset before the
    deterministic `TelemetrySimulator` runs. All optional; unset fields keep the
    preset value. The frontend sends these; it never sends a deployment mode.

    `rival_initial_soc_mj` sets ONLY the simulator's hidden rival SoC — the
    estimator still has to infer it from the generated telemetry. It is a valid
    input on both the production and demo endpoints for that reason.
    """

    initial_soc_mj: Optional[float] = Field(None, ge=0.0, le=9.0, description="Our starting SoC (MJ)")
    rival_initial_soc_mj: Optional[float] = Field(
        None, ge=0.0, le=9.0, description="HIDDEN rival SoC the simulator uses (never seen by the estimator)"
    )
    gap_to_car_ahead_s: Optional[float] = Field(None, ge=0.0, le=8.0)
    gap_to_car_behind_s: Optional[float] = Field(None, ge=0.0, le=10.0)
    rival_terminal_speed_kmh: Optional[float] = Field(None, ge=250.0, le=360.0)
    lap_energy_deployed_mj: Optional[float] = Field(None, ge=0.0, le=12.0)
    noise_scale: Optional[float] = Field(
        None, ge=0.1, le=8.0,
        description="Multiplies rival-observation noise. >1 = noisier telemetry -> wider posterior.",
    )
    rival_obs_dropout: Optional[float] = Field(
        None, ge=0.0, le=0.9,
        description="Fraction of laps with NO rival observation (freshness / data-quality demo).",
    )

    def preset_overrides(self) -> dict:
        """The subset that goes into the simulator preset (drops rival_obs_dropout)."""
        d = self.model_dump(exclude_none=True)
        d.pop("rival_obs_dropout", None)
        return d
