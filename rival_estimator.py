"""
rival_estimator.py — Rival Energy State Estimator

Estimates a posterior distribution over a rival car's SoC from four kinematic
observables using a particle filter. Returns a labeled, confidence-scored estimate
(mean ± std) — never a point value asserted as fact.

IMPORTANT one-way constraint: this module may import constants FROM rule_gate.py
but rule_gate.py must NEVER import from this module. This enforces the design
rule that rival estimates never reach Stage 1 (legality check).

All four observables are derived from publicly available FIA timing/GPS data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

# One-way import: constants only, never behavior
from rule_gate import GateConfig

_DEFAULT_GATE_CONFIG = GateConfig()


@dataclass(frozen=True)
class RivalEstimatorConfig:
    """
    Configuration for the particle-filter rival SoC estimator.
    All constants are engineered/illustrative — not measured from real car data.
    """

    n_particles: int = 1000
    soc_min_mj: float = 0.0

    # Mirrors GateConfig.max_deployment_per_lap_mj so they never drift apart
    soc_max_mj: float = field(
        default_factory=lambda: _DEFAULT_GATE_CONFIG.max_deployment_per_lap_mj
    )

    # Particle filter dynamics
    process_noise_std_mj: float = 0.3
    expected_soc_drift_per_lap_mj: float = -0.5

    # Observation noise std per signal
    observation_noise_std_speed_kmh: float = 5.0
    observation_noise_std_clip_fraction: float = 0.08
    observation_noise_std_accel_g: float = 0.15
    observation_noise_std_sector_delta_s: float = 0.12

    ess_resample_threshold_fraction: float = 0.5

    # Illustrative observation-model constants (not measured from real car data)
    baseline_accel_g: float = 1.0
    accel_gain_g: float = 0.3
    sector_gain_s: float = 0.4

    default_seed: int = 42


class RivalSocEstimate(BaseModel):
    """
    Posterior estimate of a rival car's SoC.
    A distribution (mean ± std), never a point value asserted as fact.
    """

    mean_soc_mj: float = Field(..., ge=0.0, description="Posterior mean SoC estimate (MJ)")
    std_soc_mj: float = Field(
        ..., ge=0.0, description="Posterior std — uncertainty in estimate (MJ)"
    )
    n_observations: int = Field(
        ..., ge=0, description="Number of observations used to build this estimate"
    )


class RivalStateEstimator:
    """
    Particle filter estimating a rival car's SoC from kinematic observables.

    The posterior after N laps is already informed by all N observations —
    history is captured by the recursive particle weights, not a separate field.
    That's what a particle filter is.
    """

    def __init__(
        self,
        config: Optional[RivalEstimatorConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or RivalEstimatorConfig()
        _seed = seed if seed is not None else self.config.default_seed
        self._rng = np.random.default_rng(_seed)

        cfg = self.config
        self._particles = self._rng.uniform(cfg.soc_min_mj, cfg.soc_max_mj, cfg.n_particles)
        self._weights = np.ones(cfg.n_particles) / cfg.n_particles
        self._n_observations = 0

    def predict(self) -> None:
        """Advance particle SoC estimates by one lap using the drift model."""
        cfg = self.config
        noise = self._rng.normal(0.0, cfg.process_noise_std_mj, cfg.n_particles)
        self._particles = np.clip(
            self._particles + cfg.expected_soc_drift_per_lap_mj + noise,
            cfg.soc_min_mj,
            cfg.soc_max_mj,
        )

    def update(self, observation: "RivalObservation") -> None:  # noqa: F821
        """
        Reweight particles by Gaussian likelihood of observing the given kinematics.
        Independence assumption across four observables — standard for a first-pass filter.
        """
        from telemetry_simulator import RivalObservation  # local to avoid circular

        cfg = self.config
        soc = self._particles
        soc_fraction = soc / cfg.soc_max_mj

        # Expected values per particle SoC
        gate = _DEFAULT_GATE_CONFIG
        speed_range = gate.taper_normal_end_kmh - gate.taper_normal_start_kmh
        expected_speed = gate.taper_normal_start_kmh + soc_fraction * speed_range
        expected_clip = 0.3 + soc_fraction * 0.5
        expected_accel = cfg.baseline_accel_g + soc_fraction * cfg.accel_gain_g
        expected_sector = -(soc_fraction * cfg.sector_gain_s)

        def log_gaussian(x: float, mu: np.ndarray, sigma: float) -> np.ndarray:
            return -0.5 * ((x - mu) / sigma) ** 2

        log_w = (
            log_gaussian(
                observation.terminal_speed_kmh,
                expected_speed,
                cfg.observation_noise_std_speed_kmh,
            )
            + log_gaussian(
                observation.clipping_point_fraction,
                expected_clip,
                cfg.observation_noise_std_clip_fraction,
            )
            + log_gaussian(
                observation.corner_exit_accel_g,
                expected_accel,
                cfg.observation_noise_std_accel_g,
            )
            + log_gaussian(
                observation.sector_delta_s,
                expected_sector,
                cfg.observation_noise_std_sector_delta_s,
            )
        )

        # Numerically stable normalization
        log_w -= log_w.max()
        w = np.exp(log_w)
        w_sum = w.sum()
        if w_sum == 0:
            self._weights = np.ones(cfg.n_particles) / cfg.n_particles
        else:
            self._weights = w / w_sum

        self._n_observations += 1
        self._maybe_resample()

    def _maybe_resample(self) -> None:
        cfg = self.config
        ess = 1.0 / np.sum(self._weights**2)
        if ess < cfg.ess_resample_threshold_fraction * cfg.n_particles:
            indices = self._systematic_resample()
            self._particles = self._particles[indices]
            self._weights = np.ones(cfg.n_particles) / cfg.n_particles

    def _systematic_resample(self) -> np.ndarray:
        """Standard systematic resampling — O(n), lower variance than multinomial."""
        n = self.config.n_particles
        cumsum = np.cumsum(self._weights)
        positions = (self._rng.uniform(0, 1) + np.arange(n)) / n
        return np.searchsorted(cumsum, positions)

    def estimate(self) -> RivalSocEstimate:
        """Return the current posterior estimate as weighted mean ± std."""
        mean = float(np.sum(self._weights * self._particles))
        variance = float(np.sum(self._weights * (self._particles - mean) ** 2))
        std = float(np.sqrt(max(variance, 0.0)))
        return RivalSocEstimate(
            mean_soc_mj=round(mean, 4),
            std_soc_mj=round(std, 4),
            n_observations=self._n_observations,
        )
