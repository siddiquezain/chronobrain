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

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

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

    # Particle filter dynamics.
    # `expected_soc_drift_per_lap_mj` is deliberately near-zero: without jointly
    # inferring the rival's own mode choice (out of scope, see context.md), a large
    # systematic drift biases the posterior low every lap. A small negative value
    # reflects a slight net spend without pretending we know their strategy.
    process_noise_std_mj: float = 0.40
    expected_soc_drift_per_lap_mj: float = -0.08

    # Observation noise std per signal — widened so a single noisy observation
    # cannot collapse the posterior. These are uncertainty budgets, not measured
    # sensor errors.
    observation_noise_std_speed_kmh: float = 9.0
    observation_noise_std_clip_fraction: float = 0.14
    observation_noise_std_accel_g: float = 0.25
    observation_noise_std_sector_delta_s: float = 0.20

    ess_resample_threshold_fraction: float = 0.5

    # Roughening (Gordon et al. 1993): jitter added to particles after resampling
    # to combat sample impoverishment. Without it the SIR filter reports far more
    # confidence than the evidence supports. Also acts as a floor on posterior std.
    roughening_std_mj: float = 0.30
    min_reported_std_mj: float = 0.35

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

    def bucket_distribution(self, low_edge_mj: float, high_edge_mj: float) -> dict:
        """P(LOW / MEDIUM / HIGH) from the Gaussian summary of the posterior."""
        std = max(self.std_soc_mj, 1e-6)
        p_low = _norm_cdf((low_edge_mj - self.mean_soc_mj) / std)
        p_high = 1.0 - _norm_cdf((high_edge_mj - self.mean_soc_mj) / std)
        p_med = max(0.0, 1.0 - p_low - p_high)
        total = p_low + p_med + p_high
        return {"low": p_low / total, "medium": p_med / total, "high": p_high / total}

    def bucket(self, low_edge_mj: float, high_edge_mj: float) -> str:
        d = self.bucket_distribution(low_edge_mj, high_edge_mj)
        return max(d, key=d.get).upper()


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
        self._last_observation: Optional[dict] = None  # diagnostics only

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
        self._last_observation = {
            "terminal_speed_kmh": round(float(observation.terminal_speed_kmh), 3),
            "clipping_point_fraction": round(float(observation.clipping_point_fraction), 4),
            "corner_exit_accel_g": round(float(observation.corner_exit_accel_g), 4),
            "sector_delta_s": round(float(observation.sector_delta_s), 4),
        }
        self._maybe_resample()

    def _maybe_resample(self) -> None:
        cfg = self.config
        ess = 1.0 / np.sum(self._weights**2)
        if ess < cfg.ess_resample_threshold_fraction * cfg.n_particles:
            indices = self._systematic_resample()
            self._particles = self._particles[indices]
            # Roughening: jitter the resampled particles so identical copies spread
            # back out. Combats sample impoverishment / posterior over-confidence.
            self._particles = np.clip(
                self._particles + self._rng.normal(0.0, cfg.roughening_std_mj, cfg.n_particles),
                cfg.soc_min_mj,
                cfg.soc_max_mj,
            )
            self._weights = np.ones(cfg.n_particles) / cfg.n_particles

    def _systematic_resample(self) -> np.ndarray:
        """Standard systematic resampling — O(n), lower variance than multinomial."""
        n = self.config.n_particles
        cumsum = np.cumsum(self._weights)
        positions = (self._rng.uniform(0, 1) + np.arange(n)) / n
        return np.searchsorted(cumsum, positions)

    def estimate(self) -> RivalSocEstimate:
        """Return the current posterior estimate as weighted mean ± std."""
        cfg = self.config
        mean = float(np.sum(self._weights * self._particles))
        variance = float(np.sum(self._weights * (self._particles - mean) ** 2))
        std = float(np.sqrt(max(variance, 0.0)))
        # Floor on reported uncertainty: this is a coarse model of a hidden state,
        # not a measurement. Never claim tighter than this.
        std = max(std, cfg.min_reported_std_mj)
        return RivalSocEstimate(
            mean_soc_mj=round(mean, 4),
            std_soc_mj=round(std, 4),
            n_observations=self._n_observations,
        )

    def posterior_summary(self, n_bins: int = 12) -> dict:
        """
        Compact, serializable view of the particle posterior — for a demo/debug
        visualization. NOT thousands of particles: weighted percentiles + a small
        histogram. Deterministic given the filter's state.
        """
        cfg = self.config
        order = np.argsort(self._particles)
        p_sorted = self._particles[order]
        w_sorted = self._weights[order]
        cum = np.cumsum(w_sorted)

        def wq(q: float) -> float:
            return round(float(np.interp(q, cum, p_sorted)), 4)

        edges = np.linspace(cfg.soc_min_mj, cfg.soc_max_mj, n_bins + 1)
        idx = np.clip(np.digitize(self._particles, edges) - 1, 0, n_bins - 1)
        counts = np.zeros(n_bins)
        np.add.at(counts, idx, self._weights)

        ess = float(1.0 / np.sum(self._weights**2))
        return {
            "n_particles": int(cfg.n_particles),
            "effective_sample_size": round(ess, 1),
            "percentiles": {
                "p05": wq(0.05), "p25": wq(0.25), "p50": wq(0.50),
                "p75": wq(0.75), "p95": wq(0.95),
            },
            "histogram": {
                "bin_edges_mj": [round(float(e), 3) for e in edges],
                "weights": [round(float(c), 5) for c in counts],
            },
        }
