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
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# One-way import: constants only, never behavior
from rule_gate import GateConfig

_DEFAULT_GATE_CONFIG = GateConfig()


@dataclass
class RivalObservationBaseline:
    """
    Per-driver causal running statistics for the 4 particle-filter observables.
    Updated before each Z-score computation (causal — never sees the future).
    """
    min_obs: int = 5

    _n: int = field(default=0, init=False, repr=False)
    _sum_speed: float = field(default=0.0, init=False, repr=False)
    _sum_sq_speed: float = field(default=0.0, init=False, repr=False)
    _sum_clip: float = field(default=0.0, init=False, repr=False)
    _sum_sq_clip: float = field(default=0.0, init=False, repr=False)
    _sum_accel: float = field(default=0.0, init=False, repr=False)
    _sum_sq_accel: float = field(default=0.0, init=False, repr=False)
    _sum_sector: float = field(default=0.0, init=False, repr=False)
    _sum_sq_sector: float = field(default=0.0, init=False, repr=False)

    @property
    def n(self) -> int:
        return self._n

    @property
    def is_ready(self) -> bool:
        return self._n >= self.min_obs

    def _stats(self, s: float, sq: float) -> tuple:
        if self._n < 2:
            return (s / max(1, self._n), 1.0)
        mean = s / self._n
        var = max(0.0, sq / self._n - mean * mean) * self._n / (self._n - 1)
        return (mean, max(math.sqrt(var), 1e-4))

    @property
    def mean_speed(self) -> float:
        return self._stats(self._sum_speed, self._sum_sq_speed)[0]

    @property
    def std_speed(self) -> float:
        return self._stats(self._sum_speed, self._sum_sq_speed)[1]

    @property
    def mean_clip(self) -> float:
        return self._stats(self._sum_clip, self._sum_sq_clip)[0]

    @property
    def std_clip(self) -> float:
        return self._stats(self._sum_clip, self._sum_sq_clip)[1]

    @property
    def mean_accel(self) -> float:
        return self._stats(self._sum_accel, self._sum_sq_accel)[0]

    @property
    def std_accel(self) -> float:
        return self._stats(self._sum_accel, self._sum_sq_accel)[1]

    @property
    def mean_sector(self) -> float:
        return self._stats(self._sum_sector, self._sum_sq_sector)[0]

    @property
    def std_sector(self) -> float:
        return self._stats(self._sum_sector, self._sum_sq_sector)[1]

    def update(self, obs) -> None:
        """Fold one observation into running stats. Call BEFORE z_score (causal)."""
        self._n += 1
        sp = float(obs.terminal_speed_kmh)
        cl = float(obs.clipping_point_fraction)
        ac = float(obs.corner_exit_accel_g)
        se = float(obs.sector_delta_s)
        self._sum_speed += sp; self._sum_sq_speed += sp * sp
        self._sum_clip += cl; self._sum_sq_clip += cl * cl
        self._sum_accel += ac; self._sum_sq_accel += ac * ac
        self._sum_sector += se; self._sum_sq_sector += se * se

    def z_score(self, obs) -> tuple:
        """Return (z_speed, z_clip, z_accel, z_sector). Only call when is_ready."""
        return (
            (float(obs.terminal_speed_kmh) - self.mean_speed) / self.std_speed,
            (float(obs.clipping_point_fraction) - self.mean_clip) / self.std_clip,
            (float(obs.corner_exit_accel_g) - self.mean_accel) / self.std_accel,
            (float(obs.sector_delta_s) - self.mean_sector) / self.std_sector,
        )


@dataclass
class RivalTemporalTracker:
    """
    Rolling Z-score history per rival for temporal feature extraction.
    Call update(z_speed, z_sector) once per lap, after baseline.z_score().
    Provides slope + persistence signals for the ML observation model.
    """
    window_size: int = 5
    _speed_z_history: "deque[float]" = field(init=False, repr=False)
    _speed_persist: int = field(default=0, init=False, repr=False)
    _sector_persist: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._speed_z_history = deque(maxlen=self.window_size)

    def update(self, z_speed: float, z_sector: float) -> None:
        """Update after each z_score() call. Same causal ordering as baseline."""
        self._speed_z_history.append(z_speed)
        self._speed_persist = self._speed_persist + 1 if z_speed < 0.0 else 0  # strictly negative; zero is not a low-speed signal
        self._sector_persist = self._sector_persist + 1 if z_sector > 0.0 else 0  # strictly positive; zero is not a sector-loss signal

    @property
    def speed_slope(self) -> float:
        """Linear trend of speed Z over last window_size laps. Negative = declining."""
        h = list(self._speed_z_history)
        if len(h) < 2:
            return 0.0
        x = np.arange(len(h), dtype=float)
        return float(np.polyfit(x, h, 1)[0])

    @property
    def speed_persistence(self) -> int:
        return self._speed_persist

    @property
    def sector_persistence(self) -> int:
        return self._sector_persist


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

    # Energy-state bucket boundaries for ML likelihood mapping.
    # ponytail: model assumptions — no ground-truth calibration available.
    #   Adjust if posterior diagnostics show systematic bias on real sessions.
    bucket_low_mj: float = 2.5   # SoC below this → LOW energy bucket
    bucket_high_mj: float = 5.5  # SoC above this → HIGH energy bucket

    min_obs_for_baseline: int = 5  # matches RivalObservationBaseline.min_obs

    # Z-score observation model gains.
    # (soc_frac - 0.5) * gain = expected Z-score deviation from driver's baseline.
    # High SoC -> above-average speed (positive Z), later clipping (positive Z),
    # harder corner exit (positive Z), faster sector (negative Z for delta).
    # Gains are conservative (<1σ) because ERS effect on lap kinematics is real but modest.
    # ponytail: gains are engineered estimates, not measured from real data.
    z_speed_gain: float = 0.8
    z_clip_gain: float = 0.6
    z_accel_gain: float = 0.6
    z_sector_gain: float = 0.8

    # Observation noise in Z-score units. Tighter than 1σ so the filter accumulates
    # signal fast enough to distinguish SoC levels within ~20 laps.
    # ponytail: tuned against seed=1/2 test pair; revisit if more real-data gains are measured.
    z_observation_noise: float = 0.6

    # Fallback noise before baseline ready — essentially uninformative
    z_fallback_noise: float = 3.0

    def expected_speed_z(self, soc_fraction: float) -> float:
        """Expected Z of terminal speed: high SoC -> positive (faster than baseline)."""
        return (soc_fraction - 0.5) * self.z_speed_gain

    def expected_clip_z(self, soc_fraction: float) -> float:
        """Expected Z of clipping fraction: high SoC -> positive (later clipping)."""
        return (soc_fraction - 0.5) * self.z_clip_gain

    def expected_accel_z(self, soc_fraction: float) -> float:
        """Expected Z of corner-exit accel: high SoC -> positive (harder exit)."""
        return (soc_fraction - 0.5) * self.z_accel_gain

    def expected_sector_z(self, soc_fraction: float) -> float:
        """Expected Z of sector delta: high SoC -> negative (faster = more negative delta)."""
        return -(soc_fraction - 0.5) * self.z_sector_gain


class RivalSocEstimate(BaseModel):
    """
    Posterior estimate of a rival car's SoC.
    A distribution (mean ± std), never a point value asserted as fact.
    """
    mean_soc_mj: float = Field(..., ge=0.0, description="Posterior mean SoC estimate (MJ)")
    std_soc_mj: float = Field(..., ge=0.0, description="Posterior std — uncertainty in estimate (MJ)")
    n_observations: int = Field(..., ge=0, description="Number of observations folded in")
    effective_sample_size: float = Field(
        0.0, ge=0.0,
        description="Particle ESS: 1/Σw². Low = collapsed/degenerate posterior."
    )
    evidence_quality: str = Field(
        "insufficient",
        description="insufficient | weak | moderate | strong — NOT an accuracy claim."
    )
    posterior_health: str = Field(
        "unknown",
        description="healthy | collapsed | roughened | insufficient_data"
    )
    baseline_ready: bool = Field(
        False,
        description="True once Z-score baseline has min_obs observations."
    )
    ml_model_active: bool = Field(
        False,
        description="True when ML observation model contributed to the last update cycle."
    )
    state_probs: dict = Field(
        default_factory=dict,
        description="{'low': p, 'medium': p, 'high': p} bucket distribution from posterior mean ± std."
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
        obs_model=None,  # Optional RivalObservationModel — type not imported to avoid circular
    ):
        self.config = config or RivalEstimatorConfig()
        _seed = seed if seed is not None else self.config.default_seed
        self._rng = np.random.default_rng(_seed)

        cfg = self.config
        self._particles = self._rng.uniform(cfg.soc_min_mj, cfg.soc_max_mj, cfg.n_particles)
        self._weights = np.ones(cfg.n_particles) / cfg.n_particles
        self._n_observations = 0
        self._last_observation: Optional[dict] = None  # diagnostics only
        self._baseline = RivalObservationBaseline(min_obs=cfg.min_obs_for_baseline)
        self._temporal = RivalTemporalTracker()   # temporal feature tracker
        self._obs_model = obs_model               # optional ML observation model
        self._ml_active = False                   # tracks whether ML was used last update

    @property
    def observation_count(self) -> int:
        """How many real observations have been folded in so far."""
        return self._n_observations

    def predict(self) -> None:
        """Advance particle SoC estimates by one lap using the drift model."""
        cfg = self.config
        noise = self._rng.normal(0.0, cfg.process_noise_std_mj, cfg.n_particles)
        self._particles = np.clip(
            self._particles + cfg.expected_soc_drift_per_lap_mj + noise,
            cfg.soc_min_mj,
            cfg.soc_max_mj,
        )

    def update(self, observation) -> None:
        """
        Reweight particles by Gaussian likelihood of observing the given kinematics.

        When baseline is ready (>= min_obs_for_baseline observations), works in
        Z-score space: normalizes against the rival's own running distribution,
        making the model circuit-agnostic and driver-agnostic.

        Before baseline is ready: uses a heavily downweighted single-signal fallback
        (sector delta only, wide noise) so the prior stays roughly uniform.
        """
        from telemetry_simulator import RivalObservation  # local to avoid circular

        cfg = self.config
        soc = self._particles
        soc_fraction = soc / cfg.soc_max_mj

        # Update baseline BEFORE using it for Z-scoring (strictly causal)
        self._baseline.update(observation)
        # note: z_score uses stats including the current obs; self-inclusion bias = 1/n,
        # acceptable at n >= min_obs_for_baseline (default 5).

        def log_gaussian(x, mu: np.ndarray, sigma: float) -> np.ndarray:
            return -0.5 * ((x - mu) / sigma) ** 2

        if self._baseline.is_ready:
            # Z-score model: normalize against rival's own running distribution
            z_sp, z_cl, z_ac, z_se = self._baseline.z_score(observation)
            # Update temporal tracker after z_scores are available (causal)
            self._temporal.update(z_sp, z_se)

            if self._obs_model is not None:
                # ML path: energy-state evidence → piecewise bucket particle weights
                features = np.array([[
                    z_sp, z_cl, z_ac, z_se,
                    self._temporal.speed_slope,
                    float(self._temporal.speed_persistence),
                    float(self._temporal.sector_persistence),
                ]])
                evidence = self._obs_model.predict_evidence(features)
                # Map each particle's SoC → bucket probability
                # ponytail: piecewise constant likelihood; bucket-boundary discontinuities
                #   absorbed by roughening + n_particles=1000. Upgrade to soft-bucket
                #   blending if posterior shows systematic boundary artifacts.
                log_w = np.log(np.clip(
                    np.where(
                        soc < cfg.bucket_low_mj,
                        evidence.get("LOW", 1/3),
                        np.where(
                            soc < cfg.bucket_high_mj,
                            evidence.get("MEDIUM", 1/3),
                            evidence.get("HIGH", 1/3),
                        ),
                    ),
                    1e-10,
                    None,
                ))
                self._ml_active = True
            else:
                # Gaussian fallback: existing hand-coded Z-score observation model
                noise = cfg.z_observation_noise
                log_w = (
                    log_gaussian(z_sp, cfg.expected_speed_z(soc_fraction), noise)
                    + log_gaussian(z_cl, cfg.expected_clip_z(soc_fraction), noise)
                    + log_gaussian(z_ac, cfg.expected_accel_z(soc_fraction), noise)
                    + log_gaussian(z_se, cfg.expected_sector_z(soc_fraction), noise)
                )
                self._ml_active = False
        else:
            # Not enough history to Z-score: uninformative fallback.
            # Use sector delta (already a relative signal) with wide noise,
            # heavily downweighted so the prior stays near-uniform.
            expected_sector_raw = -(soc_fraction - 0.5) * 0.3
            log_w = log_gaussian(
                float(observation.sector_delta_s), expected_sector_raw, cfg.z_fallback_noise
            ) * 0.1   # ponytail: downweight pre-baseline update; removes prior collapse

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
            "baseline_ready": self._baseline.is_ready,
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

        ess = float(1.0 / np.sum(self._weights ** 2))
        ess_fraction = ess / cfg.n_particles

        n = self._n_observations

        # Evidence quality
        if n == 0:
            evidence_quality = "insufficient"
        elif n < cfg.min_obs_for_baseline:
            evidence_quality = "weak"
        elif ess_fraction < 0.05:
            evidence_quality = "weak"
        elif n < 15:
            evidence_quality = "moderate"
        else:
            evidence_quality = "strong"

        # Posterior health
        if n == 0:
            posterior_health = "insufficient_data"
        elif ess_fraction < 0.05:
            posterior_health = "collapsed"
        elif std < cfg.roughening_std_mj * 0.8:
            posterior_health = "roughened"
        else:
            posterior_health = "healthy"

        # Floor only when evidence is present (avoids zero-std numerical issues).
        # NOT presented as empirical uncertainty — use posterior_health to distinguish.
        reported_std = max(std, cfg.min_reported_std_mj) if n > 0 else std

        # Compute state_probs from Gaussian summary of the posterior
        state_probs: dict = {}
        if self._n_observations > 0:
            _std = max(reported_std, 1e-6)
            _p_low = _norm_cdf((cfg.bucket_low_mj - mean) / _std)
            _p_high = 1.0 - _norm_cdf((cfg.bucket_high_mj - mean) / _std)
            _p_med = max(0.0, 1.0 - _p_low - _p_high)
            _tot = _p_low + _p_med + _p_high
            state_probs = {
                "low": round(_p_low / _tot, 4),
                "medium": round(_p_med / _tot, 4),
                "high": round(_p_high / _tot, 4),
            }

        return RivalSocEstimate(
            mean_soc_mj=round(mean, 4),
            std_soc_mj=round(reported_std, 4),
            n_observations=n,
            effective_sample_size=round(ess, 1),
            evidence_quality=evidence_quality,
            posterior_health=posterior_health,
            baseline_ready=self._baseline.is_ready,
            ml_model_active=self._ml_active,
            state_probs=state_probs,
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
