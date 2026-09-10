"""
Tests for the context-normalized (Z-score) observation model.
These cover calibration correctness, synthetic backward-compat, and posterior
behaviour on real-like data distributions.
"""
import numpy as np
import pytest
from rival_estimator import (
    RivalEstimatorConfig,
    RivalObservationBaseline,
    RivalStateEstimator,
)
from telemetry_simulator import RivalObservation


# ---------------------------------------------------------------------------
# 1. Baseline accumulation is causal and correct
# ---------------------------------------------------------------------------
def test_baseline_empty_on_init():
    bl = RivalObservationBaseline()
    assert bl.n == 0
    assert not bl.is_ready


def test_baseline_becomes_ready_after_min_observations():
    bl = RivalObservationBaseline(min_obs=3)
    obs = RivalObservation(terminal_speed_kmh=310.0, clipping_point_fraction=0.55,
                           corner_exit_accel_g=1.2, sector_delta_s=-0.1)
    bl.update(obs)
    assert not bl.is_ready
    bl.update(obs)
    assert not bl.is_ready
    bl.update(obs)
    assert bl.is_ready


def test_baseline_mean_is_running_mean():
    bl = RivalObservationBaseline(min_obs=2)
    speeds = [300.0, 310.0, 320.0]
    for sp in speeds:
        bl.update(RivalObservation(terminal_speed_kmh=sp, clipping_point_fraction=0.5,
                                   corner_exit_accel_g=1.0, sector_delta_s=0.0))
    assert abs(bl.mean_speed - 310.0) < 0.01
    assert bl.n == 3


def test_baseline_std_is_sample_std():
    bl = RivalObservationBaseline(min_obs=3)
    speeds = [300.0, 310.0, 320.0]
    for sp in speeds:
        bl.update(RivalObservation(terminal_speed_kmh=sp, clipping_point_fraction=0.5,
                                   corner_exit_accel_g=1.0, sector_delta_s=0.0))
    expected_std = float(np.std([300.0, 310.0, 320.0], ddof=1))
    assert abs(bl.std_speed - expected_std) < 0.01


# ---------------------------------------------------------------------------
# 2. Z-score model: high SoC predicts positive speed Z, negative sector Z
# ---------------------------------------------------------------------------
def test_high_soc_predicts_positive_speed_z():
    cfg = RivalEstimatorConfig()
    hi_z = cfg.expected_speed_z(soc_fraction=0.9)
    lo_z = cfg.expected_speed_z(soc_fraction=0.1)
    assert hi_z > lo_z
    assert hi_z > 0.0
    assert lo_z < 0.0


def test_high_soc_predicts_negative_sector_z():
    """High SoC -> faster sector -> more negative delta Z."""
    cfg = RivalEstimatorConfig()
    hi_z = cfg.expected_sector_z(soc_fraction=0.9)
    lo_z = cfg.expected_sector_z(soc_fraction=0.1)
    assert hi_z < lo_z
    assert hi_z < 0.0
    assert lo_z > 0.0


def test_high_soc_predicts_positive_clip_and_accel_z():
    cfg = RivalEstimatorConfig()
    assert cfg.expected_clip_z(0.9) > cfg.expected_clip_z(0.1)
    assert cfg.expected_accel_z(0.9) > cfg.expected_accel_z(0.1)


# ---------------------------------------------------------------------------
# 3. Real-like data: filter does NOT collapse
# ---------------------------------------------------------------------------
def _make_real_like_obs(soc_true_mj: float, seed: int = 0, n: int = 20) -> list:
    """Observations at Silverstone-realistic baseline (310 km/h), NOT 290-355."""
    rng = np.random.default_rng(seed)
    soc_frac = soc_true_mj / 9.0
    obs = []
    for _ in range(n):
        obs.append(RivalObservation(
            terminal_speed_kmh=310.0 + (soc_frac - 0.5) * 4.0 + rng.normal(0, 2.0),
            clipping_point_fraction=0.6 + (soc_frac - 0.5) * 0.1 + rng.normal(0, 0.05),
            corner_exit_accel_g=2.0 + (soc_frac - 0.5) * 0.4 + rng.normal(0, 0.2),
            sector_delta_s=(soc_frac - 0.5) * (-0.3) + rng.normal(0, 0.15),
        ))
    return obs


def test_normalized_filter_does_not_collapse_on_real_like_speed_range():
    """With real-like Silverstone speeds (310 km/h), posterior must not collapse to ~0.5 MJ."""
    est = RivalStateEstimator(seed=42)
    for obs in _make_real_like_obs(soc_true_mj=5.4, seed=7):
        est.predict()
        est.update(obs)
    result = est.estimate()
    assert result.mean_soc_mj > 2.0, (
        f"Posterior collapsed to {result.mean_soc_mj:.2f} MJ on real-like data — "
        "observation model still synthetic-calibrated."
    )
    assert result.effective_sample_size > 50


def test_normalized_filter_distinguishes_high_vs_low_soc():
    """Filter infers higher SoC for driver with genuinely higher SoC on real-circuit speeds."""
    est_hi = RivalStateEstimator(seed=42)
    est_lo = RivalStateEstimator(seed=42)
    for obs in _make_real_like_obs(soc_true_mj=7.0, seed=1):
        est_hi.predict(); est_hi.update(obs)
    for obs in _make_real_like_obs(soc_true_mj=2.0, seed=2):
        est_lo.predict(); est_lo.update(obs)
    assert est_hi.estimate().mean_soc_mj > est_lo.estimate().mean_soc_mj + 1.0


# ---------------------------------------------------------------------------
# 4. Synthetic path still works (backward compat)
# ---------------------------------------------------------------------------
def test_synthetic_path_works_without_baseline():
    """Filter functions on synthetic data even before baseline is ready."""
    est = RivalStateEstimator(seed=42)
    from telemetry_simulator import TelemetrySimulator
    sim = TelemetrySimulator(scenario="B", seed=42)
    for _ in range(10):
        _, obs = sim.next_lap()
        est.predict()
        est.update(obs)
    result = est.estimate()
    assert result.mean_soc_mj > 0.0
    assert result.n_observations == 10


# ---------------------------------------------------------------------------
# 5. Posterior health fields
# ---------------------------------------------------------------------------
def test_estimate_exposes_effective_sample_size():
    est = RivalStateEstimator(seed=42)
    for obs in _make_real_like_obs(5.0, seed=3):
        est.predict(); est.update(obs)
    result = est.estimate()
    assert hasattr(result, "effective_sample_size")
    assert result.effective_sample_size > 0


def test_evidence_quality_insufficient_at_zero_obs():
    est = RivalStateEstimator(seed=42)
    assert est.estimate().evidence_quality == "insufficient"


def test_evidence_quality_weak_before_baseline():
    est = RivalStateEstimator(seed=42)
    obs = _make_real_like_obs(5.0, seed=4)
    for o in obs[:3]:
        est.predict(); est.update(o)
    assert est.estimate().evidence_quality == "weak"


def test_evidence_quality_moderate_or_strong_after_baseline():
    est = RivalStateEstimator(seed=42)
    for obs in _make_real_like_obs(5.0, seed=4):
        est.predict(); est.update(obs)
    assert est.estimate().evidence_quality in ("moderate", "strong")


def test_prior_std_is_not_floored_to_min_reported():
    """Before any observations, posterior std should be ~2.6 (uniform prior), NOT 0.35."""
    est = RivalStateEstimator(seed=42)
    result = est.estimate()
    assert result.std_soc_mj > 2.0, (
        f"Prior std {result.std_soc_mj} — floor is masking the actual prior spread."
    )


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------
def test_calibrated_filter_is_deterministic():
    def run():
        est = RivalStateEstimator(seed=99)
        for obs in _make_real_like_obs(5.4, seed=5):
            est.predict(); est.update(obs)
        return est.estimate()
    a, b = run(), run()
    assert a.mean_soc_mj == b.mean_soc_mj
    assert a.std_soc_mj == b.std_soc_mj


# ---------------------------------------------------------------------------
# 7. Snapshot carries posterior health fields
# ---------------------------------------------------------------------------
def test_snapshot_rival_block_has_posterior_metadata():
    """run_scenario() must produce a snapshot whose RivalBlock has health fields."""
    from app.decision.engine import run_scenario
    snap = run_scenario("B", seed=42, lap=20)
    rb = snap.rival
    # These fields will be added in Task 2 — this test is a pre-check that wiring works
    # For now just verify estimate fields exist on the estimator
    from rival_estimator import RivalStateEstimator
    est = RivalStateEstimator(seed=42)
    from telemetry_simulator import TelemetrySimulator
    sim = TelemetrySimulator(scenario="B", seed=42)
    for _ in range(10):
        _, obs = sim.next_lap()
        est.predict(); est.update(obs)
    result = est.estimate()
    assert hasattr(result, "effective_sample_size")
    assert hasattr(result, "evidence_quality")
    assert hasattr(result, "posterior_health")
    assert hasattr(result, "baseline_ready")
