"""Tests for compound-stratified RivalObservationBaseline."""
import pytest
from rival_estimator import RivalObservationBaseline


class _Obs:
    """Minimal stand-in for RivalObservation."""
    def __init__(self, speed=300.0, clip=0.7, accel=1.0, sector=0.0):
        self.terminal_speed_kmh = speed
        self.clipping_point_fraction = clip
        self.corner_exit_accel_g = accel
        self.sector_delta_s = sector


def _feed(baseline, obs, compound, n=5):
    """Feed n identical observations into the baseline."""
    for _ in range(n):
        baseline.update(obs, compound=compound)


def test_compound_fallback_to_pooled_when_insufficient():
    """When compound-specific count < min_obs, z_score falls back to pooled stats."""
    bl = RivalObservationBaseline(min_obs=5)
    obs_soft = _Obs(speed=310.0)
    obs_hard = _Obs(speed=295.0)

    # 6 SOFT laps — qualifies for compound-specific stats
    _feed(bl, obs_soft, "SOFT", 6)
    # 3 HARD laps — below min_obs, should fall back to pooled
    _feed(bl, obs_hard, "HARD", 3)

    # Pooled mean ≈ (6*310 + 3*295)/9 ≈ 305; Z for 300 should be negative
    z = bl.z_score(_Obs(speed=300.0), compound="HARD")
    assert isinstance(z, tuple) and len(z) == 4
    assert z[0] < 0.0


def test_compound_specific_stats_used_when_ready():
    """When compound has >= min_obs, z_score uses compound-specific baseline."""
    bl = RivalObservationBaseline(min_obs=5)
    _feed(bl, _Obs(speed=310.0), "SOFT", 6)

    # Z-score for exactly the mean speed on SOFT should be near zero
    z = bl.z_score(_Obs(speed=310.0), compound="SOFT")
    assert abs(z[0]) < 0.1


def test_different_compounds_produce_different_z_scores():
    """Same observation speed produces opposite-sign Z on SOFT vs HARD (different means)."""
    bl = RivalObservationBaseline(min_obs=5)

    for _ in range(6):
        bl.update(_Obs(speed=320.0), compound="SOFT")  # mean 320
    for _ in range(6):
        bl.update(_Obs(speed=300.0), compound="HARD")  # mean 300

    z_soft = bl.z_score(_Obs(speed=310.0), compound="SOFT")  # 310 < 320 → negative
    z_hard = bl.z_score(_Obs(speed=310.0), compound="HARD")  # 310 > 300 → positive

    assert z_soft[0] < 0.0
    assert z_hard[0] > 0.0


def test_unknown_compound_uses_pooled():
    """compound='UNKNOWN' uses pooled baseline (all observations)."""
    bl = RivalObservationBaseline(min_obs=5)
    obs = _Obs(speed=300.0)
    for _ in range(10):
        bl.update(obs, compound="UNKNOWN")

    z = bl.z_score(_Obs(speed=300.0), compound="UNKNOWN")
    assert abs(z[0]) < 0.1


def test_no_compound_arg_backward_compatible():
    """Calling update/z_score without compound kwarg still works (defaults to UNKNOWN)."""
    bl = RivalObservationBaseline(min_obs=5)
    obs = _Obs(speed=300.0)
    for _ in range(6):
        bl.update(obs)  # no compound arg
    z = bl.z_score(obs)   # no compound arg
    assert isinstance(z, tuple) and len(z) == 4


def test_pooled_stats_include_all_compounds():
    """Pooled accumulator (_n) counts all observations regardless of compound."""
    bl = RivalObservationBaseline(min_obs=5)
    for _ in range(3):
        bl.update(_Obs(speed=300.0), compound="SOFT")
    for _ in range(3):
        bl.update(_Obs(speed=300.0), compound="HARD")
    assert bl.n == 6
