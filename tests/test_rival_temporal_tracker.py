"""Tests for RivalTemporalTracker."""
import numpy as np
import pytest
from rival_estimator import RivalTemporalTracker


def test_speed_persistence_increments_on_negative_z():
    t = RivalTemporalTracker()
    t.update(-1.0, 0.0)
    t.update(-0.5, 0.0)
    assert t.speed_persistence == 2


def test_speed_persistence_resets_on_positive_z():
    t = RivalTemporalTracker()
    t.update(-1.0, 0.0)
    t.update(-1.0, 0.0)
    t.update(0.5, 0.0)   # positive → reset
    assert t.speed_persistence == 0


def test_sector_persistence_increments_on_positive_z():
    """Positive sector Z = slower laps = energy-drain signal."""
    t = RivalTemporalTracker()
    t.update(0.0, 1.0)
    t.update(0.0, 0.8)
    assert t.sector_persistence == 2


def test_speed_slope_negative_when_declining():
    t = RivalTemporalTracker()
    for z in [1.0, 0.5, 0.0, -0.5, -1.0]:
        t.update(z, 0.0)
    assert t.speed_slope < 0


def test_speed_slope_zero_before_two_obs():
    t = RivalTemporalTracker()
    t.update(-1.0, 0.0)
    assert t.speed_slope == 0.0


def test_bucket_thresholds_in_config():
    from rival_estimator import RivalEstimatorConfig
    cfg = RivalEstimatorConfig()
    assert cfg.bucket_low_mj == 2.5
    assert cfg.bucket_high_mj == 5.5
