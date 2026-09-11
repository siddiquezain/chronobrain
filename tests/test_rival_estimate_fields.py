"""Tests for new RivalSocEstimate fields and RivalStateEstimator ML scaffolding."""
from rival_estimator import RivalStateEstimator, RivalEstimatorConfig


def _obs():
    class O:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5
    return O()


def test_estimate_has_ml_model_active_field():
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42)
    for _ in range(10):
        est.predict()
        est.update(_obs())
    result = est.estimate()
    assert hasattr(result, "ml_model_active")
    assert result.ml_model_active is False  # no model injected


def test_estimate_has_state_probs_field():
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42)
    for _ in range(10):
        est.predict()
        est.update(_obs())
    result = est.estimate()
    assert hasattr(result, "state_probs")
    assert isinstance(result.state_probs, dict)
    if result.state_probs:
        assert abs(sum(result.state_probs.values()) - 1.0) < 1e-4


def test_estimator_accepts_obs_model_none():
    """obs_model=None should be accepted without error."""
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42, obs_model=None)
    assert est is not None
