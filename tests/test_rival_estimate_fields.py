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


def test_ml_model_called_when_injected():
    """When obs_model is injected, update() must call predict_evidence() after baseline warmup."""
    from rival_estimator import RivalStateEstimator, RivalEstimatorConfig

    calls = []

    class TrackingModel:
        def predict_evidence(self, fv):
            calls.append(fv.shape)
            return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}

    est = RivalStateEstimator(
        config=RivalEstimatorConfig(), seed=42, obs_model=TrackingModel()
    )

    class Obs:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5

    # Run past baseline warm-up (min_obs_for_baseline = 5)
    for _ in range(10):
        est.predict()
        est.update(Obs())

    assert len(calls) > 0, "predict_evidence() was never called"
    assert calls[-1] == (1, 7), f"Expected feature shape (1, 7), got {calls[-1]}"
    assert est.estimate().ml_model_active is True


def test_ml_inactive_before_baseline_ready():
    """ML model must not be called during warm-up (< min_obs_for_baseline = 5)."""
    from rival_estimator import RivalStateEstimator, RivalEstimatorConfig

    calls = []

    class TrackingModel:
        def predict_evidence(self, fv):
            calls.append(True)
            return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}

    est = RivalStateEstimator(
        config=RivalEstimatorConfig(), seed=42, obs_model=TrackingModel()
    )

    class Obs:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5

    # Only 3 observations — below min_obs_for_baseline = 5
    for _ in range(3):
        est.predict()
        est.update(Obs())

    assert len(calls) == 0, "ML model called before baseline warm-up complete"
