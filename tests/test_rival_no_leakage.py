"""
Regression suite: our_soc_mj must never enter the rival inference path.

Two levels:
  1. Static  — RIVAL_OBS_FEATURE_NAMES contains no forbidden soc/energy terms.
  2. Dynamic — feature vector length and value-range checks via CaptureModel.
  3. Functional — identical rival observations with different our_soc_mj → identical posteriors.
"""
import numpy as np
import pytest
from rival_estimator import RivalStateEstimator, RivalEstimatorConfig


# ---------------------------------------------------------------------------
# Static leakage check
# ---------------------------------------------------------------------------

def test_feature_names_exclude_soc_and_energy_terms():
    from app.ml.rival_observation_model import RIVAL_OBS_FEATURE_NAMES
    forbidden_terms = {"soc", "energy_mj", "our_", "internal", "replay", "modeled"}
    for name in RIVAL_OBS_FEATURE_NAMES:
        for term in forbidden_terms:
            assert term not in name.lower(), (
                f"Forbidden term '{term}' found in rival obs feature '{name}'"
            )


# ---------------------------------------------------------------------------
# Dynamic leakage checks via CaptureModel
# ---------------------------------------------------------------------------

class _CaptureModel:
    """Stub obs_model that records every feature vector passed to predict_evidence."""
    def __init__(self):
        self.captured = []

    def predict_evidence(self, fv: np.ndarray) -> dict:
        self.captured.append(fv.copy())
        return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}


def _warmup_obs():
    class O:
        terminal_speed_kmh = 300.0
        clipping_point_fraction = 0.75
        corner_exit_accel_g = 0.90
        sector_delta_s = 0.5
    return O()


def _run_estimator_with_capture(n_obs=10):
    cap = _CaptureModel()
    est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42, obs_model=cap)
    for _ in range(n_obs):
        est.predict()
        est.update(_warmup_obs())
    return cap, est


def test_ml_feature_vector_length_is_seven():
    cap, _ = _run_estimator_with_capture()
    assert len(cap.captured) > 0, "CaptureModel was never called"
    assert cap.captured[-1].shape == (1, 7), (
        f"Feature vector shape wrong: {cap.captured[-1].shape}"
    )


def test_ml_feature_values_are_z_score_range():
    """Z-score features (first 4) must be in a small range.
    Raw SoC values are 0–9 MJ and would produce large absolute values if leaked."""
    cap, _ = _run_estimator_with_capture()
    assert len(cap.captured) > 0
    z_features = cap.captured[-1][0, :4]  # first 4 are Z-scores
    assert all(abs(v) < 20.0 for v in z_features), (
        f"Z-score features out of expected range (possible SoC leakage): {z_features}"
    )


def test_update_signature_has_no_soc_parameter():
    import inspect
    sig = inspect.signature(RivalStateEstimator.update)
    params = list(sig.parameters.keys())
    assert "our_soc_mj" not in params
    assert "soc_mj" not in params


# ---------------------------------------------------------------------------
# Functional invariant: our_soc_mj change must not affect rival posterior
# ---------------------------------------------------------------------------

def test_posterior_invariant_to_our_soc_mj_change():
    """
    Two runs with identical rival observations but different our_soc_mj values
    must produce identical posteriors.
    """
    from app.data.samples import NormalizedLap
    from app.data.normalizer import to_rival_observation

    def make_lap(lap: int, our_soc: float) -> NormalizedLap:
        return NormalizedLap(
            lap=lap,
            total_laps=50,
            data_mode="REPLAY",
            our_speed_kmh=280.0,
            our_soc_mj=our_soc,
            rival_terminal_speed_kmh=315.0,
            rival_clipping_point_fraction=0.72,
            rival_corner_exit_accel_g=0.95,
            rival_sector_delta_s=-0.3,
        )

    def run(our_soc: float) -> float:
        est = RivalStateEstimator(config=RivalEstimatorConfig(), seed=42)
        for i in range(15):
            obs = to_rival_observation(make_lap(i + 1, our_soc))
            if obs is not None:
                est.predict()
                est.update(obs)
        return est.estimate().mean_soc_mj

    mean_low  = run(0.5)
    mean_high = run(8.5)
    assert abs(mean_low - mean_high) < 1e-6, (
        f"Posterior differs with our_soc_mj: {mean_low:.4f} vs {mean_high:.4f} "
        "— possible leakage via to_rival_observation()"
    )
