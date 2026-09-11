"""
Integration tests for ML observation model → particle filter.
All tests use stub ML models — no FastF1, no model artifact required.
"""
import numpy as np
import pytest
from rival_estimator import RivalStateEstimator, RivalEstimatorConfig


def _obs(speed=300.0, clip=0.75, accel=0.90, sector=0.5):
    class O:
        terminal_speed_kmh = speed
        clipping_point_fraction = clip
        corner_exit_accel_g = accel
        sector_delta_s = sector
    return O()


class LowEnergyModel:
    """Always signals low energy — drives particles toward LOW bucket."""
    def predict_evidence(self, fv):
        return {"LOW": 0.80, "MEDIUM": 0.15, "HIGH": 0.05}


class HighEnergyModel:
    """Always signals high energy — drives particles toward HIGH bucket."""
    def predict_evidence(self, fv):
        return {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.80}


class NeutralModel:
    """Uniform — no evidence; posterior stays near prior."""
    def predict_evidence(self, fv):
        return {"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3}


def _run(model, n_laps=20, seed=42):
    """Warm up 6 laps (pre-baseline fallback), then run n_laps with the model."""
    cfg = RivalEstimatorConfig()
    est = RivalStateEstimator(config=cfg, seed=seed, obs_model=model)
    for _ in range(6):
        est.predict()
        est.update(_obs())
    for _ in range(n_laps):
        est.predict()
        est.update(_obs())
    return est.estimate(), cfg


def test_low_evidence_shifts_posterior_below_neutral():
    low_est, _  = _run(LowEnergyModel())
    neu_est, _  = _run(NeutralModel())
    assert low_est.mean_soc_mj < neu_est.mean_soc_mj, (
        f"Low-energy evidence didn't shift mean below neutral: "
        f"{low_est.mean_soc_mj:.2f} vs {neu_est.mean_soc_mj:.2f}"
    )


def test_high_evidence_shifts_posterior_above_neutral():
    hi_est, _  = _run(HighEnergyModel())
    neu_est, _ = _run(NeutralModel())
    assert hi_est.mean_soc_mj > neu_est.mean_soc_mj, (
        f"High-energy evidence didn't shift mean above neutral: "
        f"{hi_est.mean_soc_mj:.2f} vs {neu_est.mean_soc_mj:.2f}"
    )


def test_low_evidence_bucket_dist_skews_low():
    est, cfg = _run(LowEnergyModel())
    dist = est.bucket_distribution(cfg.bucket_low_mj, cfg.bucket_high_mj)
    assert dist["low"] > dist["high"], (
        f"Expected P(LOW)>P(HIGH) after low-energy evidence: {dist}"
    )


def test_high_evidence_bucket_dist_skews_high():
    est, cfg = _run(HighEnergyModel())
    dist = est.bucket_distribution(cfg.bucket_low_mj, cfg.bucket_high_mj)
    assert dist["high"] > dist["low"], (
        f"Expected P(HIGH)>P(LOW) after high-energy evidence: {dist}"
    )


def test_ml_model_active_flag_set():
    est, _ = _run(LowEnergyModel())
    assert est.ml_model_active is True


def test_ml_inactive_without_model():
    est, _ = _run(None)
    assert est.ml_model_active is False


def test_single_outlier_does_not_collapse_posterior():
    """One extreme model output after stable evidence should shift mean < 3 MJ."""
    cfg = RivalEstimatorConfig()
    est = RivalStateEstimator(config=cfg, seed=42, obs_model=NeutralModel())
    for _ in range(26):  # 6 warmup + 20 neutral
        est.predict()
        est.update(_obs())
    baseline_mean = est.estimate().mean_soc_mj

    class ExtremeModel:
        def predict_evidence(self, fv):
            return {"LOW": 0.99, "MEDIUM": 0.005, "HIGH": 0.005}

    est._obs_model = ExtremeModel()
    est.predict()
    est.update(_obs(speed=220.0, sector=5.0))
    post_mean = est.estimate().mean_soc_mj
    assert abs(post_mean - baseline_mean) < 3.0, (
        f"Single outlier caused {abs(post_mean - baseline_mean):.2f} MJ shift — "
        "check roughening_std_mj and min_reported_std_mj config"
    )


def test_posterior_deterministic_given_seed():
    """Same seed + same model + same observations → identical posterior mean."""
    def run_once():
        est, _ = _run(LowEnergyModel(), seed=99)
        return est.mean_soc_mj
    assert run_once() == run_once(), "Non-deterministic posterior detected"


def test_state_probs_populated_after_warmup():
    est, _ = _run(LowEnergyModel())
    assert est.state_probs, "state_probs dict is empty after warmup"
    assert abs(sum(est.state_probs.values()) - 1.0) < 1e-4
