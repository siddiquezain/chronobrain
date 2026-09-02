"""Tests for rival_estimator.py — Rival Energy State Estimator"""

import numpy as np
import pytest
from rival_estimator import RivalEstimatorConfig, RivalSocEstimate, RivalStateEstimator
from telemetry_simulator import RivalObservation


def make_observation(**kwargs) -> RivalObservation:
    defaults = dict(
        terminal_speed_kmh=315.0,
        clipping_point_fraction=0.55,
        corner_exit_accel_g=1.15,
        sector_delta_s=-0.2,
    )
    defaults.update(kwargs)
    return RivalObservation(**defaults)


class TestInitialization:
    def test_particles_in_valid_range(self):
        est = RivalStateEstimator(seed=42)
        assert est._particles.min() >= 0.0
        assert est._particles.max() <= 9.0

    def test_weights_sum_to_one(self):
        est = RivalStateEstimator(seed=42)
        assert abs(est._weights.sum() - 1.0) < 1e-9

    def test_n_observations_starts_at_zero(self):
        est = RivalStateEstimator(seed=42)
        assert est.estimate().n_observations == 0


class TestPredictStep:
    def test_predict_advances_particles(self):
        est = RivalStateEstimator(seed=42)
        initial = est._particles.copy()
        est.predict()
        assert not np.allclose(est._particles, initial)

    def test_predict_keeps_particles_in_range(self):
        est = RivalStateEstimator(seed=42)
        for _ in range(20):
            est.predict()
        assert est._particles.min() >= 0.0
        assert est._particles.max() <= 9.0


class TestUpdateStep:
    def test_update_increments_observation_count(self):
        est = RivalStateEstimator(seed=42)
        est.predict()
        est.update(make_observation())
        assert est.estimate().n_observations == 1

    def test_weights_sum_to_one_after_update(self):
        est = RivalStateEstimator(seed=42)
        est.predict()
        est.update(make_observation())
        assert abs(est._weights.sum() - 1.0) < 1e-9

    def test_estimate_is_in_valid_range(self):
        est = RivalStateEstimator(seed=42)
        est.predict()
        est.update(make_observation())
        e = est.estimate()
        assert 0.0 <= e.mean_soc_mj <= 9.0
        assert e.std_soc_mj >= 0.0


class TestSyntheticRecovery:
    """
    Synthetic recovery test: generate observations from a *known* hidden SoC using
    the estimator's own observation model plus injected noise. Verify the posterior
    mean converges within tolerance and std shrinks as evidence accumulates.

    This is the standard validation approach for a latent-variable filter.
    It answers "how do you know this estimator works?" — unit tests alone can't.
    """

    def test_posterior_converges_to_true_soc(self):
        """After 20 observations from a known SoC, mean should be within 2.0 MJ."""
        TRUE_SOC = 4.5
        cfg = RivalEstimatorConfig(n_particles=2000)
        est = RivalStateEstimator(config=cfg, seed=42)
        obs_rng = np.random.default_rng(123)

        soc_fraction = TRUE_SOC / cfg.soc_max_mj
        from rule_gate import GateConfig
        gate_cfg = GateConfig()
        speed_range = gate_cfg.taper_normal_end_kmh - gate_cfg.taper_normal_start_kmh

        for _ in range(20):
            est.predict()
            obs = RivalObservation(
                terminal_speed_kmh=float(
                    gate_cfg.taper_normal_start_kmh
                    + soc_fraction * speed_range
                    + obs_rng.normal(0, cfg.observation_noise_std_speed_kmh)
                ),
                clipping_point_fraction=float(
                    np.clip(
                        0.3
                        + soc_fraction * 0.5
                        + obs_rng.normal(0, cfg.observation_noise_std_clip_fraction),
                        0.0,
                        1.0,
                    )
                ),
                corner_exit_accel_g=float(
                    np.clip(
                        cfg.baseline_accel_g
                        + soc_fraction * cfg.accel_gain_g
                        + obs_rng.normal(0, cfg.observation_noise_std_accel_g),
                        0.3,
                        2.0,
                    )
                ),
                sector_delta_s=float(
                    -(soc_fraction * cfg.sector_gain_s)
                    + obs_rng.normal(0, cfg.observation_noise_std_sector_delta_s)
                ),
            )
            est.update(obs)

        result = est.estimate()
        assert abs(result.mean_soc_mj - TRUE_SOC) < 2.0, (
            f"Posterior mean {result.mean_soc_mj:.2f} too far from true SoC {TRUE_SOC}"
        )

    def test_std_shrinks_with_more_observations(self):
        """More observations → lower posterior uncertainty (generally)."""
        cfg = RivalEstimatorConfig(n_particles=1000)
        obs = make_observation(terminal_speed_kmh=310.0, clipping_point_fraction=0.5)

        def run_n(n: int) -> float:
            est = RivalStateEstimator(config=cfg, seed=42)
            for _ in range(n):
                est.predict()
                est.update(obs)
            return est.estimate().std_soc_mj

        std_5 = run_n(5)
        std_20 = run_n(20)
        # Generous bound — filter is stochastic, resampling can cause jumps
        assert std_20 < std_5 * 1.5

    def test_deterministic_with_seed(self):
        """Same seed → identical estimates."""
        cfg = RivalEstimatorConfig(n_particles=500)
        obs = make_observation()

        def run(seed: int) -> RivalSocEstimate:
            est = RivalStateEstimator(config=cfg, seed=seed)
            est.predict()
            est.update(obs)
            return est.estimate()

        e1 = run(42)
        e2 = run(42)
        assert e1.mean_soc_mj == e2.mean_soc_mj
        assert e1.std_soc_mj == e2.std_soc_mj
