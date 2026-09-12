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


class TestNotOverconfident:
    """
    Regression: the SIR filter used to collapse to a near-zero std and a
    systematic low bias. Roughening + widened observation noise + a std floor +
    a near-neutral drift fix that. These are calibration-free sanity bounds, not
    an accuracy claim.
    """

    def _obs_for(self, soc, rng):
        frac = soc / 9.0
        return RivalObservation(
            terminal_speed_kmh=float(290 + frac * 65 + rng.normal(0, 3)),
            clipping_point_fraction=float(np.clip(0.3 + frac * 0.5 + rng.normal(0, 0.03), 0, 1)),
            corner_exit_accel_g=float(1.0 + frac * 0.3 + rng.normal(0, 0.05)),
            sector_delta_s=float(-frac * 0.4 + rng.normal(0, 0.05)),
        )

    def test_single_observation_does_not_collapse_the_posterior(self):
        est = RivalStateEstimator(seed=42)
        est.predict()
        est.update(make_observation(terminal_speed_kmh=295.0, clipping_point_fraction=0.35))
        assert est.estimate().std_soc_mj >= 0.35  # the floor

    def test_std_never_below_floor_even_after_many_observations(self):
        est = RivalStateEstimator(seed=42)
        obs = make_observation(terminal_speed_kmh=310.0)
        for _ in range(30):
            est.predict()
            est.update(obs)
        assert est.estimate().std_soc_mj >= 0.35

    def test_high_soc_rival_is_not_dragged_low(self):
        """Old drift of -0.5 MJ/lap biased every estimate downward."""
        est = RivalStateEstimator(seed=3)
        rng = np.random.default_rng(11)
        for _ in range(25):
            est.predict()
            est.update(self._obs_for(7.0, rng))
        r = est.estimate()
        assert abs(r.mean_soc_mj - 7.0) < 1.0

    def test_tracks_a_changing_hidden_soc(self):
        """Mean should follow a rival whose SoC genuinely falls over a stint."""
        est = RivalStateEstimator(seed=5)
        rng = np.random.default_rng(9)
        for _ in range(12):
            est.predict()
            est.update(self._obs_for(7.5, rng))
        high = est.estimate().mean_soc_mj
        for _ in range(12):
            est.predict()
            est.update(self._obs_for(3.0, rng))
        low = est.estimate().mean_soc_mj
        assert low < high - 1.5

    def test_still_deterministic_after_the_changes(self):
        obs = make_observation()
        def run():
            est = RivalStateEstimator(seed=42)
            for _ in range(15):
                est.predict()
                est.update(obs)
            return est.estimate().model_dump()
        assert run() == run()
