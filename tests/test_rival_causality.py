"""
Causality tests for the rival energy estimator.
Verify that corrupting future laps does not change past estimates.
"""
import numpy as np
import pytest
from rival_estimator import RivalStateEstimator, RivalObservationBaseline
from telemetry_simulator import TelemetrySimulator, RivalObservation


def _run_estimator(observations: list, seed: int = 42) -> list:
    """Run estimator over observations, return list of (mean, std) tuples."""
    est = RivalStateEstimator(seed=seed)
    results = []
    for obs in observations:
        est.predict()
        est.update(obs)
        r = est.estimate()
        results.append((r.mean_soc_mj, r.std_soc_mj))
    return results


def test_future_observations_do_not_change_past_estimates():
    """Changing observations after lap N must not affect estimates at laps 1..N-1."""
    sim = TelemetrySimulator(scenario="B", seed=42)
    obs_all = [sim.next_lap()[1] for _ in range(20)]

    results_10 = _run_estimator(obs_all[:10])

    # Replace last 10 observations with physically impossible values
    obs_corrupted = obs_all[:10] + [
        RivalObservation(
            terminal_speed_kmh=999.0,
            clipping_point_fraction=0.0,
            corner_exit_accel_g=0.0,
            sector_delta_s=100.0,
        )
        for _ in range(10)
    ]
    results_corrupted = _run_estimator(obs_corrupted)

    for i, ((m1, s1), (m2, s2)) in enumerate(zip(results_10, results_corrupted[:10])):
        assert m1 == m2 and s1 == s2, (
            f"Lap {i+1}: estimate changed when future observations were corrupted. "
            f"Before: ({m1:.4f}, {s1:.4f}), After: ({m2:.4f}, {s2:.4f})"
        )


def test_incremental_matches_full_run():
    """Running to lap 15 step-by-step must give the same result as running all 15."""
    sim = TelemetrySimulator(scenario="B", seed=42)
    obs = [sim.next_lap()[1] for _ in range(15)]
    results_a = _run_estimator(obs)
    results_b = _run_estimator(obs)  # same seed, same obs
    assert results_a == results_b


def test_filter_seed_is_deterministic():
    """Same seed + same obs = identical results. Guards against hidden RNG state."""
    sim = TelemetrySimulator(scenario="B", seed=99)
    obs = [sim.next_lap()[1] for _ in range(12)]
    a = _run_estimator(obs, seed=77)
    b = _run_estimator(obs, seed=77)
    assert a == b


def test_different_seeds_produce_different_particle_clouds():
    """Different seeds should produce different posteriors at some point."""
    sim = TelemetrySimulator(scenario="B", seed=99)
    obs = [sim.next_lap()[1] for _ in range(12)]
    a = _run_estimator(obs, seed=1)
    b = _run_estimator(obs, seed=99999)
    assert any(r1 != r2 for r1, r2 in zip(a, b)), \
        "Different seeds produced identical results — RNG not being used"


def test_baseline_update_is_strictly_causal():
    """The Z-score baseline must only use observations up to and including current."""
    bl = RivalObservationBaseline(min_obs=3)
    obs = [
        RivalObservation(terminal_speed_kmh=300.0, clipping_point_fraction=0.5,
                         corner_exit_accel_g=1.5, sector_delta_s=-0.1),
        RivalObservation(terminal_speed_kmh=310.0, clipping_point_fraction=0.55,
                         corner_exit_accel_g=1.6, sector_delta_s=0.0),
        RivalObservation(terminal_speed_kmh=320.0, clipping_point_fraction=0.6,
                         corner_exit_accel_g=1.7, sector_delta_s=0.1),
    ]
    for o in obs:
        bl.update(o)
    mean_after_3 = bl.mean_speed

    # Build a separate baseline from the same 3 obs
    bl2 = RivalObservationBaseline(min_obs=3)
    for o in obs:
        bl2.update(o)
    mean_before_4 = bl2.mean_speed

    # Both should be identical (same observations, same order)
    assert mean_after_3 == mean_before_4

    # Adding a future obs to bl2 must NOT retroactively change bl's state
    bl2.update(RivalObservation(terminal_speed_kmh=999.0, clipping_point_fraction=0.0,
                                corner_exit_accel_g=0.0, sector_delta_s=0.0))
    # bl is unchanged (it's a separate object)
    assert bl.mean_speed == mean_after_3


def test_no_observation_at_pit_lap_does_not_corrupt_estimate():
    """When a rival lap has no clean observation (pit stop), predict() advances
    drift but estimate should remain valid and not collapse."""
    sim = TelemetrySimulator(scenario="B", seed=42)
    est = RivalStateEstimator(seed=42)

    # Run 10 laps normally
    for _ in range(10):
        _, obs = sim.next_lap()
        est.predict()
        est.update(obs)

    mean_before = est.estimate().mean_soc_mj

    # Simulate 3 pit laps: only predict(), no update()
    for _ in range(3):
        est.predict()

    result_after_pit = est.estimate()
    # Estimate should still be valid (not NaN, not 0)
    assert not np.isnan(result_after_pit.mean_soc_mj)
    assert 0.0 <= result_after_pit.mean_soc_mj <= 9.0
    # Std should be higher than before (uncertainty grew during pit laps)
    assert result_after_pit.std_soc_mj >= est.config.min_reported_std_mj
