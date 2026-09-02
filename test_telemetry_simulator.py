"""Tests for telemetry_simulator.py"""

import pytest
from telemetry_simulator import TelemetrySimulator, TelemetryInput, RivalObservation


class TestFixtureGeneration:
    def test_next_lap_returns_pair(self):
        sim = TelemetrySimulator(scenario="B", seed=42)
        telemetry, rival = sim.next_lap()
        assert isinstance(telemetry, TelemetryInput)
        assert isinstance(rival, RivalObservation)

    def test_generate_sequence_length(self):
        sim = TelemetrySimulator(scenario="A", seed=42)
        seq = sim.generate_sequence(10)
        assert len(seq) == 10
        assert all(isinstance(t, TelemetryInput) and isinstance(r, RivalObservation) for t, r in seq)

    def test_lap_numbers_increment(self):
        sim = TelemetrySimulator(scenario="B", seed=42)
        laps = [sim.next_lap()[0].lap_number for _ in range(5)]
        assert laps == [1, 2, 3, 4, 5]

    def test_soc_within_bounds(self):
        sim = TelemetrySimulator(scenario="B", seed=42)
        for _ in range(20):
            t, _ = sim.next_lap()
            assert 0.0 <= t.current_soc_mj <= 9.0

    def test_gap_within_bounds(self):
        sim = TelemetrySimulator(scenario="B", seed=42)
        for _ in range(10):
            t, _ = sim.next_lap()
            assert t.gap_to_car_ahead_s is None or t.gap_to_car_ahead_s >= 0.1


class TestDeterminism:
    def test_same_seed_same_output(self):
        sim1 = TelemetrySimulator(scenario="B", seed=99)
        sim2 = TelemetrySimulator(scenario="B", seed=99)
        for _ in range(5):
            t1, r1 = sim1.next_lap()
            t2, r2 = sim2.next_lap()
            assert t1.model_dump() == t2.model_dump()
            assert r1.model_dump() == r2.model_dump()

    def test_different_seeds_differ(self):
        sim1 = TelemetrySimulator(scenario="B", seed=1)
        sim2 = TelemetrySimulator(scenario="B", seed=2)
        t1, _ = sim1.next_lap()
        t2, _ = sim2.next_lap()
        assert t1.current_soc_mj != t2.current_soc_mj


class TestScenarios:
    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError):
            TelemetrySimulator(scenario="Z")

    def test_scenario_b_starts_high_soc(self):
        """Scenario B (strong overtake) should start with high SoC."""
        sim = TelemetrySimulator(scenario="B", seed=42)
        t, _ = sim.next_lap()
        assert t.current_soc_mj > 4.0

    def test_scenario_c_starts_low_soc(self):
        """Scenario C (low energy) should start with low SoC."""
        sim = TelemetrySimulator(scenario="C", seed=42)
        t, _ = sim.next_lap()
        assert t.current_soc_mj < 4.0

    def test_scenario_b_small_gap(self):
        """Scenario B should have small gap (overtake opportunity)."""
        gaps = []
        sim = TelemetrySimulator(scenario="B", seed=42)
        for _ in range(10):
            t, _ = sim.next_lap()
            if t.gap_to_car_ahead_s is not None:
                gaps.append(t.gap_to_car_ahead_s)
        assert sum(g < 1.5 for g in gaps) > 5


class TestRivalObservation:
    def test_rival_obs_fields_valid(self):
        sim = TelemetrySimulator(scenario="B", seed=42)
        _, obs = sim.next_lap()
        assert obs.terminal_speed_kmh > 0
        assert 0.0 <= obs.clipping_point_fraction <= 1.0
        assert obs.corner_exit_accel_g > 0

    def test_rival_obs_is_pydantic_model(self):
        sim = TelemetrySimulator(scenario="A", seed=42)
        _, obs = sim.next_lap()
        assert isinstance(obs, RivalObservation)
        assert isinstance(obs.terminal_speed_kmh, float)
