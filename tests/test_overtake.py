"""Tests for OvertakeEngine."""

import pytest

from app.engines.energy_engine import EnergyEngine
from app.engines.overtake_engine import OvertakeEngine
from app.models.telemetry import TelemetryState


def _tel(**kw) -> TelemetryState:
    base = dict(
        timestamp="2026-01-01T00:00:00",
        lap=25, total_laps=50, position=5,
        gap_to_car_ahead_s=2.0, gap_to_car_behind_s=2.0,
        speed_kmh=280.0, closing_speed_mps=2.0,
        throttle=0.7, brake=0.1, braking_point=False,
        sector=1, corner_id=None,
        straight_distance_m=500.0, distance_to_next_corner_m=200.0,
        slipstream_factor=0.3,
        soc_mj=5.0, soc_pct=55.6,
        energy_deployment_mj=1.0, energy_harvest_mj=1.5,
        energy_remaining_mj=5.0, energy_budget_mj=8.0,
        tyre_age_laps=15, tyre_compound="MEDIUM",
        overtake_mode_eligible=False, overtake_opportunity=False,
        track_position=0.5,
        lap_start_soc_mj=5.0, overtake_qualified_last_lap=False,
    )
    base.update(kw)
    return TelemetryState(**base)


def _energy(telemetry):
    return EnergyEngine().update_energy_state(telemetry)


def test_score_in_range():
    t = _tel()
    result = OvertakeEngine().analyze(t, _energy(t))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.probability <= 1.0


def test_small_gap_raises_score():
    eng = OvertakeEngine()
    t_close = _tel(gap_to_car_ahead_s=0.3, closing_speed_mps=8.0)
    t_far = _tel(gap_to_car_ahead_s=2.5, closing_speed_mps=1.0)
    e_close = _energy(t_close)
    e_far = _energy(t_far)
    assert eng.analyze(t_close, e_close).score > eng.analyze(t_far, e_far).score


def test_contributing_factors_not_empty_on_good_opportunity():
    t = _tel(gap_to_car_ahead_s=0.3, closing_speed_mps=12.0, slipstream_factor=0.85)
    result = OvertakeEngine().analyze(t, _energy(t))
    assert len(result.contributing_factors) > 0
