"""Tests for EnergyEngine."""

import pytest

from app.engines.energy_engine import EnergyEngine
from app.models.telemetry import TelemetryState


def _telemetry(**kwargs) -> TelemetryState:
    defaults = dict(
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
    defaults.update(kwargs)
    return TelemetryState(**defaults)


def test_soc_pct_in_range():
    eng = EnergyEngine()
    state = eng.update_energy_state(_telemetry(soc_mj=4.5))
    assert 0.0 <= state.soc_pct <= 100.0


def test_headroom_decreases_with_deployment():
    eng = EnergyEngine()
    low_dep = eng.update_energy_state(_telemetry(energy_deployment_mj=1.0))
    high_dep = eng.update_energy_state(_telemetry(energy_deployment_mj=6.0))
    assert low_dep.deployment_headroom_mj > high_dep.deployment_headroom_mj


def test_bonus_increases_headroom():
    eng = EnergyEngine()
    no_bonus = eng.update_energy_state(_telemetry(overtake_qualified_last_lap=False, energy_deployment_mj=1.0))
    with_bonus = eng.update_energy_state(_telemetry(overtake_qualified_last_lap=True, energy_deployment_mj=1.0))
    assert with_bonus.deployment_headroom_mj > no_bonus.deployment_headroom_mj


def test_low_soc_cannot_afford_aggressive():
    eng = EnergyEngine()
    state = eng.update_energy_state(_telemetry(soc_mj=1.0))
    assert state.can_afford_aggressive is False


def test_high_soc_can_afford_aggressive():
    eng = EnergyEngine()
    # Use few remaining laps so projected_reserve stays above min_energy_reserve_mj
    state = eng.update_energy_state(_telemetry(soc_mj=7.5, energy_deployment_mj=0.5, lap=48, total_laps=50))
    assert state.can_afford_aggressive is True
