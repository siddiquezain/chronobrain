"""Tests for the race pipeline end-to-end."""

import pytest

from app.pipeline.race_pipeline import run


_BASE_TELEMETRY = dict(
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


def test_pipeline_returns_expected_keys():
    payload = run(_BASE_TELEMETRY)
    assert set(payload.keys()) == {"race_state", "energy", "overtake", "strategy"}


def test_pipeline_race_state_has_lap():
    payload = run(_BASE_TELEMETRY)
    assert payload["race_state"]["lap"] == 25


def test_pipeline_strategy_has_mode():
    payload = run(_BASE_TELEMETRY)
    assert "recommended_mode" in payload["strategy"]


def test_pipeline_energy_has_soc():
    payload = run(_BASE_TELEMETRY)
    assert "soc_mj" in payload["energy"]
