"""Tests for AppRegulatoryGate (app-layer wrapper)."""

import pytest

from app.engines.regulatory_gate import AppRegulatoryGate
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


def test_evaluate_all_returns_five_modes():
    gate = AppRegulatoryGate()
    results = gate.evaluate_all(_tel())
    assert len(results) == 5


def test_all_modes_legal_on_normal_telemetry():
    """All modes legal when gap is small (≤1.0s) and overtake bonus is banked."""
    gate = AppRegulatoryGate()
    results = gate.evaluate_all(_tel(
        gap_to_car_ahead_s=0.5,
        overtake_qualified_last_lap=True,
    ))
    for mode, check in results.items():
        assert check.legal, f"{mode} should be legal: {check.violations}"


def test_over_cap_illegal():
    gate = AppRegulatoryGate()
    results = gate.evaluate_all(_tel(energy_deployment_mj=9.5))
    # At least aggressive modes should be illegal
    for mode in ("ARM_OVERTAKE_MODE", "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE"):
        assert not results[mode].legal, f"{mode} should be illegal over cap"
