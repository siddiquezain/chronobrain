"""Tests for StrategyEngine."""

import pytest

from app.engines.strategy_engine import StrategyEngine
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
        drs_available=False, overtake_opportunity=False,
        track_position=0.5,
        lap_start_soc_mj=5.0, overtake_qualified_last_lap=False,
    )
    base.update(kw)
    return TelemetryState(**base)


VALID_MODES = {
    "CONSERVE_MODE", "BALANCED_MODE", "ARM_OVERTAKE_MODE",
    "USE_OVERTAKE_BONUS_MODE", "PUSH_MODE",
}


def test_recommendation_has_valid_mode():
    eng = StrategyEngine()
    rec = eng.recommend(_tel())
    assert rec.recommended_mode in VALID_MODES


def test_confidence_in_range():
    eng = StrategyEngine()
    rec = eng.recommend(_tel())
    assert 0.0 <= rec.confidence <= 1.0


def test_explanation_not_empty():
    eng = StrategyEngine()
    rec = eng.recommend(_tel())
    assert len(rec.explanation) > 0


def test_low_energy_favors_conserve_or_balanced():
    eng = StrategyEngine()
    rec = eng.recommend(_tel(soc_mj=1.5))
    assert rec.recommended_mode in ("CONSERVE_MODE", "BALANCED_MODE")


def test_strong_overtake_window_recommends_attack(
):
    """High SoC + small gap + high closing → ARM or USE_OVERTAKE_BONUS."""
    eng = StrategyEngine()
    rec = eng.recommend(_tel(
        soc_mj=7.5, gap_to_car_ahead_s=0.5,
        closing_speed_mps=9.0, slipstream_factor=0.85,
        overtake_qualified_last_lap=True,
        energy_deployment_mj=0.5,
    ))
    assert rec.recommended_mode in ("ARM_OVERTAKE_MODE", "USE_OVERTAKE_BONUS_MODE")
