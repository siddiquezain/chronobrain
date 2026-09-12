"""
Full race pipeline: ingest → preprocess → engines → broadcast-ready payload.

Used by the simulation loop; can also be called with live telemetry when a
real data source is wired in.
"""

import logging
from typing import Any

from app.engines.energy_engine import EnergyEngine
from app.engines.overtake_engine import OvertakeEngine
from app.engines.risk_engine import RiskEngine
from app.engines.strategy_engine import StrategyEngine
from app.pipeline.ingestion import ingest
from app.pipeline.preprocessing import preprocess

logger = logging.getLogger(__name__)

_energy_engine = EnergyEngine()
_overtake_engine = OvertakeEngine()
_risk_engine = RiskEngine()
_strategy_engine = StrategyEngine()


def run(raw_telemetry: dict[str, Any]) -> dict:
    """
    Full pipeline: raw dict → broadcast payload.

    Returns the same shape as RaceSimulator.tick() so callers are interchangeable.
    """
    telemetry = preprocess(ingest(raw_telemetry))

    energy = _energy_engine.update_energy_state(telemetry)
    overtake = _overtake_engine.analyze(telemetry, energy)
    strategy = _strategy_engine.recommend(telemetry)

    race_state = {
        "timestamp": telemetry.timestamp,
        "lap": telemetry.lap,
        "total_laps": telemetry.total_laps,
        "position": telemetry.position,
        "gap_to_car_ahead_s": telemetry.gap_to_car_ahead_s,
        "gap_to_car_behind_s": telemetry.gap_to_car_behind_s,
        "speed_kmh": telemetry.speed_kmh,
        "soc_mj": telemetry.soc_mj,
        "soc_pct": telemetry.soc_pct,
        "tyre_compound": telemetry.tyre_compound,
        "tyre_age_laps": telemetry.tyre_age_laps,
        "sector": telemetry.sector,
        "overtake_mode_eligible": telemetry.overtake_mode_eligible,
    }

    return {
        "race_state": race_state,
        "energy": energy.model_dump(),
        "overtake": overtake.model_dump(),
        "strategy": strategy.model_dump(),
    }
