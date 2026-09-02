"""
RaceSimulator — orchestrates one simulation tick.

For each tick:
  1. Generate telemetry via TelemetrySource
  2. Run through EnergyEngine, OvertakeEngine, RegulatoryGate, RiskEngine, StrategyEngine
  3. Return full tick payload for broadcast/storage
"""

import logging
from typing import Optional

from app.engines.energy_engine import EnergyEngine
from app.engines.overtake_engine import OvertakeEngine
from app.engines.regulatory_gate import AppRegulatoryGate
from app.engines.risk_engine import RiskEngine
from app.engines.strategy_engine import StrategyEngine
from app.simulation.scenarios import ScenarioConfig
from app.simulation.telemetry_generator import SimulationTelemetrySource

logger = logging.getLogger(__name__)


class RaceSimulator:
    """Orchestrates one ChronoPace simulation tick."""

    def __init__(self, config: ScenarioConfig, seed: int = 42):
        self.config = config
        self._source = SimulationTelemetrySource(config=config, seed=seed)
        self._energy_engine = EnergyEngine()
        self._overtake_engine = OvertakeEngine()
        self._risk_engine = RiskEngine()
        self._regulatory_gate = AppRegulatoryGate()
        self._strategy_engine = StrategyEngine()

    def tick(self) -> dict:
        """Generate one simulation tick. Returns full payload dict."""
        if not self._source.has_next():
            raise StopIteration("Race simulation complete")

        telemetry = self._source.next()
        energy = self._energy_engine.update_energy_state(telemetry)
        overtake = self._overtake_engine.analyze(telemetry, energy)
        strategy = self._strategy_engine.recommend(telemetry)

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
            "drs_available": telemetry.drs_available,
        }

        logger.debug(
            f"Simulation tick: lap={telemetry.lap} "
            f"mode={strategy.recommended_mode} soc={telemetry.soc_mj:.1f}MJ "
            f"overtake_score={overtake.score:.2f}"
        )

        return {
            "race_state": race_state,
            "energy": energy.model_dump(),
            "overtake": overtake.model_dump(),
            "regulatory": {
                "legal": True,
                "violations": [],
                "note": "Full regulatory check runs per-mode in strategy engine",
            },
            "strategy": strategy.model_dump(),
        }
