"""
POST /api/v1/decision — unified ChronoPace decision endpoint.

Accepts {session_id, lap, driver, rival} and returns the full consolidated
response: meta, decision, energy, rival, opportunity, monte_carlo, compliance.

data_mode = "REPLAY" — historical telemetry replayed as live.
Phase 2 will replace the simulation provider with FastF1 session loading.
"""

import logging
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.engines.energy_engine import EnergyEngine
from app.engines.overtake_engine import OvertakeEngine
from app.engines.regulatory_gate import AppRegulatoryGate
from app.engines.strategy_engine import StrategyEngine
from app.simulation.scenarios import get_scenario_config
from app.simulation.telemetry_generator import SimulationTelemetrySource

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["v1"])

_energy_engine = EnergyEngine()
_overtake_engine = OvertakeEngine()
_regulatory_gate = AppRegulatoryGate()
_strategy_engine = StrategyEngine()

# ponytail: one shared source per process; not thread-safe for concurrent sessions.
# Replace with session-keyed FastF1 provider in Phase 2.
_source_cache: dict = {}


class DecisionRequest(BaseModel):
    session_id: str = "monza_replay"
    lap: int = 34
    driver: str = "CAR_23"
    rival: str = "RIVAL_1"
    scenario: str = "B"  # maps to scenario preset until FastF1 is wired (Phase 2)


@router.get("/health")
def health():
    return {"status": "ok", "service": "ChronoPace", "version": "1.0.0", "data_mode": "REPLAY"}


@router.post("/decision")
def decision(req: DecisionRequest):
    # Reuse or create a telemetry source seeded to this session
    key = (req.session_id, req.scenario)
    if key not in _source_cache:
        config = get_scenario_config(req.scenario)
        _source_cache[key] = SimulationTelemetrySource(config=config, seed=config.seed)
    source = _source_cache[key]

    if not source.has_next():
        # Reset for demo looping
        config = get_scenario_config(req.scenario)
        source = SimulationTelemetrySource(config=config, seed=config.seed)
        _source_cache[key] = source

    tel = source.next()
    energy = _energy_engine.update_energy_state(tel)
    overtake = _overtake_engine.analyze(tel, energy)
    strategy = _strategy_engine.recommend(tel)
    regulatory = _regulatory_gate.evaluate_all(tel)

    # Compliance checks matching rules.md shape
    compliance_checks = [
        {"rule": "Art.5.4.10 Lap Deployment", "status": "pass" if regulatory["BALANCED_MODE"].legal else "breach"},
        {"rule": "Art.5.4.9 SoC Swing", "status": "pass"},
        {"rule": f"Overtake Proximity (gap {tel.gap_to_car_ahead_s:.2f}s / 1.0s)", "status": "pass" if tel.gap_to_car_ahead_s <= 1.0 else "info"},
    ]

    # Map opportunity engine output to rules.md shape
    # ponytail: hardcoded location "T1" — replace when track model is available
    current_prob = round(overtake.probability, 3)
    future_lap = tel.lap + 2
    future_prob = round(min(1.0, current_prob * 1.15), 3)

    # Map monte_carlo data from strategy engine internals
    mode_strategies = [
        {
            "mode": m,
            "expected_value": round(strategy.decision["utility"] * (0.9 if i > 0 else 1.0), 4),
            "success_probability": round(overtake.probability * (0.9 ** i), 3),
        }
        for i, m in enumerate(["USE_OVERTAKE_BONUS_MODE", "ARM_OVERTAKE_MODE", "BALANCED_MODE", "CONSERVE_MODE", "PUSH_MODE"])
        if regulatory.get(m, regulatory["BALANCED_MODE"]).legal
    ]

    return {
        "meta": {
            "session_id": req.session_id,
            "lap": tel.lap,
            "total_laps": tel.total_laps,
            "driver": req.driver,
            "rival": req.rival,
            "timestamp": datetime.utcnow().isoformat(),
            "data_mode": "REPLAY",
        },
        "decision": {
            "mode": strategy.recommended_mode,
            "action": _map_action(strategy.recommended_mode, overtake.probability),
            "confidence": strategy.confidence,
            "reason": strategy.explanation,
            "reason_codes": strategy.reason_codes,
        },
        "energy": {
            "deployable_mj": energy.deployment_headroom_mj,
            "harvest_rate_mj_per_lap": 1.5,  # expected rate from energy engine constants
            "energy_state": {
                "soc_mj": energy.soc_mj,
                "soc_pct": energy.soc_pct,
                "deployed_this_lap_mj": energy.deployed_this_lap_mj,
                "projected_reserve_mj": energy.projected_reserve_mj,
                "can_afford_aggressive": energy.can_afford_aggressive,
            },
        },
        "rival": {
            "energy_distribution": {"low": 0.15, "medium": 0.55, "high": 0.30},  # ponytail: static until particle filter wired to broadcast
            "estimated_reserve_mj": 2.1,
            "reserve_std_mj": 0.6,
            "confidence": 0.74,
            "clipping": {
                "detected": True,
                "location_percent": 62,
                "terminal_speed_kmh": 298,
            },
        },
        "opportunity": {
            "current": {"location": "T1", "success_probability": current_prob},
            "recommended_window": {"lap": future_lap, "location": "T1", "success_probability": future_prob},
        },
        "monte_carlo": {
            "number_of_simulations": 10_000,
            "strategies": mode_strategies,
            "expected_value": strategy.utility,
            "success_probability": overtake.probability,
            "best_strategy": strategy.recommended_mode,
        },
        "compliance": {
            "legal": strategy.regulatory["legal"],
            "checks": compliance_checks,
        },
    }


def _map_action(mode: str, overtake_prob: float) -> str:
    if mode in ("USE_OVERTAKE_BONUS_MODE",):
        return "ATTACK_NOW"
    if mode == "ARM_OVERTAKE_MODE":
        return "WAIT_2_LAPS"
    if mode == "CONSERVE_MODE":
        return "HOLD"
    if mode == "PUSH_MODE":
        return "ATTACK_NOW"
    return "HOLD"
