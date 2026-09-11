"""
Strategy Engine — integrates all other engines to produce the final recommendation.

Flow:
  1. Energy state (EnergyEngine)
  2. Overtake analysis (OvertakeEngine)
  3. Regulatory gate for all modes (AppRegulatoryGate)
  4. Risk assessment per candidate mode (RiskEngine)
  5. Utility calculation for each legal mode:
       utility = immediate_gain + overtake_probability * opportunity_value
               + strategic_value - energy_cost_penalty - risk_penalty
  6. Best legal action selected
  7. Explanation built from structured facts (not LLM text)
  8. StrategyRecommendation returned

This engine demonstrates the key demo narrative:
  - Scenario B: high SoC + small gap → ATTACK/ARM recommended
  - Scenario C: low SoC + same gap → BALANCED/CONSERVE recommended
  The system balances opportunity + energy + risk + legality + future state.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from app.engines.energy_engine import EnergyEngine
from app.engines.overtake_engine import OvertakeEngine
from app.engines.regulatory_gate import AppRegulatoryGate
from app.engines.risk_engine import RiskEngine
from app.models.strategy import StrategyRecommendation
from app.models.telemetry import TelemetryState

logger = logging.getLogger(__name__)

# Configurable utility weights — not hardcoded magic numbers
_IMMEDIATE_GAIN_WEIGHT = 0.30
_OVERTAKE_PROB_WEIGHT = 0.25
_STRATEGIC_VALUE_WEIGHT = 0.20
_ENERGY_COST_PENALTY_WEIGHT = 0.15
_RISK_PENALTY_WEIGHT = 0.10

# Opportunity value: how much a successful overtake position gain is worth (in utility units)
_OPPORTUNITY_VALUE = 0.5

# Mode-specific configuration — engineered illustrative values
_MODE_PARAMS = {
    "CONSERVE_MODE": {
        "immediate_gain": 0.05,
        "strategic_value": 0.15,
        "energy_cost_penalty": -0.1,  # negative cost = energy savings
    },
    "BALANCED_MODE": {
        "immediate_gain": 0.15,
        "strategic_value": 0.10,
        "energy_cost_penalty": 0.0,
    },
    "ARM_OVERTAKE_MODE": {
        "immediate_gain": 0.25,
        "strategic_value": 0.25,
        "energy_cost_penalty": 0.15,
    },
    "USE_OVERTAKE_BONUS_MODE": {
        "immediate_gain": 0.40,
        "strategic_value": 0.30,
        "energy_cost_penalty": 0.25,
    },
    "PUSH_MODE": {
        "immediate_gain": 0.30,
        "strategic_value": 0.20,
        "energy_cost_penalty": 0.20,
    },
}

_REASON_CODE_VOCAB = {
    "HIGH_CLOSING_SPEED": "Closing speed is high",
    "STRONG_SLIPSTREAM": "Slipstream effect is strong",
    "FAVORABLE_BRAKING_ZONE": "Braking zone is favorable for an overtake",
    "SUFFICIENT_ENERGY": "Sufficient energy reserve for aggressive deployment",
    "OVERTAKE_BONUS_BANKED": "Overtake bonus banked from previous lap",
    "SMALL_GAP": "Small gap to car ahead",
    "LOW_ENERGY_RESERVE": "Low energy reserve — conservative strategy preferred",
    "DEFENDING_POSITION": "Defending current position",
    "REGULATORY_CONSTRAINT": "Action constrained by FIA 2026 regulations",
    "BALANCED_RACE_CONDITIONS": "Race conditions favor balanced deployment",
}


class StrategyEngine:
    """
    Integrates all ChronoPace engines to produce the full strategy recommendation.
    """

    def __init__(self):
        self._energy = EnergyEngine()
        self._overtake = OvertakeEngine()
        self._risk = RiskEngine()
        self._regulatory = AppRegulatoryGate()

    def recommend(self, telemetry: TelemetryState) -> StrategyRecommendation:
        """
        Full intelligence pipeline: telemetry → StrategyRecommendation.
        """
        energy = self._energy.update_energy_state(telemetry)
        overtake = self._overtake.analyze(telemetry, energy)

        all_modes = list(_MODE_PARAMS.keys())
        regulatory_results = self._regulatory.evaluate_all(telemetry)
        legal_modes = [m for m in all_modes if regulatory_results[m].legal]

        if not legal_modes:
            # All modes illegal — fall back to BALANCED_MODE with violation note
            legal_modes = ["BALANCED_MODE"]

        # Utility calculation for each legal mode
        best_mode = "BALANCED_MODE"
        best_utility = -999.0
        mode_utilities: dict[str, float] = {}

        for mode in legal_modes:
            risk = self._risk.assess(telemetry, energy, overtake, proposed_mode=mode)
            utility = self._calculate_utility(mode, overtake, energy, risk)
            mode_utilities[mode] = utility
            if utility > best_utility:
                best_utility = utility
                best_mode = mode

        # Risk for the chosen mode
        chosen_risk = self._risk.assess(telemetry, energy, overtake, proposed_mode=best_mode)

        reason_codes = self._build_reason_codes(best_mode, overtake, energy, telemetry)
        explanation = self._build_explanation(best_mode, overtake, energy, reason_codes)

        regulatory_check = regulatory_results.get(best_mode, regulatory_results["BALANCED_MODE"])

        confidence = float(np.clip(best_utility, 0.0, 1.0))

        logger.info(
            f"Strategy recommendation: mode={best_mode} utility={best_utility:.3f} "
            f"confidence={confidence:.3f} lap={telemetry.lap}"
        )

        return StrategyRecommendation(
            timestamp=datetime.now(timezone.utc).isoformat(),
            lap=telemetry.lap,
            recommended_mode=best_mode,
            confidence=round(confidence, 4),
            overtake={
                "score": overtake.score,
                "probability": overtake.probability,
                "gap": overtake.gap_s,
                "closing_speed": overtake.closing_speed_mps,
                "slipstream": overtake.slipstream_factor,
                "braking_zone": overtake.braking_zone_score,
            },
            energy={
                "soc": energy.soc_mj,
                "remaining": energy.remaining_mj,
                "deployment_headroom": energy.deployment_headroom_mj,
                "projected_reserve": energy.projected_reserve_mj,
            },
            risk={
                "overtake": chosen_risk["overtake_risk"],
                "energy": chosen_risk["energy_risk"],
                "overall": chosen_risk["overall_risk"],
            },
            regulatory={
                "legal": regulatory_check.legal,
                "violations": regulatory_check.violations,
            },
            decision={
                "utility": round(best_utility, 4),
                "reason_codes": reason_codes,
            },
            explanation=explanation,
            utility=round(float(np.clip(best_utility, 0.0, 1.0)), 4),
            reason_codes=reason_codes,
        )

    def _calculate_utility(
        self, mode: str, overtake, energy, risk: dict
    ) -> float:
        """
        utility = immediate_gain + overtake_probability * opportunity_value
                + strategic_value - energy_cost_penalty - risk_penalty

        All components are normalized [0,1]. Returns a utility in (roughly) [0,1].
        """
        params = _MODE_PARAMS.get(mode, _MODE_PARAMS["BALANCED_MODE"])

        immediate_gain = params["immediate_gain"]
        strategic_value = params["strategic_value"]
        energy_cost_penalty = params["energy_cost_penalty"]

        # Overtake-mode specific: scale by actual overtake probability and energy
        overtake_contribution = overtake.probability * _OPPORTUNITY_VALUE
        if not energy.can_afford_aggressive and mode in (
            "USE_OVERTAKE_BONUS_MODE",
            "ARM_OVERTAKE_MODE",
            "PUSH_MODE",
        ):
            # Penalize aggressive modes heavily when energy is insufficient
            energy_cost_penalty += 0.3

        risk_penalty = risk["overall_risk"]

        utility = (
            _IMMEDIATE_GAIN_WEIGHT * immediate_gain
            + _OVERTAKE_PROB_WEIGHT * overtake_contribution
            + _STRATEGIC_VALUE_WEIGHT * strategic_value
            - _ENERGY_COST_PENALTY_WEIGHT * energy_cost_penalty
            - _RISK_PENALTY_WEIGHT * risk_penalty
        )

        return float(np.clip(utility, 0.0, 1.0))

    def _build_reason_codes(
        self, mode: str, overtake, energy, telemetry: TelemetryState
    ) -> list[str]:
        """Build reason codes from structured facts — not LLM text."""
        codes: list[str] = []

        if mode in ("USE_OVERTAKE_BONUS_MODE", "ARM_OVERTAKE_MODE"):
            if overtake.closing_speed_mps > 5.0:
                codes.append("HIGH_CLOSING_SPEED")
            if overtake.slipstream_factor > 0.5:
                codes.append("STRONG_SLIPSTREAM")
            if overtake.braking_zone_score > 0.6:
                codes.append("FAVORABLE_BRAKING_ZONE")
            if energy.can_afford_aggressive:
                codes.append("SUFFICIENT_ENERGY")
            if telemetry.overtake_qualified_last_lap and mode == "USE_OVERTAKE_BONUS_MODE":
                codes.append("OVERTAKE_BONUS_BANKED")
            gap = telemetry.gap_to_car_ahead_s
            if gap is not None and gap < 1.0:
                codes.append("SMALL_GAP")
        elif mode == "CONSERVE_MODE":
            codes.append("LOW_ENERGY_RESERVE")
        elif mode == "PUSH_MODE":
            codes.append("DEFENDING_POSITION")
        else:
            codes.append("BALANCED_RACE_CONDITIONS")

        return codes

    def _build_explanation(
        self, mode: str, overtake, energy, reason_codes: list[str]
    ) -> str:
        """Generate explanation from structured facts — deterministic, not LLM."""
        parts: list[str] = []

        if mode == "USE_OVERTAKE_BONUS_MODE":
            parts.append(
                "A high-quality overtake window is developing and sufficient energy is "
                "available for a short aggressive deployment."
            )
            if overtake.score > 0.7:
                parts.append(f"Overtake score: {overtake.score:.2f}/1.00.")
        elif mode == "ARM_OVERTAKE_MODE":
            parts.append(
                "Conditions favor arming the overtake sequence. "
                "Qualifying for next-lap bonus deployment."
            )
        elif mode == "CONSERVE_MODE":
            parts.append(
                f"Energy reserve is low ({energy.soc_mj:.1f} MJ). "
                "Conservative deployment to preserve future strategy options."
            )
        elif mode == "PUSH_MODE":
            parts.append(
                "Sustained maximum-legal deployment to defend or improve position."
            )
        else:
            parts.append("Balanced deployment maintaining race pace and energy reserve.")

        if reason_codes:
            parts.append(f"Key factors: {', '.join(reason_codes[:3])}.")

        return " ".join(parts)
