"""
reason_codes.py — the single reason-code vocabulary + a deterministic selector.

Replaces the two overlapping vocabularies that existed before (narrator.REASON_CODES
and strategy_engine._REASON_CODE_VOCAB). Python selects codes from `VOCAB` based on
computed facts; the LLM narrator is handed the selected codes and may only phrase
them — it never invents, drops, or reinterprets one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

VOCAB: dict[str, str] = {
    # rival energy state (from the particle-filter posterior)
    "RIVAL_LOW_ENERGY": "Rival is estimated to have a low energy reserve.",
    "RIVAL_MEDIUM_ENERGY": "Rival is estimated to have a medium energy reserve.",
    "RIVAL_HIGH_ENERGY": "Rival is estimated to have a high energy reserve.",
    "RIVAL_ESTIMATE_UNCERTAIN": "The rival energy estimate is too uncertain to act on.",
    # opportunity / timing
    "STRONG_CURRENT_WINDOW": "A strong overtake window is open now.",
    "FUTURE_WINDOW_STRONGER": "A projected future window is worth more than attacking now.",
    "CURRENT_OPPORTUNITY_OUTVALUES_PROJECTED_WAIT": "The current opportunity exceeds the projected value of waiting.",
    "NO_CLEAR_WINDOW": "No overtake window is currently strong enough to commit energy.",
    # our energy
    "SUFFICIENT_OWN_ENERGY": "Our own energy reserve supports this action.",
    "ENERGY_RESERVE_LOW": "Our own energy reserve is low — conserving.",
    "CONSERVING_FOR_FUTURE": "Holding energy back for a better future opportunity.",
    # regulation
    "COMPLIANCE_VALID": "The selected action passes every regulatory check.",
    "REGULATORY_CONSTRAINT": "Regulatory limits constrain the available actions.",
    "NO_LEGAL_AGGRESSIVE_OPTION": "No aggressive mode is legal on this lap.",
    "OVERTAKE_BONUS_BANKED": "An overtake bonus is banked from the previous lap.",
    "OVERTAKE_BONUS_NOT_AVAILABLE": "No overtake bonus is banked for this lap.",
    # confidence
    "STATISTICAL_EVIDENCE_WEAK": "Statistical evidence for the top mode is below threshold.",
    "PRACTICAL_DIFFERENCE_SMALL": "The gap between the top two modes is not practically meaningful.",
    "DRIVER_LOAD_HIGH": "Driver cognitive load is high — deferring the mode change.",
    # defence
    "DEFENDING_POSITION": "Defending track position against a closing car behind.",
    # ML input
    "ML_OVERTAKE_PROB_HIGH": "The overtake-success model rates completion likely.",
    "ML_OVERTAKE_PROB_LOW": "The overtake-success model rates completion unlikely.",
}


def describe(code: str) -> str:
    return VOCAB.get(code, code)


@dataclass(frozen=True)
class ReasonInputs:
    final_mode: str
    action: str
    legal_modes: tuple[str, ...]
    all_modes_illegal: bool
    overtake_bonus_banked: bool
    confidence_overridden: bool
    override_reason: str
    rival_bucket: Optional[str]           # "LOW" | "MEDIUM" | "HIGH" | None
    rival_estimate_uncertain: bool
    horizon_strategy: str                 # "ATTACK_NOW" | "WAIT_2" | ... | "HOLD"
    horizon_prefers_wait: bool
    energy_reserve_low: bool
    can_afford_aggressive: bool
    ml_overtake_prob: Optional[float]
    ml_prob_high: float
    ml_prob_low: float


def select(inp: ReasonInputs) -> List[str]:
    """Deterministic reason-code selection. Order is stable and meaningful."""
    codes: List[str] = []
    aggressive = inp.final_mode in (
        "ARM_OVERTAKE_MODE",
        "USE_OVERTAKE_BONUS_MODE",
        "PUSH_MODE",
    )

    # --- regulation first ---
    if inp.all_modes_illegal:
        codes += ["REGULATORY_CONSTRAINT", "NO_LEGAL_AGGRESSIVE_OPTION"]
    else:
        codes.append("COMPLIANCE_VALID")
        if "USE_OVERTAKE_BONUS_MODE" in inp.legal_modes or inp.overtake_bonus_banked:
            codes.append(
                "OVERTAKE_BONUS_BANKED"
                if inp.overtake_bonus_banked
                else "OVERTAKE_BONUS_NOT_AVAILABLE"
            )
        elif aggressive:
            codes.append("OVERTAKE_BONUS_NOT_AVAILABLE")

    # --- confidence override (not meaningful when the lap itself is illegal) ---
    if inp.confidence_overridden and not inp.all_modes_illegal:
        if "STATISTICAL_RELIABILITY" in inp.override_reason:
            codes.append("STATISTICAL_EVIDENCE_WEAK")
        if "PRACTICAL_SIGNIFICANCE" in inp.override_reason:
            codes.append("PRACTICAL_DIFFERENCE_SMALL")
        if "DCLI" in inp.override_reason:
            codes.append("DRIVER_LOAD_HIGH")
        if "RIVAL_CONFIDENCE" in inp.override_reason:
            codes.append("RIVAL_ESTIMATE_UNCERTAIN")

    # --- rival state ---
    if inp.rival_estimate_uncertain and "RIVAL_ESTIMATE_UNCERTAIN" not in codes:
        codes.append("RIVAL_ESTIMATE_UNCERTAIN")
    elif inp.rival_bucket == "LOW":
        codes.append("RIVAL_LOW_ENERGY")
    elif inp.rival_bucket == "MEDIUM":
        codes.append("RIVAL_MEDIUM_ENERGY")
    elif inp.rival_bucket == "HIGH":
        codes.append("RIVAL_HIGH_ENERGY")

    # --- opportunity / timing ---
    if inp.horizon_prefers_wait and not aggressive:
        codes += ["FUTURE_WINDOW_STRONGER", "CONSERVING_FOR_FUTURE"]
    elif inp.horizon_strategy == "ATTACK_NOW" and aggressive:
        codes += ["STRONG_CURRENT_WINDOW", "CURRENT_OPPORTUNITY_OUTVALUES_PROJECTED_WAIT"]
    elif not aggressive and inp.final_mode == "BALANCED_MODE" and not inp.confidence_overridden:
        codes.append("NO_CLEAR_WINDOW")

    # --- our energy ---
    if inp.energy_reserve_low:
        codes.append("ENERGY_RESERVE_LOW")
    elif aggressive and inp.can_afford_aggressive:
        codes.append("SUFFICIENT_OWN_ENERGY")

    if inp.final_mode == "CONSERVE_MODE" and "CONSERVING_FOR_FUTURE" not in codes:
        codes.append("CONSERVING_FOR_FUTURE")
    if inp.final_mode == "PUSH_MODE":
        codes.append("DEFENDING_POSITION")

    # --- ML overtake probability (input, not decider) ---
    if inp.ml_overtake_prob is not None and aggressive:
        if inp.ml_overtake_prob >= inp.ml_prob_high:
            codes.append("ML_OVERTAKE_PROB_HIGH")
        elif inp.ml_overtake_prob <= inp.ml_prob_low:
            codes.append("ML_OVERTAKE_PROB_LOW")

    # dedupe, preserve order
    seen = set()
    return [c for c in codes if not (c in seen or seen.add(c))]
