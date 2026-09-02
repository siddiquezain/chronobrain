"""
narrator.py — Stage 4: LLM Narrator

Turns the verified, deterministic decision (ConfidenceGateResult) into a
plain-language sentence using Claude. The LLM NEVER computes, decides, or
changes a number — it only narrates what it was given.

Per §2 of context.md: if the LLM is removed entirely, the input JSON is still
a complete, useful, machine-readable decision — that's the test for whether
this stage stayed in its lane.

THIS STAGE IS OPTIONAL. The backend returns complete, useful JSON without it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from confidence_gate import ConfidenceGateConfig, ConfidenceGateResult
from rule_gate import DeploymentMode

logger = logging.getLogger(__name__)

# Fixed vocabulary — Python selects codes based on gate results.
# The LLM narrates the codes it is given; it never invents new ones.
REASON_CODES: dict[str, str] = {
    "STRONG_OVERTAKE_OPPORTUNITY": "A strong overtake opportunity is developing.",
    "LOW_ESTIMATED_RIVAL_RESERVE": "Rival is estimated to have low energy reserve.",
    "SUFFICIENT_OWN_ENERGY": "Sufficient own energy available for this action.",
    "CURRENT_OPPORTUNITY_OUTVALUES_PROJECTED_WAIT": "Current opportunity exceeds projected future value.",
    "INSUFFICIENT_OWN_ENERGY": "Own energy reserve is insufficient for aggressive deployment.",
    "HIGH_RIVAL_UNCERTAINTY": "Rival energy estimate has high uncertainty — caution warranted.",
    "STATISTICAL_EVIDENCE_WEAK": "Statistical evidence for top mode is below confidence threshold.",
    "DRIVER_LOAD_HIGH": "Driver cognitive load is high — mode change deferred.",
    "REGULATORY_CONSTRAINT": "Action constrained by FIA 2026 regulatory limits.",
    "DEFENDING_POSITION": "Defending current position against approaching rival.",
    "CONSERVING_FOR_FUTURE": "Conserving energy for a better future opportunity.",
    "OVERTAKE_BONUS_BANKED": "Overtake bonus is banked from previous lap — available to spend.",
    "OVERTAKE_BONUS_NOT_AVAILABLE": "Overtake bonus not available (not within gap threshold last lap).",
}


class StrategyCall(BaseModel):
    """
    Complete machine-readable decision that Stage 4 narrates.
    Every field here is derived from prior stages — no new computation.
    """

    decision: DeploymentMode
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_position_gain: float = 0.0
    energy_cost_mj: float = 0.0
    rival_energy_estimate: Optional[dict] = None
    reason_codes: list[str]
    stage2_was_different: bool = False
    override_reason: str = ""
    narration: Optional[str] = None


def build_reason_codes(
    gate_result: ConfidenceGateResult, extra_codes: Optional[list[str]] = None
) -> list[str]:
    """Select reason codes from the fixed vocabulary based on gate results."""
    codes: list[str] = []

    if gate_result.overridden:
        if "STATISTICAL_RELIABILITY" in gate_result.override_reason:
            codes.append("STATISTICAL_EVIDENCE_WEAK")
        if "DCLI" in gate_result.override_reason:
            codes.append("DRIVER_LOAD_HIGH")
        if "RIVAL_CONFIDENCE" in gate_result.override_reason:
            codes.append("HIGH_RIVAL_UNCERTAINTY")
    else:
        mode = gate_result.recommended_mode
        if mode == DeploymentMode.USE_OVERTAKE_BONUS_MODE:
            codes.extend(
                [
                    "STRONG_OVERTAKE_OPPORTUNITY",
                    "CURRENT_OPPORTUNITY_OUTVALUES_PROJECTED_WAIT",
                    "OVERTAKE_BONUS_BANKED",
                ]
            )
        elif mode == DeploymentMode.ARM_OVERTAKE_MODE:
            codes.append("STRONG_OVERTAKE_OPPORTUNITY")
        elif mode == DeploymentMode.CONSERVE_MODE:
            codes.extend(["INSUFFICIENT_OWN_ENERGY", "CONSERVING_FOR_FUTURE"])
        elif mode == DeploymentMode.PUSH_MODE:
            codes.append("DEFENDING_POSITION")

        if gate_result.rival_confidence_passed:
            codes.append("SUFFICIENT_OWN_ENERGY")
        else:
            codes.append("HIGH_RIVAL_UNCERTAINTY")

    if extra_codes:
        codes.extend(c for c in extra_codes if c not in codes)

    return list(dict.fromkeys(codes))  # deduplicate, preserve order


def derive_confidence(gate_result: ConfidenceGateResult) -> float:
    """
    Deterministic confidence scalar [0,1] derived from CI lower bound and gate passage.
    Not an independently eyeballed number.
    """
    if gate_result.overridden:
        return 0.35

    cfg = ConfidenceGateConfig()
    margin = gate_result.ci_lower_bound_s - cfg.min_actionable_laptime_delta_s
    raw = float(np.clip(margin / 1.0, 0.0, 1.0))
    return round(0.5 + 0.45 * raw, 4)


def build_strategy_call(
    gate_result: ConfidenceGateResult,
    energy_cost_mj: float = 0.0,
    rival_estimate: Optional[dict] = None,
    extra_reason_codes: Optional[list[str]] = None,
) -> StrategyCall:
    """Assemble the StrategyCall from verified gate outputs. No LLM call here."""
    reason_codes = build_reason_codes(gate_result, extra_reason_codes)
    confidence = derive_confidence(gate_result)

    return StrategyCall(
        decision=gate_result.recommended_mode,
        confidence=confidence,
        energy_cost_mj=energy_cost_mj,
        rival_energy_estimate=rival_estimate,
        reason_codes=reason_codes,
        stage2_was_different=gate_result.overridden,
        override_reason=gate_result.override_reason,
    )


def narrate(strategy_call: StrategyCall) -> str:
    """
    Generate a plain-language narration of the decision using Claude.
    Falls back to structured text narration if ANTHROPIC_API_KEY is not set.

    The LLM MUST NOT: change any number, alter the decision, override legality,
    invent new reason codes, or introduce race-state facts not given to it.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _narrate_with_claude(strategy_call, api_key)
        except Exception as e:
            logger.warning(f"Claude narration failed, using structured fallback: {e}")

    return _narrate_structured(strategy_call)


def _narrate_structured(call: StrategyCall) -> str:
    """Structured text narration — used when LLM is unavailable."""
    lines = [
        f"Decision: {call.decision.value}",
        f"Confidence: {int(call.confidence * 100)}%",
        "Reasons:",
    ]
    for code in call.reason_codes:
        description = REASON_CODES.get(code, code)
        lines.append(f"  - {description}")

    if call.stage2_was_different:
        lines.append(
            f"Note: Stage 2 recommended an aggressive mode; overridden due to "
            f"{call.override_reason}."
        )

    return "\n".join(lines)


def _narrate_with_claude(call: StrategyCall, api_key: str) -> str:
    """
    Call Claude to turn the StrategyCall JSON into a plain-language narration.
    Claude receives the verified numbers and is instructed not to change them.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    payload = call.model_dump()
    payload["decision"] = call.decision.value

    system_prompt = (
        "You are a motorsport strategy narrator for ChronoPace. "
        "You receive a JSON decision object and turn it into a clear, concise plain-language "
        "explanation. RULES: Do NOT change any number. Do NOT alter the decision. "
        "Do NOT invent reason codes. Do NOT introduce race-state facts not in the JSON. "
        "Output ONLY the narration text, no JSON, no markdown. "
        "Format: Decision line, Confidence line, Reasons as bullet points."
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Narrate this decision:\n{json.dumps(payload, indent=2)}",
            }
        ],
    )
    return message.content[0].text.strip()
