"""
narrator.py — DecisionSnapshot -> human-readable explanation.

Wraps the Stack A `narrator._narrate_with_claude` transport but feeds it the new
snapshot contract. The model receives verified numbers and an instruction not to
change them; whatever it returns is used only as free text. `narrate()` never
raises — on any failure it returns the structured fallback.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # avoid a runtime import cycle
    from app.decision.snapshot import DecisionSnapshot

_SYSTEM_PROMPT = (
    "You are the strategy narrator for ChronoPace, a Formula 1 energy-deployment "
    "decision engine. You are given a JSON decision that has ALREADY been computed "
    "and verified by deterministic code. Your ONLY job is to turn it into a short, "
    "clear explanation for a race strategist.\n"
    "HARD RULES:\n"
    "- Never change the mode, the action, the confidence, or any number.\n"
    "- Never contradict the compliance result.\n"
    "- Only use the reason codes provided; do not invent reasons.\n"
    "- Do not introduce race facts that are not in the JSON.\n"
    "Output plain text: a decision line, a confidence line, then the reasons as bullets."
)


def narrate(snapshot: "DecisionSnapshot") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _narrate_with_claude(snapshot, api_key)
        except Exception as exc:  # noqa: BLE001 - never let narration break the response
            logger.warning("Claude narration failed, using structured fallback: %s", exc)
    return structured_fallback(snapshot)


def structured_fallback(snapshot: "DecisionSnapshot") -> str:
    d = snapshot.decision
    lines = [
        f"Decision: {d.mode} ({d.action})",
        f"Confidence: {int(round(d.confidence * 100))}%",
        "Reasons:",
    ]
    lines += [f"  - {r}" for r in snapshot.reasons]
    if d.confidence_overridden:
        lines.append(
            f"Note: the planner preferred {d.stage2_mode}; the confidence gate "
            f"overrode to {d.mode} ({d.override_reason})."
        )
    if not snapshot.compliance.legal:
        lines.append("Note: no legal aggressive mode on this lap — holding.")
    return "\n".join(lines)


def _payload(snapshot: "DecisionSnapshot") -> dict:
    return {
        "decision": snapshot.decision.model_dump(),
        "confidence": snapshot.confidence.model_dump(),
        "rival": snapshot.rival.model_dump(),
        "opportunity": {
            "recommended_strategy": snapshot.opportunity.recommended_strategy,
            "prefers_wait": snapshot.opportunity.prefers_wait,
            "foregone_strategy": snapshot.opportunity.foregone_strategy,
            "foregone_value_gap_s": snapshot.opportunity.foregone_value_gap_s,
        },
        "compliance": {
            "legal": snapshot.compliance.legal,
            "legal_modes": snapshot.compliance.legal_modes,
        },
        "reason_codes": snapshot.reason_codes,
        "reasons": snapshot.reasons,
    }


def _narrate_with_claude(snapshot: "DecisionSnapshot", api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=350,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": "Narrate this verified decision:\n" + json.dumps(_payload(snapshot), indent=2),
        }],
    )
    text = msg.content[0].text.strip()
    return text or structured_fallback(snapshot)
