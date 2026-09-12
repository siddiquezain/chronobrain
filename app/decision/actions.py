"""
actions.py — candidate strategic actions vs. the feasible set.

Candidate actions are PLANNING alternatives, not deployment modes:

    ATTACK_NOW | WAIT_2_LAPS | WAIT_5_LAPS | CONSERVE | HOLD

Each maps to one or more of the five canonical deployment modes (which are
unchanged). The regulatory gate + an energy check reduce the candidate set to the
FEASIBLE set BEFORE the Monte Carlo horizon evaluates anything — the planner never
optimises an action already known to be illegal or unaffordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from rule_gate import DeploymentMode, GateResult


class CandidateAction(str, Enum):
    ATTACK_NOW = "ATTACK_NOW"
    WAIT_2_LAPS = "WAIT_2_LAPS"
    WAIT_5_LAPS = "WAIT_5_LAPS"
    CONSERVE = "CONSERVE"
    HOLD = "HOLD"


# The deployment mode(s) an action would call on. HOLD/CONSERVE are always-available
# baselines; the attack actions need the overtake modes to be legal.
CANDIDATE_TO_MODES = {
    CandidateAction.ATTACK_NOW: [DeploymentMode.ARM_OVERTAKE_MODE, DeploymentMode.USE_OVERTAKE_BONUS_MODE],
    CandidateAction.WAIT_2_LAPS: [DeploymentMode.ARM_OVERTAKE_MODE, DeploymentMode.USE_OVERTAKE_BONUS_MODE],
    CandidateAction.WAIT_5_LAPS: [DeploymentMode.ARM_OVERTAKE_MODE, DeploymentMode.USE_OVERTAKE_BONUS_MODE],
    CandidateAction.CONSERVE: [DeploymentMode.CONSERVE_MODE],
    CandidateAction.HOLD: [DeploymentMode.BALANCED_MODE],
}

CANDIDATE_DELAY = {
    CandidateAction.ATTACK_NOW: 0,
    CandidateAction.WAIT_2_LAPS: 2,
    CandidateAction.WAIT_5_LAPS: 5,
}


@dataclass
class FeasibilityInput:
    gate_result: GateResult
    soc_mj: float
    projected_reserve_mj: float
    can_afford_aggressive: bool
    reserve_floor_mj: float
    laps_remaining: int
    # MODEL_ASSUMPTION: SoC a non-attacking car recovers per lap while it waits.
    # Mirrors planner.PlannerConfig.harvest_per_lap_mj — that field is the
    # canonical home (the Opportunity Horizon reads it from there); this default
    # is kept numerically in sync manually since this dataclass has no config
    # wiring of its own in _run_feasibility().
    harvest_per_lap_mj: float = 0.3
    # MODEL_ASSUMPTION: energy an aggressive attack sequence needs above the floor.
    # Mirrors planner.PlannerConfig.attack_cost_mj — same note as above.
    attack_cost_mj: float = 1.6


def feasible_actions(
    fin: FeasibilityInput,
) -> Tuple[List[CandidateAction], List[Tuple[str, str]]]:
    """Return (feasible actions, [(action, reason_rejected)])."""
    legal = set(fin.gate_result.legal_modes)
    feasible: List[CandidateAction] = []
    rejected: List[Tuple[str, str]] = []

    for action in CandidateAction:
        if action in (CandidateAction.HOLD, CandidateAction.CONSERVE):
            # always available (BALANCED is the safe fallback; CONSERVE only if legal,
            # else HOLD covers it)
            if action == CandidateAction.CONSERVE and DeploymentMode.CONSERVE_MODE not in legal:
                rejected.append((action.value, "CONSERVE_MODE not legal this lap"))
                continue
            feasible.append(action)
            continue

        needed = CANDIDATE_TO_MODES[action]
        if not any(m in legal for m in needed):
            rejected.append((action.value, "no overtake mode is legal this lap"))
            continue

        delay = CANDIDATE_DELAY[action]
        if delay == 0 and not fin.can_afford_aggressive:
            rejected.append((action.value, "insufficient energy to attack now"))
            continue
        if delay > 0 and delay >= max(1, fin.laps_remaining):
            rejected.append((action.value, f"only {fin.laps_remaining} laps remain"))
            continue
        # WAIT_N is feasible if the car can recover enough energy DURING the wait
        # (not attacking lets it harvest) to fund the attack when it arrives.
        if delay > 0:
            soc_at_window = fin.soc_mj + delay * fin.harvest_per_lap_mj
            if soc_at_window < fin.reserve_floor_mj + fin.attack_cost_mj:
                rejected.append((action.value, "energy would still be too low at the window"))
                continue

        feasible.append(action)

    return feasible, rejected
