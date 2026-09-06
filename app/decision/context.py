"""
context.py — the canonical decision context.

`DecisionContext` is the single object every stage reads from and writes to inside
`run_decision()`. Before it existed, the pipeline threaded ~7 loosely-related
objects (GateResult, PlanningContext, PlannerResult, RivalSocEstimate,
DriverLoadInput, HorizonResult, ...) between free functions, which is exactly how
stale state creeps in. One mutable context, populated stage by stage, is the fix.

It is a plain dataclass, not a Pydantic model — it is internal to one
`run_decision()` call and never serialized. The serialized output is
`DecisionSnapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from confidence_gate import ConfidenceGateResult
from opportunity_engine import HorizonResult
from planner import PlannerResult
from rival_estimator import RivalSocEstimate
from rule_gate import GateResult

from app.data.samples import NormalizedLap
from app.decision.config import DecisionConfig


@dataclass
class EnergyFeatures:
    soc_mj: Optional[float]
    soc_pct: Optional[float]
    lap_start_soc_mj: Optional[float]
    deployed_this_lap_mj: float
    deployment_headroom_mj: float
    projected_reserve_mj: float
    projected_end_of_race_mj: float
    can_afford_aggressive: bool
    reserve_low: bool
    energy_is_modeled: bool


@dataclass
class RivalFeatures:
    estimate: RivalSocEstimate
    bucket: str                     # LOW | MEDIUM | HIGH
    distribution: dict              # low/medium/high probabilities
    confidence: float
    uncertain: bool


@dataclass
class DecisionContext:
    # inputs
    config: DecisionConfig
    lap_history: List[NormalizedLap]           # laps 1..N (N = target lap)
    target_lap: NormalizedLap

    # threaded facts
    overtake_bonus_banked: bool = False

    # stage outputs (filled in order)
    energy: Optional[EnergyFeatures] = None
    ml_overtake_prob: Optional[float] = None
    rival: Optional[RivalFeatures] = None
    gate_result: Optional[GateResult] = None
    planner_result: Optional[PlannerResult] = None
    horizon_result: Optional[HorizonResult] = None
    confidence_result: Optional[ConfidenceGateResult] = None

    # planner instance retained so the confidence gate can pull raw MC samples
    _planner: object = None

    # fusion outputs
    final_mode: Optional[str] = None
    action: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)

    @property
    def seed(self) -> int:
        return self.config.seed
