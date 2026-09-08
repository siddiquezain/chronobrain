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

from app.data.quality import DataQuality
from app.data.samples import NormalizedLap
from app.decision.config import DecisionConfig
from app.decision.window import WindowFeatures


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
    freshness_laps: int = 0         # laps since the last usable rival observation
    p_defend: float = 0.0           # deterministic scalar: P(rival actively defends)


@dataclass
class DecisionContext:
    # inputs
    config: DecisionConfig
    lap_history: List[NormalizedLap]           # laps 1..N (N = target lap)
    target_lap: NormalizedLap

    # threaded facts
    overtake_bonus_banked: bool = False

    # stage outputs (filled in order)
    data_quality: Optional[DataQuality] = None
    window: Optional[WindowFeatures] = None
    energy: Optional[EnergyFeatures] = None
    ml_overtake_prob: Optional[float] = None
    rival: Optional[RivalFeatures] = None
    gate_result: Optional[GateResult] = None
    candidate_actions: List[str] = field(default_factory=list)
    feasible_actions: List[str] = field(default_factory=list)
    rejected_alternatives: List[dict] = field(default_factory=list)  # [{action, reason}]
    planner_result: Optional[PlannerResult] = None
    horizon_result: Optional[HorizonResult] = None
    opportunity_uncertain: bool = False
    prefers_wait: bool = False
    wait_n: int = 0
    confidence_result: Optional[ConfidenceGateResult] = None

    # instances retained for diagnostics: the confidence gate pulls raw MC samples
    # from the planner; the demo/validation layer reads the estimator's posterior.
    _planner: object = None
    _estimator: object = None                    # the CURRENT rival's filter
    _estimators: object = None                   # {driver: filter} — identity-aware bank

    # fusion outputs
    final_mode: Optional[str] = None
    action: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)

    @property
    def seed(self) -> int:
        return self.config.seed
