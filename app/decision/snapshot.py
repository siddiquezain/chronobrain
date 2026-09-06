"""
snapshot.py — the verified decision contract.

`DecisionSnapshot` is what `run_decision()` returns and what the LLM narrator is
handed. Every number here is produced by deterministic Python upstream. The
`narrative` field is filled in later (by `app.narrative`) and is the ONLY field a
language model may write — it can never alter any other field.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DecisionBlock(BaseModel):
    mode: str = Field(..., description="One of the 5 canonical DeploymentMode enum values")
    action: str = Field(..., description="ATTACK_NOW | WAIT_2_LAPS | WAIT_5_LAPS | HOLD | PUSH | CONSERVE")
    confidence: float = Field(..., ge=0.0, le=1.0)
    stage2_mode: str = Field(..., description="Planner's top pick before the confidence gate")
    confidence_overridden: bool
    override_reason: str = ""


class EnergyBlock(BaseModel):
    soc_mj: Optional[float]
    soc_pct: Optional[float]
    lap_start_soc_mj: Optional[float]
    deployed_this_lap_mj: float
    deployment_headroom_mj: float
    projected_reserve_mj: float
    projected_end_of_race_mj: float
    can_afford_aggressive: bool
    energy_is_modeled: bool = Field(
        ..., description="True when SoC came from a model, not the feed (always True for REPLAY)"
    )


class RivalBlock(BaseModel):
    mean_reserve_mj: float
    reserve_std_mj: float
    n_observations: int
    confidence: float = Field(..., ge=0.0, le=1.0, description="1 - normalized posterior std")
    estimate_uncertain: bool
    bucket: str = Field(..., description="LOW | MEDIUM | HIGH — argmax of the distribution")
    distribution: dict = Field(..., description="{'low': p, 'medium': p, 'high': p}, sums to 1")


class OpportunityStrategy(BaseModel):
    name: str
    delay_laps: int
    mean_horizon_delta_s: float
    std_horizon_delta_s: float
    ci_lower_s: float


class OpportunityBlock(BaseModel):
    recommended_strategy: str
    prefers_wait: bool
    foregone_strategy: str
    foregone_value_gap_s: float
    current_window_overtake_prob: float
    projected_window_overtake_prob: float
    projected_window_lap: Optional[int]
    ranked_strategies: List[OpportunityStrategy]
    uncertainty_note: str


class MonteCarloMode(BaseModel):
    mode: str
    mean_laptime_delta_s: float
    std_laptime_delta_s: float
    overtake_probability: float
    sharpe_ratio: float
    energy_cost_mj: float


class MonteCarloBlock(BaseModel):
    n_iterations: int
    seed: int
    recommended_mode: str
    ranked_modes: List[MonteCarloMode]


class ComplianceCheck(BaseModel):
    rule: str
    provenance: str = Field(..., description="VERIFIED_FIA | MODEL_ASSUMPTION | DEMO_CONSTANT")
    status: str = Field(..., description="pass | breach | info")
    detail: str = ""


class ComplianceBlock(BaseModel):
    legal: bool
    legal_modes: List[str]
    illegal_modes: List[str]
    checks: List[ComplianceCheck]


class ConfidenceBlock(BaseModel):
    overall: float = Field(..., ge=0.0, le=1.0)
    statistical_reliability_passed: bool
    practical_significance_passed: bool
    dcli_passed: bool
    rival_confidence_passed: bool
    ci_lower_bound_s: float
    t_statistic: float
    dcli_score: float


class TraceStep(BaseModel):
    stage: str
    detail: str


class SnapshotMeta(BaseModel):
    lap: int
    total_laps: int
    data_mode: str
    source_detail: str
    seed: int
    generated_at: str = Field(..., description="Wall-clock only — excluded from determinism checks")


class DecisionSnapshot(BaseModel):
    """One authoritative decision snapshot. `narrative` is populated separately."""

    meta: SnapshotMeta
    decision: DecisionBlock
    energy: EnergyBlock
    rival: RivalBlock
    opportunity: OpportunityBlock
    monte_carlo: MonteCarloBlock
    compliance: ComplianceBlock
    confidence: ConfidenceBlock
    reason_codes: List[str]
    reasons: List[str] = Field(..., description="Human strings for reason_codes, same order")
    trace: List[TraceStep]
    narrative: Optional[str] = Field(
        None, description="LLM- or template-generated prose. Explanation only; never authoritative."
    )

    def deterministic_dict(self) -> dict:
        """Snapshot as a dict with the wall-clock field blanked — for golden/equality tests."""
        d = self.model_dump()
        d["meta"]["generated_at"] = "<omitted>"
        d["narrative"] = None
        return d
