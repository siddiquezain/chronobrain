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
    # --- additive; None on paths that don't model the accounting components ---
    soc_capacity_mj: float = Field(9.0, description="SoC ceiling for the scale soc_mj is on (MODEL_ASSUMPTION)")
    recovered_this_lap_mj: Optional[float] = Field(
        None, description="Modeled ERS recovery this lap (MJ). MODEL_ASSUMPTION."
    )
    net_swing_mj: Optional[float] = Field(
        None, description="Modeled net SoC change this lap = recovered - deployed (vs a nominal lap)"
    )
    modeled_mgu_k_peak_kw: Optional[float] = Field(
        None, description="Modeled peak MGU-K power this lap (kW) from the real throttle trace. MODEL_ASSUMPTION."
    )
    mgu_k_power_ceiling_kw: float = Field(350.0, description="Regulatory ceiling (VERIFIED_FIA)")
    energy_provenance: str = Field(
        "MODELED",
        description=(
            "MEASURED: directly from telemetry feed. "
            "INFERRED: probabilistic hidden-state estimate. "
            "MODELED: ChronoPace simulation/reconstruction (NOT actual FIA battery SoC)."
        ),
    )


class RivalBlock(BaseModel):
    mean_reserve_mj: float
    reserve_std_mj: float
    n_observations: int
    confidence: float = Field(..., ge=0.0, le=1.0, description="1 - normalized posterior std")
    estimate_uncertain: bool
    bucket: str = Field(..., description="LOW | MEDIUM | HIGH — argmax of the distribution")
    distribution: dict = Field(..., description="{'low': p, 'medium': p, 'high': p}, sums to 1")
    freshness_laps: int = Field(0, ge=0, description="Laps since the last usable rival observation")
    p_defend: float = Field(0.0, ge=0.0, le=1.0, description="Deterministic P(rival actively defends)")

    # --- posterior health (from RivalSocEstimate) ---
    effective_sample_size: float = Field(
        0.0, ge=0.0,
        description="Particle ESS: 1/Σw². Low = collapsed/degenerate posterior."
    )
    evidence_quality: str = Field(
        "insufficient",
        description="insufficient | weak | moderate | strong — NOT an accuracy claim."
    )
    posterior_health: str = Field(
        "unknown",
        description="healthy | collapsed | roughened | insufficient_data"
    )
    baseline_ready: bool = Field(
        False,
        description="True once the Z-score baseline has min_obs observations."
    )
    energy_provenance: str = Field(
        "INFERRED",
        description="Always INFERRED for rival estimates — particle-filter posterior.",
    )

    # --- dynamic strategic-rival selection (additive; None for the fixed two-car
    #     replay and the synthetic path — the energy estimate above is then for the
    #     single configured rival exactly as before) ---
    driver: Optional[str] = Field(
        None, description="Whose observable performance the estimate above is inferred from"
    )
    role: Optional[str] = Field(
        None, description="ATTACK_TARGET | DEFENDING_THREAT | POSITION_BATTLE | STRATEGICALLY_RELEVANT | NONE"
    )
    strategic_position: Optional[int] = Field(None, ge=1, description="Rival's running position this lap")
    strategic_gap_s: Optional[float] = Field(None, ge=0.0, description="Gap to the strategic rival (unsigned)")
    strategic_rival_ahead: Optional[bool] = Field(None, description="True = rival ahead of us")
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="How strategically relevant this opponent is right now"
    )


class DataQualityBlock(BaseModel):
    status: str = Field(..., description="GOOD | DEGRADED | INVALID")
    quality_score: float = Field(..., ge=0.0, le=1.0)
    freshness_laps: int
    dropped_samples: int
    out_of_order: bool
    missing_fields: List[str]
    checks: List[str]


class WindowBlock(BaseModel):
    n_laps: int
    speed_trend_kmh_per_lap: float
    gap_ahead_trend_s_per_lap: Optional[float]
    soc_trend_mj_per_lap: Optional[float]
    rival_terminal_speed_trend: Optional[float]
    rival_sector_delta_trend: Optional[float]
    closing: bool
    opportunity_trend: str = Field(..., description="IMPROVING | STABLE | DECAYING")


class RejectedAlternative(BaseModel):
    action: str
    reason: str


class ConstraintsBlock(BaseModel):
    regulatory: str = Field(..., description="PASS | FAIL")
    data_quality: str = Field(..., description="GOOD | DEGRADED | INVALID")
    energy: str = Field(..., description="OK | RESERVE_LOW")


class OpportunityStrategy(BaseModel):
    name: str
    delay_laps: int
    mean_horizon_delta_s: float
    std_horizon_delta_s: float
    ci_lower_s: float
    end_soc_mj: float = 0.0
    energy_spent_mj: float = 0.0
    current_opportunity_value: float = 0.0
    future_opportunity_value: float = 0.0
    energy_opportunity_cost: float = Field(
        0.0, description="Function of THIS strategy's own spent/end_soc only — "
        "never another strategy's opportunity value (no double-counting)."
    )
    strategic_value: float = 0.0
    attack_completion_probability: Optional[float] = Field(
        None, description="P(the completion-lap overtake succeeds), from a per-iteration "
        "rival-SoC draw. None if this strategy never reaches a completion lap or no "
        "rival estimate was available."
    )
    attack_completion_probability_std: Optional[float] = Field(
        None, description="Std of the per-iteration completion probability — the "
        "undiluted signal of rival uncertainty (scales with rival std_soc_mj)."
    )
    utility_std: float = Field(
        0.0, description="Std of strategic_value across iterations — widens with rival "
        "uncertainty even when the mean strategic_value does not."
    )
    downside_probability: float = Field(
        0.0, description="P(this strategy's strategic_value < HOLD's). 0.0 for HOLD."
    )


class OpportunityBlock(BaseModel):
    recommended_strategy: str
    prefers_wait: bool
    foregone_strategy: str
    foregone_value_gap_s: float
    future_energy_value_active: bool = False
    opportunity_uncertain: bool = False
    opportunity_trend: str = "STABLE"
    current_window_overtake_prob: float
    projected_window_overtake_prob: float
    projected_window_lap: Optional[int]
    ranked_strategies: List[OpportunityStrategy]
    uncertainty_note: str


class MonteCarloMode(BaseModel):
    mode: str
    mean_laptime_delta_s: float
    std_laptime_delta_s: float
    overtake_probability: float = Field(
        ..., description="LEGACY NAME, kept for API compatibility: mean(samples < 0) "
        "= P(this mode nets faster than BALANCED), NOT literally P(overtake completes). "
        "~0.5 for CONSERVE/BALANCED (no attempt modelled). Use "
        "attack_completion_probability for the correctly-named metric."
    )
    attack_completion_probability: Optional[float] = Field(
        None, description="P(the simulated overtake attempt itself succeeds). None for "
        "CONSERVE/BALANCED (no attempt is modelled)."
    )
    sharpe_ratio: float
    energy_cost_mj: float


class MonteCarloBlock(BaseModel):
    n_iterations: int
    seed: int
    recommended_mode: str
    ranked_modes: List[MonteCarloMode]
    runner_up_mode: Optional[str] = Field(
        None, description="Second-ranked legal mode, if any — counterfactual companion "
        "to recommended_mode."
    )
    mode_value_gap_s: float = Field(
        0.0, description="How much better recommended_mode is than runner_up_mode (s)."
    )


class CounterfactualBlock(BaseModel):
    """What would happen if we did NOT choose the recommended action."""
    recommended_action: str = Field(..., description="The action the engine recommends")
    counterfactual_action: str = Field(..., description="The next-best alternative (what we forego)")
    recommended_expected_gain_s: float = Field(
        ..., description="Expected laptime delta of the recommended mode (s, negative = faster than BALANCED baseline)"
    )
    counterfactual_expected_gain_s: float = Field(
        ..., description="Expected laptime delta of the foregone mode (s)"
    )
    gain_delta_s: float = Field(
        ..., description="How much better the recommended mode is vs. the counterfactual (s, >= 0)"
    )
    energy_cost_recommended_mj: float = Field(
        0.0, description="Energy cost of the recommended mode (MJ)"
    )
    energy_cost_counterfactual_mj: float = Field(
        0.0, description="Energy cost of the foregone mode (MJ)"
    )
    future_opportunity_impact: str = Field(
        "SIMILAR",
        description="BETTER | SIMILAR | WORSE — how choosing the counterfactual affects future windows"
    )
    summary: str = Field(
        "", description="One-line human explanation of why the recommendation beats the alternative"
    )


class ContextAttributionBlock(BaseModel):
    """
    Tyre context and active aero attribution for the rival's observed pace delta.

    The compound-stratified Z-score baseline (rival_estimator.py) removes compound-
    specific performance differences before feeding the particle filter. This block
    surfaces that attribution explicitly.

    IMPORTANT: ChronoPace does NOT model full tyre degradation curves. Tyre context
    is handled by Z-score normalization against the rival's own compound-specific
    baseline. No absolute 'tyre-explained seconds' figure is computed.
    """
    rival_tyre_compound: Optional[str] = Field(
        None, description="Rival's tyre compound this lap (SOFT/MEDIUM/HARD/UNKNOWN). "
        "None when unavailable (synthetic mode)."
    )
    compound_baseline_active: bool = Field(
        False,
        description="True when the compound-specific Z-score baseline has >= 5 laps on "
        "this compound, enabling compound-stratified normalisation."
    )
    observed_sector_delta_s: Optional[float] = Field(
        None, description="Rival's raw sector delta vs. their own running baseline (s). "
        "Negative = rival faster than their own average."
    )
    context_explained_note: str = Field(
        "",
        description="Human-readable note on how tyre context was handled."
    )
    active_aero_mode: str = Field(
        "UNKNOWN",
        description="OVERTAKE_ELIGIBLE | STRAIGHT_MODE | CORNER_MODE | UNKNOWN — "
        "our car's active aero state inferred from gap and speed."
    )
    residual_evidence_confidence: str = Field(
        "UNAVAILABLE",
        description="HIGH | MEDIUM | LOW | UNAVAILABLE — confidence that the residual "
        "(after context attribution) reflects rival energy management."
    )


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
    data_quality_passed: bool = True
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
    pipeline_version: str = "2.0"
    config_fingerprint: str = Field(
        "", description="Hash of DecisionConfig + pipeline version + ML model — same value ⇒ reproducible"
    )
    generated_at: str = Field(..., description="Wall-clock only — excluded from determinism checks")


class DecisionSnapshot(BaseModel):
    """One authoritative decision snapshot. `narrative` is populated separately."""

    meta: SnapshotMeta
    decision: DecisionBlock
    data_quality: DataQualityBlock
    window: WindowBlock
    energy: EnergyBlock
    rival: RivalBlock
    opportunity: OpportunityBlock
    monte_carlo: MonteCarloBlock
    compliance: ComplianceBlock
    confidence: ConfidenceBlock
    constraints: ConstraintsBlock
    counterfactual: Optional[CounterfactualBlock] = Field(
        None, description="What would happen if the recommended action is NOT taken. "
        "None when there is only one feasible mode (no comparison possible)."
    )
    context_attribution: Optional[ContextAttributionBlock] = Field(
        None, description="Tyre compound context and active aero attribution for rival pace delta."
    )
    candidate_actions: List[str]
    feasible_actions: List[str]
    rejected_alternatives: List[RejectedAlternative]
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
