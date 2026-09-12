"""
engine.py — run_decision(): the one deterministic entrypoint.

Pipeline order (each stage writes into DecisionContext):

    laps 1..N  ->  thread bonus qualification + rival particle filter
                ->  feature extraction (energy, ML P(overtake success))
                ->  regulatory gate            (Stage 1, legality only)
                ->  Monte Carlo planner        (Stage 2, ranks legal modes)
                ->  opportunity horizon        (ATTACK_NOW vs WAIT_N vs HOLD)
                ->  confidence gate            (Stage 3, may override to BALANCED)
                ->  DECISION ENGINE fusion     (final mode + action + reason codes)
                ->  DecisionSnapshot           (verified — the LLM sees only this)
                ->  DecisionTrace

No language model is called anywhere in this module. Randomness is confined to the
seeded planner / horizon / particle filter; everything else is a pure function of
the inputs. Replaying laps 1..N each call (rather than caching mutable state) is
what makes "same laps + same config + same seed => identical snapshot" true.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from confidence_gate import ConfidenceGate, DriverLoadInput
from opportunity_engine import OpportunityEngine
from planner import MonteCarloPlanner, PlanningContext
from rival_estimator import RivalStateEstimator
from rule_gate import DeploymentMode, RegulatoryGate

from app.data.normalizer import to_rival_observation, to_telemetry_input
from app.data.providers import TelemetryProvider, build_provider
from app.data.quality import assess_quality
from app.data.samples import NormalizedLap
from app.decision.actions import (
    CANDIDATE_DELAY,
    CandidateAction,
    FeasibilityInput,
    feasible_actions,
)
from app.decision.config import DecisionConfig
from app.decision.context import DecisionContext, EnergyFeatures, RivalFeatures
from app.decision.window import compute_window
from app.decision import reason_codes as rc
from app.decision.snapshot import (
    ComplianceBlock,
    ComplianceCheck,
    ConfidenceBlock,
    ConstraintsBlock,
    DataQualityBlock,
    DecisionBlock,
    DecisionSnapshot,
    EnergyBlock,
    MonteCarloBlock,
    MonteCarloMode,
    OpportunityBlock,
    OpportunityStrategy,
    RejectedAlternative,
    RivalBlock,
    SnapshotMeta,
    WindowBlock,
)
from app.decision.trace import build_trace
from app.regulation.constants import REGULATORY_CONSTANTS
from app.narrative.narrator import narrate as _narrate
from app.ml.rival_observation_model import RivalObservationModel as _RivalObsModel

PIPELINE_VERSION = "2.0"  # bumped: data-quality gate, event-time window, FEV, feasible set

_rival_obs_model: Optional["_RivalObsModel"] = None
_rival_obs_model_loaded: bool = False


def _get_rival_obs_model() -> Optional["_RivalObsModel"]:
    """Load rival observation model once per process. None = Gaussian fallback active."""
    global _rival_obs_model, _rival_obs_model_loaded
    if not _rival_obs_model_loaded:
        _rival_obs_model = _RivalObsModel.load_or_none()
        _rival_obs_model_loaded = True
    return _rival_obs_model

_AGGRESSIVE = {
    DeploymentMode.ARM_OVERTAKE_MODE,
    DeploymentMode.USE_OVERTAKE_BONUS_MODE,
    DeploymentMode.PUSH_MODE,
}
_SOC_CEIL_MJ = 9.0
_DEFEND_GAP_S = 1.0  # MODEL_ASSUMPTION: rearward gap under which PUSH-to-defend is justified
# MODEL_ASSUMPTION: net per-lap SoC balance under normal (non-attacking) running.
# Near-neutral — a car not spending its bonus roughly harvests what it deploys.
_NET_SOC_PER_LAP_MJ = -0.05
_MIN_RESERVE_MJ = 1.0
# MODEL_ASSUMPTION: mean throttle fraction at/above which the modelled MGU-K peak
# reaches its regulatory ceiling. A full-racing lap deploys at the ceiling on the
# straights; only a genuine lift-and-coast / conserve lap pulls the modelled peak
# below it. This is a modelled value from the REAL throttle trace, never measured.
_MGU_K_FULL_POWER_THROTTLE = 0.70


# ===========================================================================
# public API
# ===========================================================================
def run_decision(
    provider: TelemetryProvider,
    *,
    lap: Optional[int] = None,
    config: Optional[DecisionConfig] = None,
    with_narrative: bool = False,
) -> DecisionSnapshot:
    """
    Run the full pipeline for one lap of `provider` and return a verified snapshot.

    `lap` defaults to the provider's last lap. `with_narrative=True` additionally
    fills `snapshot.narrative` via app.narrative (never changes any other field).
    """
    cfg = config or DecisionConfig()
    ctx = run_pipeline(provider, lap=lap, config=cfg)
    snapshot = _assemble(ctx, provider.describe())
    if with_narrative:
        snapshot.narrative = _narrate(snapshot)
    return snapshot


def run_pipeline(
    provider: TelemetryProvider,
    *,
    lap: Optional[int] = None,
    config: Optional[DecisionConfig] = None,
) -> DecisionContext:
    """
    Run every stage and return the populated `DecisionContext` WITHOUT assembling
    the snapshot. `run_decision()` wraps this; the demo/validation layer uses it to
    also read `ctx._estimator` / `ctx.rival`. Same determinism guarantees.
    """
    cfg = config or DecisionConfig()
    all_laps = provider.laps()
    if not all_laps:
        raise ValueError("provider yielded no laps")

    target_lap_no = lap if lap is not None else all_laps[-1].lap
    history = [nl for nl in all_laps if nl.lap <= target_lap_no]
    if not history or history[-1].lap != target_lap_no:
        raise ValueError(f"provider has no lap {target_lap_no}")

    ctx = DecisionContext(config=cfg, lap_history=history, target_lap=history[-1])

    _run_data_quality(ctx)      # Data Quality Gate (before anything else)
    _run_window(ctx)            # event-time sliding window -> trends
    _thread_state(ctx)          # bonus qualification + rival particle filter over laps 1..N
    _extract_features(ctx)      # energy + ML P(overtake success)
    _run_gate(ctx)              # Stage 1 — legality
    _run_feasibility(ctx)       # candidate actions -> feasible set
    _run_planner(ctx)           # Stage 2 — Monte Carlo over legal modes
    _run_horizon(ctx)           # opportunity horizon + Future Energy Value
    _run_confidence(ctx)        # Stage 3 — significance + DCLI + rival + data quality
    _fuse(ctx)                  # decision engine — pick from the feasible set
    return ctx


def assemble_snapshot(ctx: DecisionContext, source_detail: str = "") -> DecisionSnapshot:
    """Public wrapper around the snapshot assembler — used by the demo layer,
    which also needs the raw `ctx`."""
    return _assemble(ctx, source_detail)


def run_scenario(
    scenario: str = "B",
    *,
    seed: int = 42,
    lap: Optional[int] = None,
    total_laps: int = 50,
    config: Optional[DecisionConfig] = None,
    with_narrative: bool = False,
) -> DecisionSnapshot:
    """Convenience wrapper: synthetic provider for `scenario` -> run_decision()."""
    cfg = config or DecisionConfig(seed=seed)
    provider = build_provider(
        "synthetic", scenario=scenario, seed=cfg.seed, total_laps=total_laps
    )
    return run_decision(provider, lap=lap, config=cfg, with_narrative=with_narrative)


# ===========================================================================
# stages
# ===========================================================================
def _run_data_quality(ctx: DecisionContext) -> None:
    ctx.data_quality = assess_quality(
        ctx.lap_history, ctx.target_lap, ctx.target_lap.data_mode
    )


def _run_window(ctx: DecisionContext) -> None:
    ctx.window = compute_window(ctx.lap_history, window=ctx.config.window_laps)


def _driver_seed_offset(driver: str) -> int:
    """Deterministic, process-independent per-driver seed offset — so two tracked
    rivals get distinct particle clouds without any RNG salt."""
    import zlib
    return int(zlib.adler32(driver.encode("utf-8")) % 100_000)


def _thread_state(ctx: DecisionContext) -> None:
    """
    Walk laps 1..N. `overtake_bonus_banked` for lap N is set by lap N-1's gate
    (qualification is a telemetry fact, not a mode choice).

    IDENTITY-AWARE rival energy: one particle filter PER tracked driver. Each
    filter only ever sees that driver's own observations, in lap order, censored
    to laps <= N. When the strategic rival switches away and later comes back, its
    filter *resumes* from where it left off (we do have prior evidence about a car
    we watched before) — it is not restarted from scratch, and it never sees
    another driver's observations. Synthetic / fixed two-car replay carry
    `strategic_rival=None` and use a single default filter, exactly as before.
    """
    cfg = ctx.config
    gate = RegulatoryGate(config=cfg.gate_config())

    obs_model = _get_rival_obs_model()
    default_est = RivalStateEstimator(config=cfg.rival_config(), seed=cfg.seed, obs_model=obs_model)
    estimators: dict[str, RivalStateEstimator] = {}

    def _est_for(driver: Optional[str]) -> RivalStateEstimator:
        if driver is None:
            return default_est
        if driver not in estimators:
            estimators[driver] = RivalStateEstimator(
                config=cfg.rival_config(),
                seed=cfg.seed + _driver_seed_offset(driver),
                obs_model=obs_model,
            )
        return estimators[driver]

    banked = False
    current_est = default_est
    for i, nl in enumerate(ctx.lap_history):
        ti = to_telemetry_input(nl, overtake_qualified_last_lap=banked)
        gr = gate.evaluate(ti)

        rival_id = nl.strategic_rival.driver if nl.strategic_rival is not None else None
        est = _est_for(rival_id)

        obs = to_rival_observation(nl)
        if obs is not None:
            est.predict()
            est.update(obs, compound=nl.rival_compound or "UNKNOWN")
        elif est.observation_count > 0:
            # rival lap was pit / out / invalid — a lap passed but there is no
            # clean observation. Advance the drift (uncertainty grows) but do NOT
            # update: a +20 s pit-lap delta must never collapse the posterior.
            est.predict()
        current_est = est

        if i == len(ctx.lap_history) - 1:
            ctx.overtake_bonus_banked = banked
            ctx.gate_result = gr  # provisional; re-evaluated in _run_gate for clarity
        banked = gr.qualifies_for_overtake_bonus_next_lap

    ctx._estimator = current_est
    ctx._estimators = estimators
    ctx.rival = _rival_features(ctx, current_est)


def _rival_features(ctx: DecisionContext, estimator: RivalStateEstimator) -> Optional[RivalFeatures]:
    if not any(nl.has_rival_observation for nl in ctx.lap_history):
        return None
    est = estimator.estimate()
    cfg = ctx.config

    lo, hi = cfg.rival_low_soc_mj, cfg.rival_high_soc_mj
    dist = est.bucket_distribution(lo, hi)
    bucket = est.bucket(lo, hi)

    threshold = cfg.confidence_config().rival_confidence_threshold_mj
    uncertain = est.std_soc_mj > threshold
    confidence = float(np.clip(1.0 - est.std_soc_mj / 2.6, 0.0, 1.0))

    freshness = ctx.window.stale_laps if ctx.window is not None else 0
    p_defend = _p_defend(est.mean_soc_mj, uncertain, ctx.window)

    return RivalFeatures(
        estimate=est, bucket=bucket, distribution=dist,
        confidence=round(confidence, 4), uncertain=uncertain,
        freshness_laps=freshness, p_defend=p_defend,
    )


def _p_defend(rival_mean_soc_mj: float, uncertain: bool, window) -> float:
    """
    Deterministic scalar P(rival actively defends). A rival with more energy can
    defend harder; a rival being caught is more likely to; an uncertain estimate
    is pulled toward a neutral 0.4. Small statistical input to the planner — NOT a
    behavioural model and NOT a replacement for the rival-energy estimator.
    """
    soc_frac = float(np.clip(rival_mean_soc_mj / _SOC_CEIL_MJ, 0.0, 1.0))
    base = 0.20 + 0.45 * soc_frac
    if window is not None and getattr(window, "closing", False):
        base += 0.15
    if uncertain:
        base = 0.5 * base + 0.5 * 0.4
    return round(float(np.clip(base, 0.0, 0.9)), 4)


def _modeled_mgu_k_peak_kw(nl: NormalizedLap, ceiling_kw: float) -> Optional[float]:
    """Modelled peak MGU-K electrical power for the lap, from the REAL throttle
    trace. MODEL_ASSUMPTION — clamped to the regulatory ceiling by construction, so
    the compliance check is a genuine (if trivially-satisfied) verification, not a
    hardcoded string."""
    thr = getattr(nl, "our_mean_throttle", None)
    if thr is None or getattr(nl, "lap_status", "racing") != "racing":
        return None
    frac = min(1.0, max(0.0, thr) / _MGU_K_FULL_POWER_THROTTLE)
    return round(min(ceiling_kw, ceiling_kw * frac), 1)


def _extract_features(ctx: DecisionContext) -> None:
    nl = ctx.target_lap
    cfg = ctx.config
    ti = to_telemetry_input(nl, overtake_qualified_last_lap=ctx.overtake_bonus_banked)

    soc = ti.current_soc_mj
    capacity = float(getattr(nl, "our_soc_capacity_mj", None) or _SOC_CEIL_MJ)
    laps_remaining = max(0, nl.total_laps - nl.lap)
    headroom = _deployment_headroom(ti.lap_energy_deployed_mj, ctx.overtake_bonus_banked, cfg)
    projected = float(np.clip(soc + laps_remaining * _NET_SOC_PER_LAP_MJ, 0.0, _SOC_CEIL_MJ))
    end_of_race = projected
    can_afford = soc >= 3.0 and headroom >= 1.5 and projected >= _MIN_RESERVE_MJ
    reserve_low = projected < cfg.low_reserve_mj or soc < cfg.low_reserve_mj

    ceiling_kw = cfg.gate_config().max_ers_k_power_kw
    mgu_k_peak_kw = _modeled_mgu_k_peak_kw(nl, ceiling_kw)

    ctx.energy = EnergyFeatures(
        soc_mj=round(soc, 3),
        soc_pct=round(soc / capacity * 100.0, 2),
        soc_capacity_mj=round(capacity, 3),
        lap_start_soc_mj=ti.lap_start_soc_mj,
        deployed_this_lap_mj=round(ti.lap_energy_deployed_mj, 3),
        recovered_this_lap_mj=(round(nl.our_lap_energy_recovered_mj, 3)
                               if nl.our_lap_energy_recovered_mj is not None else None),
        net_swing_mj=(round(nl.our_lap_net_swing_mj, 4)
                      if nl.our_lap_net_swing_mj is not None else None),
        modeled_mgu_k_peak_kw=mgu_k_peak_kw,
        mgu_k_power_ceiling_kw=round(ceiling_kw, 1),
        deployment_headroom_mj=round(headroom, 3),
        projected_reserve_mj=round(projected, 3),
        projected_end_of_race_mj=round(end_of_race, 3),
        can_afford_aggressive=bool(can_afford),
        reserve_low=bool(reserve_low),
        energy_is_modeled=nl.energy_is_modeled,
    )
    ctx.ml_overtake_prob = _ml_overtake_probability(nl, soc, ctx.window)


def _run_gate(ctx: DecisionContext) -> None:
    ti = to_telemetry_input(ctx.target_lap, overtake_qualified_last_lap=ctx.overtake_bonus_banked)
    ctx.gate_result = RegulatoryGate(config=ctx.config.gate_config()).evaluate(ti)


def _run_feasibility(ctx: DecisionContext) -> None:
    """Candidate strategic actions -> feasible set (legality + energy), before the planner."""
    e = ctx.energy
    fin = FeasibilityInput(
        gate_result=ctx.gate_result,
        soc_mj=e.soc_mj if e else 0.0,
        projected_reserve_mj=e.projected_reserve_mj if e else 0.0,
        can_afford_aggressive=e.can_afford_aggressive if e else False,
        reserve_floor_mj=ctx.config.reserve_floor_mj,
        laps_remaining=max(0, ctx.target_lap.total_laps - ctx.target_lap.lap),
    )
    feasible, rejected = feasible_actions(fin)
    ctx.candidate_actions = [a.value for a in CandidateAction]
    ctx.feasible_actions = [a.value for a in feasible]
    ctx.rejected_alternatives = [{"action": a, "reason": r} for a, r in rejected]


def _run_planner(ctx: DecisionContext) -> None:
    cfg = ctx.config
    planner = MonteCarloPlanner(config=cfg.planner_config(), seed=cfg.seed)
    pc = PlanningContext(
        gap_to_car_ahead_s=ctx.target_lap.gap_to_car_ahead_s,
        gap_to_car_behind_s=ctx.target_lap.gap_to_car_behind_s,
        rival_soc_estimate=ctx.rival.estimate if ctx.rival else None,
        p_defend=ctx.rival.p_defend if ctx.rival else None,
        own_soc_mj=ctx.energy.soc_mj if ctx.energy else None,
        low_reserve_mj=cfg.low_reserve_mj,
        opportunity_strength=ctx.ml_overtake_prob,
    )
    ctx.planner_result = planner.plan(ctx.gate_result, pc)
    ctx._planner = planner  # kept for the confidence gate (raw samples)


def _run_horizon(ctx: DecisionContext) -> None:
    cfg = ctx.config
    engine = OpportunityEngine(config=cfg.horizon_config(), seed=cfg.seed)
    ti = to_telemetry_input(ctx.target_lap, overtake_qualified_last_lap=ctx.overtake_bonus_banked)
    e = ctx.energy

    feasible_delays = {
        CANDIDATE_DELAY[CandidateAction(a)]
        for a in ctx.feasible_actions
        if a in {ca.value for ca in CANDIDATE_DELAY}
    }
    # ATTACK_NOW (delay 0) is always evaluated as the reference even when infeasible —
    # "would attacking now beat waiting?" is the whole question.
    restrict = feasible_delays | {0}
    improving = ctx.window is not None and ctx.window.opportunity_trend == "IMPROVING"

    ctx.horizon_result = engine.evaluate(
        ti,
        ctx.gate_result,
        ctx.rival.estimate if ctx.rival else None,
        window_strength_by_delay=_window_strength(ctx),
        start_soc_mj=e.soc_mj if e else None,
        laps_remaining=max(0, ti.total_laps - ti.lap_number),
        reserve_floor_mj=cfg.reserve_floor_mj,
        overtake_reward_s=cfg.overtake_reward_s,
        future_window_bias=cfg.future_window_bias_s if improving else 0.0,
        restrict_to_delays=restrict,
        # same deterministic P(rival defends) scalar Stage 2 already receives —
        # keeps the two Monte Carlo layers' rival-uncertainty inputs compatible.
        p_defend=ctx.rival.p_defend if ctx.rival else None,
    )
    ctx.opportunity_uncertain = _opportunity_uncertain(ctx)


def _window_strength(ctx: DecisionContext) -> dict:
    """
    How well does an overtake window hold up N laps from now?

    Base is mild decay (a bird in the hand). It decays LESS when the rival is
    confidently LOW (not recovering) and MORE when the estimate is uncertain or
    the rival has reserve. The event-time window then adjusts: an IMPROVING
    situation (gap closing, rival fading) can make future windows *stronger* than
    now (>1.0), a DECAYING one accelerates the decay. This is the path by which
    rival-state uncertainty AND short-horizon trends reach the timing decision.
    MODEL_ASSUMPTION.
    """
    per_lap = 0.92
    if ctx.rival is not None:
        if ctx.rival.bucket == "LOW" and not ctx.rival.uncertain:
            per_lap = 0.985
        elif ctx.rival.uncertain or ctx.rival.bucket == "HIGH":
            per_lap = 0.85
    if ctx.window is not None:
        if ctx.window.opportunity_trend == "IMPROVING":
            per_lap = min(1.03, per_lap + 0.05)
        elif ctx.window.opportunity_trend == "DECAYING":
            per_lap = max(0.82, per_lap - 0.08)
    return {d: round(per_lap ** d, 4) for d in ctx.config.delay_laps}


def _opportunity_uncertain(ctx: DecisionContext) -> bool:
    """
    INFORMATIONAL only (surfaced in the snapshot, NOT a confidence-gate input).

    Fires when the horizon's top strategy is a WAIT that beats ATTACK_NOW by less
    than the decisive margin AND that lead is within the sample noise — i.e. the
    horizon leans toward waiting but not convincingly. The decision engine already
    handles this correctly (it does not defer below the decisive margin), so this
    flag does not force an abstention; it is here so the frontend / trace can show
    "the timing call was close".
    """
    h = ctx.horizon_result
    if h is None or not h.future_energy_value_active or len(h.ranked_strategies) < 3:
        return False
    top = h.ranked_strategies[0]
    attack = next((s for s in h.ranked_strategies if s.strategy_name == "ATTACK_NOW"), None)
    if attack is None or not top.strategy_name.startswith("WAIT"):
        return False
    lead = top.strategic_value - attack.strategic_value
    return 0.0 < lead < ctx.config.horizon_decisive_margin_s and top.std_horizon_delta_s > 1.0


def _run_confidence(ctx: DecisionContext) -> None:
    gate = ConfidenceGate(config=ctx.config.confidence_config())
    load = DriverLoadInput(
        laps_since_last_mode_change=3.0,           # MODEL_ASSUMPTION: single-lap API has no history of mode changes
        recent_laptime_std_s=0.25,
        gap_to_car_ahead_s=ctx.target_lap.gap_to_car_ahead_s,
    )
    dq = ctx.data_quality.quality_score if ctx.data_quality else 1.0
    ctx.confidence_result = gate.evaluate(
        ctx.planner_result,
        driver_load=load,
        rival_estimate=ctx.rival.estimate if ctx.rival else None,
        planner=getattr(ctx, "_planner", None),
        data_quality_score=dq,
        # opportunity_uncertain is informational only (see _opportunity_uncertain) —
        # the decision engine's decisive-margin rule already handles a close call.
        opportunity_uncertain=False,
    )


# ===========================================================================
# decision engine (fusion) — the "which legal strategy do we actually pick"
# ===========================================================================
def _fuse(ctx: DecisionContext) -> None:
    cfg = ctx.config
    g = ctx.gate_result
    cgr = ctx.confidence_result
    h = ctx.horizon_result

    legal = list(g.legal_modes)
    all_illegal = len(legal) == 0
    cg_mode = cgr.recommended_mode
    cg_aggressive = cg_mode in _AGGRESSIVE

    can_afford = bool(ctx.energy and ctx.energy.can_afford_aggressive)
    reserve_low = bool(ctx.energy and ctx.energy.reserve_low)
    behind = ctx.target_lap.gap_to_car_behind_s
    defending = behind is not None and behind <= _DEFEND_GAP_S

    # A banked bonus + a model-confident window + the energy to use it is the moment
    # ChronoPace exists to catch — only defer it for a SUBSTANTIALLY better future
    # window (3x the normal decisive margin), never a marginal one.
    window_is_prime = (
        ctx.overtake_bonus_banked
        and can_afford
        and ctx.ml_overtake_prob is not None
        and ctx.ml_overtake_prob >= cfg.ml_prob_high
    )
    prefers_wait, wait_n = _horizon_prefers_wait(ctx, margin_mult=3.0 if window_is_prime else 1.0)
    ctx.prefers_wait, ctx.wait_n = prefers_wait, wait_n

    gap_ahead = ctx.target_lap.gap_to_car_ahead_s
    in_proximity = (
        gap_ahead is not None
        and gap_ahead <= cfg.gate_config().overtake_detection_gap_threshold_s
    )
    arm_legal = DeploymentMode.ARM_OVERTAKE_MODE in legal
    use_bonus_legal = DeploymentMode.USE_OVERTAKE_BONUS_MODE in legal

    def _hold_mode() -> DeploymentMode:
        if reserve_low and DeploymentMode.CONSERVE_MODE in legal:
            return DeploymentMode.CONSERVE_MODE
        return DeploymentMode.BALANCED_MODE

    # An overtake mode is only a real "attack now" if there is in fact a car ahead
    # to use it on. With directional strategic-rival selection the relevant rival
    # can be BEHIND us (gap_to_car_ahead_s is None) — spending a banked bonus on
    # empty track ahead is not an attack.
    has_target_ahead = gap_ahead is not None

    def _attack_mode() -> DeploymentMode:
        """
        'Attack now' resolves to a specific mode:
          - a car ahead + a banked bonus                   -> USE_OVERTAKE_BONUS_MODE
          - a car ahead in proximity, no bonus banked yet  -> ARM_OVERTAKE_MODE
            (attack this lap AND qualify next lap's bonus — the actual meaning of ARM)
          - otherwise fall back to the planner's aggressive pick (e.g. PUSH)
        """
        if use_bonus_legal and ctx.overtake_bonus_banked and has_target_ahead:
            return DeploymentMode.USE_OVERTAKE_BONUS_MODE
        if arm_legal and in_proximity and not ctx.overtake_bonus_banked:
            return DeploymentMode.ARM_OVERTAKE_MODE
        return cg_mode if cg_mode in legal else DeploymentMode.BALANCED_MODE

    if all_illegal:
        final_mode, action = DeploymentMode.BALANCED_MODE, "HOLD"
    elif cgr.overridden:
        final_mode, action = DeploymentMode.BALANCED_MODE, "HOLD"
    elif cg_aggressive and prefers_wait and f"WAIT_{wait_n}_LAPS" in ctx.feasible_actions:
        # horizon values a future window more than attacking now, AND that wait is
        # feasible (energy recovers during it) -> defer, deploy conservatively now
        final_mode = _hold_mode()
        action = f"WAIT_{wait_n}_LAPS"
    elif cg_aggressive and not can_afford:
        # "is the attack worth the energy?" — no, and no feasible wait either.
        final_mode = _hold_mode()
        action = "CONSERVE" if final_mode == DeploymentMode.CONSERVE_MODE else "HOLD"
    elif cg_aggressive and prefers_wait:
        # prefers a future window but the wait isn't feasible -> just hold
        final_mode = _hold_mode()
        action = "CONSERVE" if final_mode == DeploymentMode.CONSERVE_MODE else "HOLD"
    elif cg_aggressive:
        attack_mode = _attack_mode()
        if attack_mode == DeploymentMode.PUSH_MODE:
            # PUSH spends extra energy for pace with no bonus-qualification benefit.
            # Worth it only to defend a real threat behind; otherwise just hold.
            if defending:
                final_mode, action = DeploymentMode.PUSH_MODE, "PUSH"
            else:
                final_mode, action = DeploymentMode.BALANCED_MODE, "HOLD"
        else:
            final_mode, action = attack_mode, "ATTACK_NOW"  # ARM or USE_OVERTAKE_BONUS
    else:
        final_mode = cg_mode
        action = "CONSERVE" if cg_mode == DeploymentMode.CONSERVE_MODE else "HOLD"

    # guardrail: an illegal mode can never be the final decision
    if not all_illegal and final_mode not in legal:
        for fallback in (DeploymentMode.BALANCED_MODE, DeploymentMode.CONSERVE_MODE):
            if fallback in legal:
                final_mode = fallback
                break
        else:
            final_mode = legal[0]
        action = "HOLD"

    # guardrail: a WAIT action must correspond to a feasible candidate
    if action.startswith("WAIT_") and ctx.feasible_actions:
        want = f"WAIT_{wait_n}_LAPS"
        if want not in ctx.feasible_actions:
            final_mode, action = _hold_mode(), "HOLD"

    ctx.final_mode = final_mode.value
    ctx.action = action

    ri = rc.ReasonInputs(
        final_mode=final_mode.value,
        action=action,
        legal_modes=tuple(m.value for m in legal),
        all_modes_illegal=all_illegal,
        overtake_bonus_banked=ctx.overtake_bonus_banked,
        confidence_overridden=cgr.overridden,
        override_reason=cgr.override_reason,
        rival_bucket=ctx.rival.bucket if ctx.rival else None,
        rival_estimate_uncertain=ctx.rival.uncertain if ctx.rival else False,
        rival_p_defend=ctx.rival.p_defend if ctx.rival else 0.0,
        horizon_strategy=h.recommended_strategy,
        horizon_prefers_wait=prefers_wait,
        energy_reserve_low=ctx.energy.reserve_low if ctx.energy else False,
        can_afford_aggressive=ctx.energy.can_afford_aggressive if ctx.energy else False,
        ml_overtake_prob=ctx.ml_overtake_prob,
        ml_prob_high=cfg.ml_prob_high,
        ml_prob_low=cfg.ml_prob_low,
        data_quality_status=ctx.data_quality.status if ctx.data_quality else "GOOD",
        opportunity_ambiguous=ctx.opportunity_uncertain,
    )
    ctx.reason_codes = rc.select(ri)


def _horizon_prefers_wait(ctx: DecisionContext, margin_mult: float = 1.0) -> tuple[bool, int]:
    h = ctx.horizon_result
    if h is None or not h.recommended_strategy.startswith("WAIT"):
        return False, 0
    attack_now = next(
        (s for s in h.ranked_strategies if s.strategy_name == "ATTACK_NOW"), None
    )
    best = h.ranked_strategies[0]
    if attack_now is None:
        return False, 0
    if h.future_energy_value_active:
        # strategic_value: higher is better; best beats ATTACK_NOW by this much
        margin = best.strategic_value - attack_now.strategic_value
    else:
        margin = attack_now.mean_horizon_delta_s - best.mean_horizon_delta_s
    if margin < ctx.config.horizon_decisive_margin_s * margin_mult:
        return False, 0
    try:
        n = int(h.recommended_strategy.split("_")[1])
    except (IndexError, ValueError):
        n = 0
    return True, n


# ===========================================================================
# assemble snapshot
# ===========================================================================
def _assemble(ctx: DecisionContext, source_detail: str) -> DecisionSnapshot:
    nl = ctx.target_lap
    e = ctx.energy
    cgr = ctx.confidence_result
    pr = ctx.planner_result
    h = ctx.horizon_result

    from narrator import derive_confidence

    overall_conf = derive_confidence(cgr)

    energy_block = EnergyBlock(
        soc_mj=e.soc_mj, soc_pct=e.soc_pct, lap_start_soc_mj=e.lap_start_soc_mj,
        deployed_this_lap_mj=e.deployed_this_lap_mj,
        deployment_headroom_mj=e.deployment_headroom_mj,
        projected_reserve_mj=e.projected_reserve_mj,
        projected_end_of_race_mj=e.projected_end_of_race_mj,
        can_afford_aggressive=e.can_afford_aggressive,
        energy_is_modeled=e.energy_is_modeled,
        soc_capacity_mj=getattr(e, "soc_capacity_mj", 9.0),
        recovered_this_lap_mj=getattr(e, "recovered_this_lap_mj", None),
        net_swing_mj=getattr(e, "net_swing_mj", None),
        modeled_mgu_k_peak_kw=getattr(e, "modeled_mgu_k_peak_kw", None),
        mgu_k_power_ceiling_kw=getattr(e, "mgu_k_power_ceiling_kw", 350.0),
        energy_provenance="MODELED",
    )

    sr = ctx.target_lap.strategic_rival
    sr_fields = dict(
        driver=(sr.driver if sr else None),
        role=(sr.role if sr else None),
        strategic_position=(sr.position if sr else None),
        strategic_gap_s=(sr.gap_s if sr else None),
        strategic_rival_ahead=(sr.ahead if sr else None),
        relevance_score=(sr.relevance_score if sr else None),
    )

    if ctx.rival is not None:
        r = ctx.rival
        est = r.estimate
        rival_block = RivalBlock(
            mean_reserve_mj=est.mean_soc_mj,
            reserve_std_mj=est.std_soc_mj,
            n_observations=est.n_observations,
            confidence=r.confidence,
            estimate_uncertain=r.uncertain,
            bucket=r.bucket,
            distribution={k: round(v, 4) for k, v in r.distribution.items()},
            freshness_laps=r.freshness_laps,
            p_defend=r.p_defend,
            effective_sample_size=est.effective_sample_size,
            evidence_quality=est.evidence_quality,
            posterior_health=est.posterior_health,
            baseline_ready=est.baseline_ready,
            energy_provenance="INFERRED",
            **sr_fields,
        )
    else:
        rival_block = RivalBlock(
            mean_reserve_mj=0.0, reserve_std_mj=0.0, n_observations=0,
            confidence=0.0, estimate_uncertain=True, bucket="MEDIUM",
            distribution={"low": 0.0, "medium": 1.0, "high": 0.0},
            freshness_laps=len(ctx.lap_history), p_defend=0.0,
            effective_sample_size=0.0, evidence_quality="insufficient",
            posterior_health="insufficient_data", baseline_ready=False,
            energy_provenance="INFERRED",
            **sr_fields,
        )

    ranked = [
        OpportunityStrategy(
            name=s.strategy_name, delay_laps=s.delay_laps,
            mean_horizon_delta_s=s.mean_horizon_delta_s,
            std_horizon_delta_s=s.std_horizon_delta_s,
            ci_lower_s=s.confidence_ci_lower_s,
            end_soc_mj=s.end_soc_mj, energy_spent_mj=s.energy_spent_mj,
            current_opportunity_value=s.current_opportunity_value,
            future_opportunity_value=s.future_opportunity_value,
            energy_opportunity_cost=s.energy_opportunity_cost,
            strategic_value=s.strategic_value,
            attack_completion_probability=s.attack_completion_probability,
            attack_completion_probability_std=s.attack_completion_probability_std,
            utility_std=s.utility_std,
            downside_probability=s.downside_probability,
        )
        for s in h.ranked_strategies
    ]
    cur_prob = round(float(ctx.ml_overtake_prob or 0.0), 4)
    prefers_wait, wait_n = ctx.prefers_wait, ctx.wait_n
    proj_prob = round(min(1.0, cur_prob * 1.12), 4) if prefers_wait else cur_prob
    opportunity_block = OpportunityBlock(
        recommended_strategy=h.recommended_strategy,
        prefers_wait=prefers_wait,
        foregone_strategy=h.foregone_strategy,
        foregone_value_gap_s=h.foregone_value_gap_s,
        future_energy_value_active=h.future_energy_value_active,
        opportunity_uncertain=ctx.opportunity_uncertain,
        opportunity_trend=ctx.window.opportunity_trend if ctx.window else "STABLE",
        current_window_overtake_prob=cur_prob,
        projected_window_overtake_prob=proj_prob,
        projected_window_lap=(nl.lap + wait_n) if prefers_wait else None,
        ranked_strategies=ranked,
        uncertainty_note=h.uncertainty_note,
    )

    mc_block = MonteCarloBlock(
        n_iterations=pr.n_iterations,
        seed=ctx.seed,
        recommended_mode=pr.recommended_mode.value,
        ranked_modes=[
            MonteCarloMode(
                mode=m.mode.value,
                mean_laptime_delta_s=m.mean_laptime_delta_s,
                std_laptime_delta_s=m.std_laptime_delta_s,
                overtake_probability=m.overtake_probability,
                attack_completion_probability=m.attack_completion_probability,
                sharpe_ratio=m.sharpe_ratio,
                energy_cost_mj=m.energy_cost_mj,
            )
            for m in pr.ranked_modes
        ],
        runner_up_mode=(pr.runner_up_mode.value if pr.runner_up_mode else None),
        mode_value_gap_s=pr.mode_value_gap_s,
    )

    compliance_block = _compliance_block(ctx)

    confidence_block = ConfidenceBlock(
        overall=overall_conf,
        statistical_reliability_passed=cgr.statistical_reliability_passed,
        practical_significance_passed=cgr.practical_significance_passed,
        dcli_passed=cgr.dcli_passed,
        rival_confidence_passed=cgr.rival_confidence_passed,
        data_quality_passed=getattr(cgr, "data_quality_passed", True),
        ci_lower_bound_s=cgr.ci_lower_bound_s,
        t_statistic=cgr.t_statistic,
        dcli_score=cgr.dcli_score,
    )

    dq = ctx.data_quality
    dq_block = DataQualityBlock(
        status=dq.status, quality_score=dq.quality_score, freshness_laps=dq.freshness_laps,
        dropped_samples=dq.dropped_samples, out_of_order=dq.out_of_order,
        missing_fields=dq.missing_fields, checks=dq.checks,
    )
    w = ctx.window
    window_block = WindowBlock(
        n_laps=w.n_laps, speed_trend_kmh_per_lap=w.speed_trend_kmh_per_lap,
        gap_ahead_trend_s_per_lap=w.gap_ahead_trend_s_per_lap,
        soc_trend_mj_per_lap=w.soc_trend_mj_per_lap,
        rival_terminal_speed_trend=w.rival_terminal_speed_trend,
        rival_sector_delta_trend=w.rival_sector_delta_trend,
        closing=w.closing, opportunity_trend=w.opportunity_trend,
    )
    constraints_block = ConstraintsBlock(
        regulatory="PASS" if compliance_block.legal else "FAIL",
        data_quality=dq.status,
        energy="RESERVE_LOW" if (ctx.energy and ctx.energy.reserve_low) else "OK",
    )

    decision_block = DecisionBlock(
        mode=ctx.final_mode,
        action=ctx.action,
        confidence=overall_conf,
        stage2_mode=cgr.stage2_recommended_mode.value,
        confidence_overridden=cgr.overridden,
        override_reason=cgr.override_reason,
    )

    meta = SnapshotMeta(
        lap=nl.lap, total_laps=nl.total_laps, data_mode=nl.data_mode,
        source_detail=source_detail, seed=ctx.seed,
        pipeline_version=PIPELINE_VERSION,
        config_fingerprint=_config_fingerprint(ctx.config),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    return DecisionSnapshot(
        meta=meta,
        decision=decision_block,
        data_quality=dq_block,
        window=window_block,
        energy=energy_block,
        rival=rival_block,
        opportunity=opportunity_block,
        monte_carlo=mc_block,
        compliance=compliance_block,
        confidence=confidence_block,
        constraints=constraints_block,
        candidate_actions=list(ctx.candidate_actions),
        feasible_actions=list(ctx.feasible_actions),
        rejected_alternatives=[RejectedAlternative(**r) for r in ctx.rejected_alternatives],
        reason_codes=list(ctx.reason_codes),
        reasons=[rc.describe(c) for c in ctx.reason_codes],
        trace=build_trace(ctx),
    )


def _config_fingerprint(cfg: DecisionConfig) -> str:
    import dataclasses
    import hashlib

    payload = {"pipeline": PIPELINE_VERSION, **dataclasses.asdict(cfg)}
    try:
        from app.ml.predict import model_info

        payload["ml"] = model_info().get("dataset_hash", "none")
    except Exception:
        payload["ml"] = "none"
    blob = repr(sorted(payload.items())).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _compliance_block(ctx: DecisionContext) -> ComplianceBlock:
    g = ctx.gate_result
    ti = to_telemetry_input(ctx.target_lap, overtake_qualified_last_lap=ctx.overtake_bonus_banked)
    legal_modes = [m.value for m in g.legal_modes]
    illegal_modes = [m.value for m in DeploymentMode if m not in g.legal_modes]

    def _c(field: str, rule: str, status: str, detail: str = "") -> ComplianceCheck:
        rcst = REGULATORY_CONSTANTS.get(field)
        return ComplianceCheck(
            rule=rule,
            provenance=(rcst.provenance.value if rcst else "MODEL_ASSUMPTION"),
            status=status,
            detail=detail,
        )

    base_cap = ctx.config.gate_config().max_deployment_per_lap_mj
    over_cap = ti.lap_energy_deployed_mj > base_cap
    power_ceiling_kw = ctx.config.gate_config().max_ers_k_power_kw
    modeled_peak_kw = (ctx.energy.modeled_mgu_k_peak_kw
                       if ctx.energy is not None else None)
    if modeled_peak_kw is None:
        mgu_k_status, mgu_k_detail = "info", f"ceiling {power_ceiling_kw:.0f} kW (no modelled peak)"
    else:
        mgu_k_status = "breach" if modeled_peak_kw > power_ceiling_kw + 1e-6 else "pass"
        mgu_k_detail = f"{modeled_peak_kw:.0f} / {power_ceiling_kw:.0f} kW (modelled from throttle trace)"
    checks = [
        _c("max_deployment_per_lap_mj", "Per-lap deployment cap",
           "breach" if over_cap else "pass",
           f"{ti.lap_energy_deployed_mj:.2f} / {base_cap:.1f} MJ"),
        _c("max_delta_soc_mj", "Per-lap SoC swing",
           "info" if ti.lap_start_soc_mj is None else "pass",
           "skipped (no lap-start SoC)" if ti.lap_start_soc_mj is None
           else f"{abs(ti.lap_start_soc_mj - ti.current_soc_mj):.2f} / {ctx.config.gate_config().max_delta_soc_mj:.1f} MJ"),
        _c("overtake_detection_gap_threshold_s", "Override / Overtake Mode proximity",
           "pass" if (ti.gap_to_car_ahead_s is not None and ti.gap_to_car_ahead_s <= 1.0) else "info",
           f"gap {ti.gap_to_car_ahead_s}s / 1.0s"),
        _c("overtake_bonus_mj", "Overtake bonus banking",
           "pass" if ctx.overtake_bonus_banked else "info",
           "banked from previous lap" if ctx.overtake_bonus_banked else "not banked"),
        _c("max_ers_k_power_kw", "MGU-K power ceiling", mgu_k_status, mgu_k_detail),
    ]
    return ComplianceBlock(
        legal=(ctx.final_mode in legal_modes) or (len(legal_modes) == 0 and ctx.final_mode == "BALANCED_MODE" and not _hard_illegal(ctx)),
        legal_modes=legal_modes,
        illegal_modes=illegal_modes,
        checks=checks,
    )


def _hard_illegal(ctx: DecisionContext) -> bool:
    """True when the lap itself breaches a cap (not merely 'no aggressive option')."""
    ti = to_telemetry_input(ctx.target_lap, overtake_qualified_last_lap=ctx.overtake_bonus_banked)
    cfg = ctx.config.gate_config()
    if ti.lap_energy_deployed_mj > cfg.max_deployment_per_lap_mj:
        return True
    if ti.lap_start_soc_mj is not None and abs(ti.lap_start_soc_mj - ti.current_soc_mj) > cfg.max_delta_soc_mj:
        return True
    return False


# ===========================================================================
# feature helpers
# ===========================================================================
def _deployment_headroom(deployed: float, has_bonus: bool, cfg: DecisionConfig) -> float:
    gc = cfg.gate_config()
    cap = gc.max_deployment_per_lap_mj + (gc.overtake_bonus_mj if has_bonus else 0.0)
    return float(np.clip(cap - deployed, 0.0, cap))


def _ml_overtake_probability(
    nl: NormalizedLap, soc_mj: float, window
) -> Optional[float]:
    """
    P(overtake success) from the RandomForest classifier (heuristic fallback if
    the model file is absent). Every feature is genuinely computed from the
    normalized lap + the event-time window — no hardcoded placeholders. Both code
    paths are deterministic. This is an INPUT to the decision, never the decider.
    """
    try:
        from app.ml.features import build_features
        from app.ml.predict import predict_probability

        feats = build_features(
            gap_to_car_ahead_s=nl.gap_to_car_ahead_s,
            gap_trend_s_per_lap=(window.gap_ahead_trend_s_per_lap if window is not None else None),
            our_soc_mj=soc_mj,
            our_speed_kmh=nl.our_speed_kmh,
            drs_available=nl.overtake_mode_eligible,
            rival_terminal_speed_kmh=nl.rival_terminal_speed_kmh,
        )
        return round(float(predict_probability(feats)), 4)
    except Exception:
        return None


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
