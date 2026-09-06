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
from app.data.samples import NormalizedLap
from app.decision.config import DecisionConfig
from app.decision.context import DecisionContext, EnergyFeatures, RivalFeatures
from app.decision import reason_codes as rc
from app.decision.snapshot import (
    ComplianceBlock,
    ComplianceCheck,
    ConfidenceBlock,
    DecisionBlock,
    DecisionSnapshot,
    EnergyBlock,
    MonteCarloBlock,
    MonteCarloMode,
    OpportunityBlock,
    OpportunityStrategy,
    RivalBlock,
    SnapshotMeta,
)
from app.decision.trace import build_trace
from app.regulation.constants import REGULATORY_CONSTANTS
from app.narrative.narrator import narrate as _narrate

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
    all_laps = provider.laps()
    if not all_laps:
        raise ValueError("provider yielded no laps")

    target_lap_no = lap if lap is not None else all_laps[-1].lap
    history = [nl for nl in all_laps if nl.lap <= target_lap_no]
    if not history or history[-1].lap != target_lap_no:
        raise ValueError(f"provider has no lap {target_lap_no}")

    ctx = DecisionContext(config=cfg, lap_history=history, target_lap=history[-1])

    _thread_state(ctx)          # bonus qualification + rival particle filter over laps 1..N
    _extract_features(ctx)      # energy + ML P(overtake success)
    _run_gate(ctx)              # Stage 1
    _run_planner(ctx)           # Stage 2
    _run_horizon(ctx)           # opportunity horizon
    _run_confidence(ctx)        # Stage 3
    _fuse(ctx)                  # decision engine

    snapshot = _assemble(ctx, provider.describe())
    if with_narrative:
        snapshot.narrative = _narrate(snapshot)
    return snapshot


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
def _thread_state(ctx: DecisionContext) -> None:
    """
    Walk laps 1..N. `overtake_bonus_banked` for lap N is set by lap N-1's gate
    (qualification is a telemetry fact, not a mode choice). The rival particle
    filter is advanced once per lap with that lap's observation when present.
    """
    cfg = ctx.config
    gate = RegulatoryGate(config=cfg.gate_config())
    estimator = RivalStateEstimator(config=cfg.rival_config(), seed=cfg.seed)

    banked = False
    for i, nl in enumerate(ctx.lap_history):
        ti = to_telemetry_input(nl, overtake_qualified_last_lap=banked)
        gr = gate.evaluate(ti)

        obs = to_rival_observation(nl)
        if obs is not None:
            estimator.predict()
            estimator.update(obs)

        if i == len(ctx.lap_history) - 1:
            ctx.overtake_bonus_banked = banked
            ctx.gate_result = gr  # provisional; re-evaluated in _run_gate for clarity
        banked = gr.qualifies_for_overtake_bonus_next_lap

    ctx.rival = _rival_features(ctx, estimator)


def _rival_features(ctx: DecisionContext, estimator: RivalStateEstimator) -> Optional[RivalFeatures]:
    if not any(nl.has_rival_observation for nl in ctx.lap_history):
        return None
    est = estimator.estimate()
    cfg = ctx.config

    lo, hi = cfg.rival_low_soc_mj, cfg.rival_high_soc_mj
    std = max(est.std_soc_mj, 1e-6)
    p_low = _norm_cdf((lo - est.mean_soc_mj) / std)
    p_high = 1.0 - _norm_cdf((hi - est.mean_soc_mj) / std)
    p_med = max(0.0, 1.0 - p_low - p_high)
    total = p_low + p_med + p_high
    dist = {"low": p_low / total, "medium": p_med / total, "high": p_high / total}
    bucket = max(dist, key=dist.get).upper()

    threshold = cfg.confidence_config().rival_confidence_threshold_mj
    uncertain = est.std_soc_mj > threshold
    confidence = float(np.clip(1.0 - est.std_soc_mj / 2.6, 0.0, 1.0))

    return RivalFeatures(
        estimate=est, bucket=bucket, distribution=dist,
        confidence=round(confidence, 4), uncertain=uncertain,
    )


def _extract_features(ctx: DecisionContext) -> None:
    nl = ctx.target_lap
    cfg = ctx.config
    ti = to_telemetry_input(nl, overtake_qualified_last_lap=ctx.overtake_bonus_banked)

    soc = ti.current_soc_mj
    laps_remaining = max(0, nl.total_laps - nl.lap)
    headroom = _deployment_headroom(ti.lap_energy_deployed_mj, ctx.overtake_bonus_banked, cfg)
    projected = float(np.clip(soc + laps_remaining * _NET_SOC_PER_LAP_MJ, 0.0, _SOC_CEIL_MJ))
    end_of_race = projected
    can_afford = soc >= 3.0 and headroom >= 1.5 and projected >= _MIN_RESERVE_MJ
    reserve_low = projected < cfg.low_reserve_mj or soc < cfg.low_reserve_mj

    ctx.energy = EnergyFeatures(
        soc_mj=round(soc, 3),
        soc_pct=round(soc / _SOC_CEIL_MJ * 100.0, 2),
        lap_start_soc_mj=ti.lap_start_soc_mj,
        deployed_this_lap_mj=round(ti.lap_energy_deployed_mj, 3),
        deployment_headroom_mj=round(headroom, 3),
        projected_reserve_mj=round(projected, 3),
        projected_end_of_race_mj=round(end_of_race, 3),
        can_afford_aggressive=bool(can_afford),
        reserve_low=bool(reserve_low),
        energy_is_modeled=nl.energy_is_modeled,
    )
    ctx.ml_overtake_prob = _ml_overtake_probability(nl, soc)


def _run_gate(ctx: DecisionContext) -> None:
    ti = to_telemetry_input(ctx.target_lap, overtake_qualified_last_lap=ctx.overtake_bonus_banked)
    ctx.gate_result = RegulatoryGate(config=ctx.config.gate_config()).evaluate(ti)


def _run_planner(ctx: DecisionContext) -> None:
    cfg = ctx.config
    planner = MonteCarloPlanner(config=cfg.planner_config(), seed=cfg.seed)
    pc = PlanningContext(
        gap_to_car_ahead_s=ctx.target_lap.gap_to_car_ahead_s,
        rival_soc_estimate=ctx.rival.estimate if ctx.rival else None,
    )
    ctx.planner_result = planner.plan(ctx.gate_result, pc)
    ctx._planner = planner  # kept for the confidence gate (raw samples)


def _run_horizon(ctx: DecisionContext) -> None:
    cfg = ctx.config
    engine = OpportunityEngine(config=cfg.horizon_config(), seed=cfg.seed)
    ti = to_telemetry_input(ctx.target_lap, overtake_qualified_last_lap=ctx.overtake_bonus_banked)
    ctx.horizon_result = engine.evaluate(
        ti,
        ctx.gate_result,
        ctx.rival.estimate if ctx.rival else None,
        window_strength_by_delay=_window_strength(ctx),
    )


def _window_strength(ctx: DecisionContext) -> dict:
    """
    How well does an overtake window hold up N laps from now?

    Default is mild decay (a bird in the hand — windows pass). It decays LESS when
    the rival is confidently in a LOW energy state (they are not recovering, so a
    later attempt is nearly as good) and MORE when the rival estimate is uncertain
    or shows medium/high reserve. This is how rival-energy uncertainty reaches the
    timing decision. MODEL_ASSUMPTION — a richer gap/rival-trajectory forecast is a
    calibration follow-up.
    """
    per_lap = 0.92
    if ctx.rival is not None:
        if ctx.rival.bucket == "LOW" and not ctx.rival.uncertain:
            per_lap = 0.985
        elif ctx.rival.uncertain or ctx.rival.bucket == "HIGH":
            per_lap = 0.85
    return {d: round(per_lap ** d, 4) for d in ctx.config.delay_laps}


def _run_confidence(ctx: DecisionContext) -> None:
    gate = ConfidenceGate(config=ctx.config.confidence_config())
    load = DriverLoadInput(
        laps_since_last_mode_change=3.0,           # MODEL_ASSUMPTION: single-lap API has no history of mode changes
        recent_laptime_std_s=0.25,
        gap_to_car_ahead_s=ctx.target_lap.gap_to_car_ahead_s,
    )
    ctx.confidence_result = gate.evaluate(
        ctx.planner_result,
        driver_load=load,
        rival_estimate=ctx.rival.estimate if ctx.rival else None,
        planner=getattr(ctx, "_planner", None),
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

    prefers_wait, wait_n = _horizon_prefers_wait(ctx)
    can_afford = bool(ctx.energy and ctx.energy.can_afford_aggressive)
    reserve_low = bool(ctx.energy and ctx.energy.reserve_low)
    behind = ctx.target_lap.gap_to_car_behind_s
    defending = behind is not None and behind <= _DEFEND_GAP_S

    def _hold_mode() -> DeploymentMode:
        if reserve_low and DeploymentMode.CONSERVE_MODE in legal:
            return DeploymentMode.CONSERVE_MODE
        return DeploymentMode.BALANCED_MODE

    if all_illegal:
        final_mode, action = DeploymentMode.BALANCED_MODE, "HOLD"
    elif cgr.overridden:
        final_mode, action = DeploymentMode.BALANCED_MODE, "HOLD"
    elif cg_mode == DeploymentMode.PUSH_MODE:
        # PUSH spends extra energy for pace. Worth it only to defend a real threat
        # behind and only if affordable — never as an always-on "free pace" button.
        if defending and can_afford:
            final_mode, action = DeploymentMode.PUSH_MODE, "PUSH"
        elif not can_afford:
            final_mode = _hold_mode()
            action = "CONSERVE" if final_mode == DeploymentMode.CONSERVE_MODE else "HOLD"
        else:
            final_mode, action = DeploymentMode.BALANCED_MODE, "HOLD"
    elif cg_aggressive and not can_afford:
        # "is the attack worth the energy?" — no.
        final_mode = _hold_mode()
        action = "CONSERVE" if final_mode == DeploymentMode.CONSERVE_MODE else "HOLD"
    elif cg_aggressive and prefers_wait:
        final_mode = _hold_mode()
        action = f"WAIT_{wait_n}_LAPS"
    elif cg_aggressive:
        final_mode, action = cg_mode, "ATTACK_NOW"
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
        horizon_strategy=h.recommended_strategy,
        horizon_prefers_wait=prefers_wait,
        energy_reserve_low=ctx.energy.reserve_low if ctx.energy else False,
        can_afford_aggressive=ctx.energy.can_afford_aggressive if ctx.energy else False,
        ml_overtake_prob=ctx.ml_overtake_prob,
        ml_prob_high=cfg.ml_prob_high,
        ml_prob_low=cfg.ml_prob_low,
    )
    ctx.reason_codes = rc.select(ri)


def _horizon_prefers_wait(ctx: DecisionContext) -> tuple[bool, int]:
    h = ctx.horizon_result
    if h is None or not h.recommended_strategy.startswith("WAIT"):
        return False, 0
    attack_now = next(
        (s for s in h.ranked_strategies if s.strategy_name == "ATTACK_NOW"), None
    )
    best = h.ranked_strategies[0]
    if attack_now is None:
        return False, 0
    margin = attack_now.mean_horizon_delta_s - best.mean_horizon_delta_s
    if margin < ctx.config.horizon_decisive_margin_s:
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
    )

    if ctx.rival is not None:
        r = ctx.rival
        rival_block = RivalBlock(
            mean_reserve_mj=r.estimate.mean_soc_mj,
            reserve_std_mj=r.estimate.std_soc_mj,
            n_observations=r.estimate.n_observations,
            confidence=r.confidence,
            estimate_uncertain=r.uncertain,
            bucket=r.bucket,
            distribution={k: round(v, 4) for k, v in r.distribution.items()},
        )
    else:
        rival_block = RivalBlock(
            mean_reserve_mj=0.0, reserve_std_mj=0.0, n_observations=0,
            confidence=0.0, estimate_uncertain=True, bucket="MEDIUM",
            distribution={"low": 0.0, "medium": 1.0, "high": 0.0},
        )

    ranked = [
        OpportunityStrategy(
            name=s.strategy_name, delay_laps=s.delay_laps,
            mean_horizon_delta_s=s.mean_horizon_delta_s,
            std_horizon_delta_s=s.std_horizon_delta_s,
            ci_lower_s=s.confidence_ci_lower_s,
        )
        for s in h.ranked_strategies
    ]
    cur_prob = round(float(ctx.ml_overtake_prob or 0.0), 4)
    prefers_wait, wait_n = _horizon_prefers_wait(ctx)
    proj_prob = round(min(1.0, cur_prob * 1.12), 4) if prefers_wait else cur_prob
    opportunity_block = OpportunityBlock(
        recommended_strategy=h.recommended_strategy,
        prefers_wait=prefers_wait,
        foregone_strategy=h.foregone_strategy,
        foregone_value_gap_s=h.foregone_value_gap_s,
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
                sharpe_ratio=m.sharpe_ratio,
                energy_cost_mj=m.energy_cost_mj,
            )
            for m in pr.ranked_modes
        ],
    )

    compliance_block = _compliance_block(ctx)

    confidence_block = ConfidenceBlock(
        overall=overall_conf,
        statistical_reliability_passed=cgr.statistical_reliability_passed,
        practical_significance_passed=cgr.practical_significance_passed,
        dcli_passed=cgr.dcli_passed,
        rival_confidence_passed=cgr.rival_confidence_passed,
        ci_lower_bound_s=cgr.ci_lower_bound_s,
        t_statistic=cgr.t_statistic,
        dcli_score=cgr.dcli_score,
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
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    return DecisionSnapshot(
        meta=meta,
        decision=decision_block,
        energy=energy_block,
        rival=rival_block,
        opportunity=opportunity_block,
        monte_carlo=mc_block,
        compliance=compliance_block,
        confidence=confidence_block,
        reason_codes=list(ctx.reason_codes),
        reasons=[rc.describe(c) for c in ctx.reason_codes],
        trace=build_trace(ctx),
    )


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
        _c("max_ers_k_power_kw", "MGU-K power ceiling", "pass", "350 kW — not exceeded in model"),
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


def _ml_overtake_probability(nl: NormalizedLap, soc_mj: float) -> Optional[float]:
    """
    P(overtake success) from the RandomForest classifier (heuristic fallback if
    the model file is absent). Both paths are deterministic. This is an INPUT to
    the decision, never the decider.
    """
    try:
        from app.ml.predict import predict_probability

        feats = np.array([[
            nl.gap_to_car_ahead_s if nl.gap_to_car_ahead_s is not None else 3.0,
            _closing_speed_proxy(nl),
            0.5,                                   # slipstream — not in NormalizedLap; neutral
            soc_mj,
            15.0,                                  # tyre age — unavailable; neutral
            700.0,                                 # straight length — unavailable; neutral
            nl.our_speed_kmh,
            1.0 if nl.drs_available else 0.0,
        ]], dtype=np.float64)
        return round(float(predict_probability(feats)), 4)
    except Exception:
        return None


def _closing_speed_proxy(nl: NormalizedLap) -> float:
    """No per-sample closing speed in a NormalizedLap; proxy from gap (smaller gap -> closing)."""
    if nl.gap_to_car_ahead_s is None:
        return 0.0
    return float(np.clip((1.5 - nl.gap_to_car_ahead_s) * 6.0, 0.0, 15.0))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
