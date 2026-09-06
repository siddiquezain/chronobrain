"""
trace.py — deterministic decision trace.

Every line is rendered from `DecisionContext` fields that were actually computed
this run. Nothing here is hardcoded; run the pipeline on different telemetry and
every number moves.
"""

from __future__ import annotations

from typing import List

from app.decision.context import DecisionContext
from app.decision.snapshot import TraceStep


def build_trace(ctx: DecisionContext) -> List[TraceStep]:
    t = ctx.target_lap
    steps: List[TraceStep] = []

    gap = "n/a" if t.gap_to_car_ahead_s is None else f"{t.gap_to_car_ahead_s:.2f}s"
    soc = "n/a" if (ctx.energy is None or ctx.energy.soc_mj is None) else f"{ctx.energy.soc_mj:.2f} MJ"
    modeled = " (modeled)" if (ctx.energy and ctx.energy.energy_is_modeled) else ""
    steps.append(TraceStep(
        stage="Telemetry",
        detail=f"Lap {t.lap}/{t.total_laps} | gap ahead {gap} | SoC {soc}{modeled} | source: {t.data_mode}",
    ))

    if ctx.data_quality is not None:
        dq = ctx.data_quality
        steps.append(TraceStep(
            stage="Data quality",
            detail=(
                f"{dq.status} (score {dq.quality_score:.2f}) | "
                f"dropped {dq.dropped_samples} | rival stale {dq.freshness_laps} lap(s) | "
                + ("; ".join(dq.checks[:2]))
            ),
        ))

    if ctx.window is not None:
        w = ctx.window
        gap_tr = "n/a" if w.gap_ahead_trend_s_per_lap is None else f"{w.gap_ahead_trend_s_per_lap:+.3f}s/lap"
        steps.append(TraceStep(
            stage="Event-time window",
            detail=(
                f"{w.n_laps} laps | opportunity {w.opportunity_trend} | "
                f"gap trend {gap_tr} | {'closing' if w.closing else 'not closing'}"
            ),
        ))

    if ctx.rival is not None:
        d = ctx.rival.distribution
        steps.append(TraceStep(
            stage="Rival energy",
            detail=(
                f"{ctx.rival.bucket} "
                f"(LOW {d['low']*100:.0f}% / MED {d['medium']*100:.0f}% / HIGH {d['high']*100:.0f}%) | "
                f"mean {ctx.rival.estimate.mean_soc_mj:.2f} +/- {ctx.rival.estimate.std_soc_mj:.2f} MJ | "
                f"{ctx.rival.estimate.n_observations} obs | P(defend) {ctx.rival.p_defend:.2f} | "
                f"{'UNCERTAIN' if ctx.rival.uncertain else 'usable'}"
            ),
        ))

    if ctx.gate_result is not None:
        g = ctx.gate_result
        rejected = [m.value for m in _all_modes() if m not in g.legal_modes]
        steps.append(TraceStep(
            stage="Regulatory gate",
            detail=(
                f"legal modes: {len(g.legal_modes)}/5"
                + (f" | rejected {', '.join(rejected)}" if rejected else "")
            ),
        ))

    if ctx.candidate_actions:
        rej = "; ".join(f"{r['action']} ({r['reason']})" for r in ctx.rejected_alternatives) or "none"
        steps.append(TraceStep(
            stage="Feasible set",
            detail=f"feasible: {', '.join(ctx.feasible_actions)} | rejected: {rej}",
        ))

    if ctx.horizon_result is not None:
        h = ctx.horizon_result
        cur = next((s for s in h.ranked_strategies if s.strategy_name == "ATTACK_NOW"), None)
        if h.future_energy_value_active and cur is not None:
            best = h.ranked_strategies[0]
            detail = (
                f"best = {h.recommended_strategy} (strategic value {best.strategic_value:+.3f}) | "
                f"ATTACK_NOW value {cur.strategic_value:+.3f} "
                f"(opp {cur.current_opportunity_value:+.2f}, e-cost {cur.energy_opportunity_cost:.2f}) | "
                f"end SoC {best.end_soc_mj:.1f} MJ"
            )
        elif cur is not None:
            detail = (
                f"best = {h.recommended_strategy} | ATTACK_NOW horizon d {cur.mean_horizon_delta_s:+.3f}s | "
                f"foregone {h.foregone_strategy} by {h.foregone_value_gap_s:.3f}s"
            )
        else:
            detail = f"best = {h.recommended_strategy}"
        steps.append(TraceStep(stage="Opportunity horizon (future energy value)", detail=detail))

    if ctx.planner_result is not None:
        p = ctx.planner_result
        top = p.ranked_modes[0] if p.ranked_modes else None
        steps.append(TraceStep(
            stage="Monte Carlo planner",
            detail=(
                f"{p.n_iterations} iters, seed {ctx.seed} | "
                + (f"top {top.mode.value} (mean {top.mean_laptime_delta_s:+.3f}s, "
                   f"Sharpe {top.sharpe_ratio:.2f})" if top else "no legal modes")
            ),
        ))

    if ctx.confidence_result is not None:
        c = ctx.confidence_result
        gates = (
            f"stat {_pf(c.statistical_reliability_passed)} / "
            f"prac {_pf(c.practical_significance_passed)} / "
            f"DCLI {_pf(c.dcli_passed)} / rival {_pf(c.rival_confidence_passed)} / "
            f"data {_pf(getattr(c, 'data_quality_passed', True))}"
        )
        steps.append(TraceStep(
            stage="Confidence gate",
            detail=(
                f"CI lower {c.ci_lower_bound_s:+.3f}s | {gates}"
                + (f" | OVERRIDE -> BALANCED ({c.override_reason})" if c.overridden else " | PASS")
            ),
        ))

    steps.append(TraceStep(
        stage="Decision engine",
        detail=_fusion_detail(ctx),
    ))
    steps.append(TraceStep(
        stage="FINAL",
        detail=f"{ctx.final_mode} / {ctx.action}",
    ))
    return steps


def _fusion_detail(ctx: DecisionContext) -> str:
    c = ctx.confidence_result
    h = ctx.horizon_result
    if ctx.gate_result is not None and not ctx.gate_result.legal_modes:
        return "no legal mode on this lap -> BALANCED_MODE / HOLD, compliance.legal = false"
    if c is not None and c.overridden:
        return f"confidence override in effect -> BALANCED_MODE, action HOLD"
    if ctx.action and ctx.action.startswith("WAIT"):
        return (
            f"single-lap pick was aggressive but horizon strategy {h.recommended_strategy} "
            f"beats attacking now by >= {ctx.config.horizon_decisive_margin_s}s -> defer, {ctx.action}"
        )
    if ctx.action == "ATTACK_NOW":
        if h is not None and h.recommended_strategy == "ATTACK_NOW":
            return "aggressive single-lap pick and horizon both favour attacking now"
        return (
            "aggressive single-lap pick stands; horizon's margin for waiting is below "
            f"the {ctx.config.horizon_decisive_margin_s}s decisive threshold"
        )
    if ctx.action == "PUSH":
        return "defending a car within the rearward gap threshold -> PUSH"
    if ctx.action in ("HOLD", "CONSERVE"):
        if ctx.energy and not ctx.energy.can_afford_aggressive:
            return f"attack not worth the energy spend (reserve too low) -> {ctx.final_mode}"
        return f"no window worth committing energy to -> {ctx.final_mode}"
    return f"pick stands ({ctx.final_mode})"


def _pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _all_modes():
    from rule_gate import DeploymentMode

    return list(DeploymentMode)
