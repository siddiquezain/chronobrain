"""
opportunity_engine.py — Opportunity Horizon

Multi-lap strategy comparison. Answers: "Is this the best moment to spend limited
energy, or will a better opportunity appear later?"

Sits ON TOP of Stages 1-3. Chains repeated calls to MonteCarloPlanner across a
bounded future horizon per candidate strategy. Nothing about single-lap physics
changes; what's new is comparing *sequences* of laps instead of one lap.

Uncertainty MUST widen with distance (horizon_uncertainty_growth in PlannerConfig).
An unqualified "wait 5 laps" claim is the overconfident behavior this project exists
to avoid — the widening is a hard requirement, not a nice-to-have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from planner import (
    DEFAULT_MODE_DYNAMICS,
    MonteCarloPlanner,
    PlannerConfig,
    PlanningContext,
)
from rival_estimator import RivalSocEstimate
from rule_gate import DeploymentMode, GateConfig, GateResult
from telemetry_simulator import TelemetryInput

HOLD_STRATEGY = "HOLD"
ATTACK_NOW_STRATEGY = "ATTACK_NOW"

_DEFAULT_GATE_CONFIG = GateConfig()


def _norm_cdf(z: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class HorizonConfig:
    """Configuration for the Opportunity Horizon."""

    delay_laps: list = field(default_factory=lambda: [0, 2, 5])
    horizon_laps: int = 10
    n_iterations: int = 10_000
    default_seed: int = 42

    # Rival-SoC sampling std cap for the Horizon's own per-iteration defense draw
    # (see OpportunityEngine._rival_defense_eff_prob). MODEL_ASSUMPTION.
    # Deliberately LESS restrictive than Stage 2's planner_rival_soc_std_cap_mj
    # (1.2 MJ) — that cap exists to protect Stage 2's statistical-significance
    # test from an overly wide posterior destabilizing it. The Horizon has no such
    # test to protect; its entire job here is to let rival uncertainty show up as
    # strategic risk (utility_std / downside_probability), so capping it as
    # tightly as Stage 2 would blunt exactly the property it needs to expose.
    # This is the ONE intentional divergence between the two layers' otherwise
    # shared rival-uncertainty definition (same mean/std source, same
    # defense_penalty_weight, same 9.0 MJ normalization).
    horizon_rival_soc_std_cap_mj: float = 3.0


class StrategyOutcome(BaseModel):
    strategy_name: str
    delay_laps: int
    mean_horizon_delta_s: float = Field(
        ...,
        description="Sum of mean laptime deltas over horizon (s). Negative = faster.",
    )
    std_horizon_delta_s: float
    confidence_ci_lower_s: float = Field(
        ..., description="One-sided 95% CI lower bound on advantage over HOLD (s)"
    )

    # --- Future Energy Value fields (populated only when start_soc_mj is supplied) ---
    end_soc_mj: float = Field(0.0, description="Modelled SoC at the end of the horizon")
    energy_spent_mj: float = Field(0.0, description="Net energy the strategy deploys over the horizon")
    current_opportunity_value: float = Field(0.0, description="Value of the window this strategy takes now")
    future_opportunity_value: float = Field(0.0, description="Value credited for reaching a later window")
    energy_opportunity_cost: float = Field(
        0.0,
        description="Cost of the energy THIS strategy spends, s-equiv. A pure function of "
        "this strategy's own spent/end_soc — never a reference to another strategy's "
        "current/future_opportunity_value (that would double-count it; see module docstring).",
    )
    strategic_value: float = Field(
        0.0,
        description="pace + current/future opportunity value - energy opportunity cost. "
        "Ranking key when Future Energy Value is active; higher is better.",
    )
    attack_completion_probability: Optional[float] = Field(
        None,
        description="P(the completion-lap overtake attempt succeeds), from a per-iteration "
        "rival-SoC draw — None if this strategy never reaches a completion lap, or no "
        "rival estimate was supplied (deterministic fallback: full credit if attacked).",
    )
    attack_completion_probability_std: Optional[float] = Field(
        None,
        description="Std of the per-iteration completion probability itself — the "
        "UNDILUTED signal of rival uncertainty (scales ~linearly with rival std_soc_mj, "
        "clipped near 0/1). utility_std below folds this into the horizon's total pace "
        "variance, where — at typical horizon_laps — it is a small contribution; this "
        "field is the honest, undiluted place to see 'different plausible rival states "
        "produce different attack outcomes'.",
    )
    utility_std: float = Field(
        0.0,
        description="Std of strategic_value across iterations = sqrt(pace_std^2 + "
        "(overtake_reward_s * window_strength * eff_prob_std)^2) — widens with rival "
        "uncertainty even though the MEAN strategic_value does not (see module docstring).",
    )
    downside_probability: float = Field(
        0.0,
        description="P(this strategy's strategic_value < HOLD's), from a normal "
        "approximation using strategic_value and utility_std. 0.0 for HOLD itself.",
    )


class HorizonResult(BaseModel):
    recommended_strategy: str
    ranked_strategies: list[StrategyOutcome]
    foregone_strategy: str
    foregone_value_gap_s: float = Field(
        ...,
        description="How much better the chosen strategy is vs runner-up (s)",
    )
    uncertainty_note: str
    future_energy_value_active: bool = Field(
        False, description="True when strategies were ranked by strategic_value (energy carried forward)"
    )


class OpportunityEngine:
    """
    Compares a small number of named multi-lap strategies via Monte Carlo horizon simulation.

    Strategies:
        ATTACK_NOW: ARM_OVERTAKE_MODE → USE_OVERTAKE_BONUS_MODE → BALANCED_MODE
        WAIT_N:     BALANCED_MODE × N laps → ATTACK_NOW sequence → BALANCED_MODE
        HOLD:       BALANCED_MODE for entire horizon (safe baseline)
    """

    def __init__(
        self,
        config: Optional[HorizonConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or HorizonConfig()
        cfg_seed = seed if seed is not None else self.config.default_seed
        self._rng = np.random.default_rng(cfg_seed)
        self._planner_config = PlannerConfig(
            n_iterations=self.config.n_iterations, default_seed=cfg_seed
        )

    def evaluate(
        self,
        current_telemetry: TelemetryInput,
        gate_result: GateResult,
        rival_estimate: Optional[RivalSocEstimate] = None,
        window_strength_by_delay: Optional[dict] = None,
        *,
        start_soc_mj: Optional[float] = None,
        laps_remaining: Optional[int] = None,
        reserve_floor_mj: float = 1.0,
        overtake_reward_s: float = 0.3,
        future_window_bias: float = 0.0,
        restrict_to_delays: Optional[set] = None,
        p_defend: Optional[float] = None,
    ) -> HorizonResult:
        """
        Evaluate candidate multi-lap strategies and return ranked outcomes.

        `window_strength_by_delay` (optional): {delay_laps: strength} — scales the
        laptime advantage of that strategy's ARM/USE_OVERTAKE_BONUS laps. 1.0 =
        neutral. Omitted -> all 1.0 and byte-identical to before this parameter.

        Future Energy Value (only when `start_soc_mj` is supplied): SoC is carried
        forward lap-by-lap; a strategy that would starve the reserve before its
        window cannot actually attack, and each strategy gets a `strategic_value`
        = pace + current/future opportunity value - energy opportunity cost.
        Strategies are then ranked by `strategic_value` instead of raw pace.

        `restrict_to_delays`: if given, only strategies whose delay is in this set
        (plus HOLD) are evaluated — the feasible-set filter.

        `rival_estimate` / `p_defend` (optional, additive): when supplied, each
        strategy's completion-lap attempt draws a per-iteration rival-SoC sample
        (same formula as planner.py's Stage-2 modulation) instead of crediting a
        flat "attacked -> full value" outcome. This is what makes rival
        UNCERTAINTY (not just its mean) visible in `utility_std` /
        `downside_probability` — see `_rival_defense_eff_prob`. Omitted -> the
        deterministic pre-existing behaviour, byte-identical.

        No-double-counting invariant: `energy_opportunity_cost(s)` is computed
        ONLY from strategy s's own `spent`/`end_soc` — it never reads another
        strategy's `current_opportunity_value`/`future_opportunity_value`.
        `current_opportunity_value` and `future_opportunity_value` are mutually
        exclusive within one strategy (gated by delay==0 vs delay>0). So every
        opportunity-value term is summed into the objective exactly once, never
        twice, either within a strategy or across the strategies being compared.
        """
        cfg = self.config
        self._window_strength = window_strength_by_delay or {}
        fev = start_soc_mj is not None

        strategies: list[tuple[str, int]] = [(HOLD_STRATEGY, -1)]
        for d in sorted(cfg.delay_laps):
            if restrict_to_delays is not None and d not in restrict_to_delays:
                continue
            name = ATTACK_NOW_STRATEGY if d == 0 else f"WAIT_{d}"
            strategies.append((name, d))

        outcomes: list[StrategyOutcome] = []
        hold_strategic_value: Optional[float] = None

        for strategy_name, delay in strategies:
            (mean_total, std_total, end_soc, spent, attacked,
             completion_prob, eff_prob_std) = self._simulate_strategy(
                strategy_name, delay, gate_result, rival_estimate,
                start_soc_mj=start_soc_mj, reserve_floor_mj=reserve_floor_mj,
                p_defend=p_defend,
            )
            ci_lower = float(mean_total - 1.96 * std_total / np.sqrt(cfg.n_iterations))

            cur_val = fut_val = e_cost = strat_val = 0.0
            utility_std = std_total
            downside_probability = 0.0
            if fev:
                strength = self._window_strength.get(max(delay, 0), 1.0)
                # Continuous credit — mean of the per-iteration completion
                # probability when a rival estimate was supplied, else the old
                # deterministic "attacked -> full credit" fallback (byte-identical
                # when rival_estimate is None).
                credit = completion_prob if completion_prob is not None else (1.0 if attacked else 0.0)
                opp_value = overtake_reward_s * strength * credit
                if delay == 0:
                    cur_val = opp_value
                elif delay > 0 and attacked:
                    # value of the later window, plus a bias when the window is improving
                    fut_val = opp_value + future_window_bias

                # Energy opportunity cost — a pure function of THIS strategy's own
                # spent/end_soc (see no-double-counting invariant in the docstring
                # above). Replaces the old undocumented flat 0.08 s/MJ rate with a
                # rate built from existing, already-documented config: the
                # strategic seconds-value of one future attack (overtake_reward_s),
                # scaled by how strong the best still-reachable future window looks
                # (best_future_strength), per MJ that attack costs (attack_cost_mj).
                best_future_strength = max(
                    (self._window_strength.get(d, 1.0) for d in cfg.delay_laps if d > max(delay, 0)),
                    default=strength,
                )
                attack_cost = max(self._planner_config.attack_cost_mj, 1e-6)
                value_density = overtake_reward_s * best_future_strength / attack_cost
                # marginal energy value rises as the horizon-end reserve nears the floor
                scarcity = max(0.0, (2.0 * reserve_floor_mj - end_soc) / max(reserve_floor_mj, 1e-6))
                e_cost = spent * value_density * (1.0 + scarcity)
                strat_val = (-mean_total) + cur_val + fut_val - e_cost

                if eff_prob_std is not None and eff_prob_std > 0.0:
                    opp_value_std = overtake_reward_s * strength * eff_prob_std
                    utility_std = float(np.sqrt(std_total ** 2 + opp_value_std ** 2))

                if strategy_name == HOLD_STRATEGY:
                    hold_strategic_value = strat_val
                elif hold_strategic_value is not None:
                    if utility_std > 1e-9:
                        downside_probability = round(
                            _norm_cdf((hold_strategic_value - strat_val) / utility_std), 4
                        )
                    else:
                        downside_probability = 1.0 if strat_val < hold_strategic_value else 0.0

            outcomes.append(
                StrategyOutcome(
                    strategy_name=strategy_name,
                    delay_laps=delay,
                    mean_horizon_delta_s=round(mean_total, 4),
                    std_horizon_delta_s=round(std_total, 4),
                    confidence_ci_lower_s=round(ci_lower, 4),
                    end_soc_mj=round(end_soc, 4),
                    energy_spent_mj=round(spent, 4),
                    current_opportunity_value=round(cur_val, 4),
                    future_opportunity_value=round(fut_val, 4),
                    energy_opportunity_cost=round(e_cost, 4),
                    strategic_value=round(strat_val, 4),
                    attack_completion_probability=(
                        round(completion_prob, 4) if completion_prob is not None else None
                    ),
                    attack_completion_probability_std=(
                        round(eff_prob_std, 4) if eff_prob_std is not None else None
                    ),
                    utility_std=round(utility_std, 4),
                    downside_probability=downside_probability,
                )
            )

        key = (lambda o: -o.strategic_value) if fev else (lambda o: o.mean_horizon_delta_s)
        outcomes.sort(key=key)

        best = outcomes[0]
        runner_up = outcomes[1] if len(outcomes) > 1 else outcomes[0]
        if fev:
            value_gap = float(abs(runner_up.strategic_value - best.strategic_value))
        else:
            value_gap = float(abs(runner_up.mean_horizon_delta_s - best.mean_horizon_delta_s))

        return HorizonResult(
            recommended_strategy=best.strategy_name,
            ranked_strategies=outcomes,
            foregone_strategy=runner_up.strategy_name,
            foregone_value_gap_s=round(value_gap, 4),
            future_energy_value_active=fev,
            uncertainty_note=(
                f"Confidence decreases with delay: each additional lap of wait widens uncertainty "
                f"by ~{int(self._planner_config.horizon_uncertainty_growth * 100)}% per lap. "
                "WAIT_5 carries substantially less confidence than ATTACK_NOW."
            ),
        )

    def _rival_defense_eff_prob(
        self,
        base_prob: float,
        n: int,
        rival_estimate: RivalSocEstimate,
        p_defend: Optional[float],
    ) -> np.ndarray:
        """
        Per-iteration effective completion probability for one attack attempt,
        modulated by a per-iteration rival-SoC draw. Deliberately the SAME
        formula as `planner.py::MonteCarloPlanner._effective_overtake_prob` —
        same `defense_penalty_weight` (PlannerConfig), same 9.0 MJ normalization
        — so Stage 2 and the Horizon share one definition of "rival uncertainty".
        The only difference is the std cap (`HorizonConfig.horizon_rival_soc_std_cap_mj`
        vs Stage 2's `planner_rival_soc_std_cap_mj`) — see HorizonConfig docstring
        for why that divergence is intentional.
        """
        pc = self._planner_config
        cap_mj = _DEFAULT_GATE_CONFIG.max_deployment_per_lap_mj
        std = min(rival_estimate.std_soc_mj, self.config.horizon_rival_soc_std_cap_mj)
        rival_soc = self._rng.normal(rival_estimate.mean_soc_mj, std, n)
        rival_soc = np.clip(rival_soc, 0.0, cap_mj)
        defense_factor = rival_soc / cap_mj
        eff_prob = base_prob - pc.defense_penalty_weight * defense_factor
        if p_defend is not None:
            eff_prob = eff_prob - pc.defense_penalty_weight * float(p_defend)
        return np.clip(eff_prob, 0.0, 1.0)

    def _simulate_strategy(
        self,
        strategy_name: str,
        delay: int,
        gate_result: GateResult,
        rival_estimate: Optional[RivalSocEstimate],
        *,
        start_soc_mj: Optional[float] = None,
        reserve_floor_mj: float = 1.0,
        p_defend: Optional[float] = None,
    ) -> tuple[float, float, float, float, bool, Optional[float], Optional[float]]:
        """
        Simulate strategy over the horizon.
        Returns (mean_total_delta, std_total_delta, end_soc, energy_spent, attacked,
        attack_completion_probability, eff_prob_std).
        When start_soc_mj is None, energy carry is skipped and the laptime figures
        are byte-identical to the pre-FEV implementation.

        attack_completion_probability / eff_prob_std are populated ONLY at the
        strategy's completion lap (USE_OVERTAKE_BONUS_MODE) AND only when a rival
        estimate is supplied — otherwise both are None (the caller falls back to
        the deterministic "attacked -> full credit" behaviour, unchanged).
        """
        cfg = self.config
        pc = self._planner_config
        n = cfg.n_iterations
        horizon = cfg.horizon_laps

        total_samples = np.zeros(n)
        strength = getattr(self, "_window_strength", {}).get(max(delay, 0), 1.0)

        soc = start_soc_mj
        spent = 0.0
        attacked = False
        completion_prob: Optional[float] = None
        eff_prob_std: Optional[float] = None
        harvest_per_lap = pc.harvest_per_lap_mj  # canonical home: PlannerConfig (mirrors actions.py)

        for lap_offset in range(horizon):
            mode = self._strategy_lap_mode(strategy_name, delay, lap_offset, gate_result)
            is_attack = mode in (
                DeploymentMode.ARM_OVERTAKE_MODE,
                DeploymentMode.USE_OVERTAKE_BONUS_MODE,
            )

            starved = False
            if soc is not None and is_attack:
                cost = DEFAULT_MODE_DYNAMICS[mode].energy_cost_mj
                if soc - cost < reserve_floor_mj:
                    # can't deploy what we don't have -> fall back to BALANCED this lap
                    starved = True
                    mode = DeploymentMode.BALANCED_MODE
                    is_attack = False

            uncertainty_scale = 1.0 + pc.horizon_uncertainty_growth * lap_offset
            dyn = DEFAULT_MODE_DYNAMICS[mode]
            lap_samples = self._rng.normal(
                dyn.mean_laptime_delta_s, dyn.std_laptime_delta_s * uncertainty_scale, n
            )
            if strength != 1.0 and is_attack:
                lap_samples = lap_samples * strength
            if starved:
                lap_samples = lap_samples + 0.35  # missed-attack penalty
            total_samples += lap_samples

            if is_attack and mode == DeploymentMode.USE_OVERTAKE_BONUS_MODE and rival_estimate is not None:
                eff_prob = self._rival_defense_eff_prob(
                    dyn.overtake_success_prob, n, rival_estimate, p_defend
                )
                completion_prob = float(np.mean(eff_prob))
                eff_prob_std = float(np.std(eff_prob))

            if soc is not None:
                soc = max(0.0, soc - max(0.0, dyn.energy_cost_mj) + (harvest_per_lap if not is_attack else 0.0))
                if is_attack:
                    spent += dyn.energy_cost_mj
                    attacked = True

        end_soc = float(soc) if soc is not None else 0.0
        return (
            float(np.mean(total_samples)),
            float(np.std(total_samples)),
            end_soc,
            float(spent),
            bool(attacked),
            completion_prob,
            eff_prob_std,
        )

    def _strategy_lap_mode(
        self,
        strategy_name: str,
        delay: int,
        lap_offset: int,
        gate_result: GateResult,
    ) -> DeploymentMode:
        """Determine deployment mode for a given lap offset within a strategy."""
        if strategy_name == HOLD_STRATEGY:
            return DeploymentMode.BALANCED_MODE

        attack_start = delay if delay >= 0 else 0
        if strategy_name == ATTACK_NOW_STRATEGY:
            attack_start = 0

        rel = lap_offset - attack_start
        if rel == 0 and DeploymentMode.ARM_OVERTAKE_MODE in gate_result.legal_modes:
            return DeploymentMode.ARM_OVERTAKE_MODE
        elif rel == 1 and DeploymentMode.USE_OVERTAKE_BONUS_MODE in gate_result.legal_modes:
            return DeploymentMode.USE_OVERTAKE_BONUS_MODE
        else:
            return DeploymentMode.BALANCED_MODE
