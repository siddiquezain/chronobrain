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
from rule_gate import DeploymentMode, GateResult
from telemetry_simulator import TelemetryInput

HOLD_STRATEGY = "HOLD"
ATTACK_NOW_STRATEGY = "ATTACK_NOW"


@dataclass
class HorizonConfig:
    """Configuration for the Opportunity Horizon."""

    delay_laps: list = field(default_factory=lambda: [0, 2, 5])
    horizon_laps: int = 10
    n_iterations: int = 10_000
    default_seed: int = 42


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
    energy_opportunity_cost: float = Field(0.0, description="Cost of the energy this strategy spends, s-equiv")
    strategic_value: float = Field(
        0.0,
        description="pace + current/future opportunity value - energy opportunity cost. "
        "Ranking key when Future Energy Value is active; higher is better.",
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

        for strategy_name, delay in strategies:
            mean_total, std_total, end_soc, spent, attacked = self._simulate_strategy(
                strategy_name, delay, gate_result, rival_estimate,
                start_soc_mj=start_soc_mj, reserve_floor_mj=reserve_floor_mj,
            )
            ci_lower = float(mean_total - 1.96 * std_total / np.sqrt(cfg.n_iterations))

            cur_val = fut_val = e_cost = strat_val = 0.0
            if fev:
                strength = self._window_strength.get(max(delay, 0), 1.0)
                opp_value = overtake_reward_s * strength if attacked else 0.0
                if delay == 0:
                    cur_val = opp_value
                elif delay > 0:
                    # value of the later window, plus a bias when the window is improving
                    fut_val = opp_value + future_window_bias
                # marginal energy value rises as the horizon-end reserve nears the floor
                scarcity = max(0.0, (2.0 * reserve_floor_mj - end_soc) / max(reserve_floor_mj, 1e-6))
                e_cost = spent * 0.08 * (1.0 + scarcity)
                strat_val = (-mean_total) + cur_val + fut_val - e_cost

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

    def _simulate_strategy(
        self,
        strategy_name: str,
        delay: int,
        gate_result: GateResult,
        rival_estimate: Optional[RivalSocEstimate],
        *,
        start_soc_mj: Optional[float] = None,
        reserve_floor_mj: float = 1.0,
    ) -> tuple[float, float, float, float, bool]:
        """
        Simulate strategy over the horizon.
        Returns (mean_total_delta, std_total_delta, end_soc, energy_spent, attacked).
        When start_soc_mj is None, energy carry is skipped and the laptime figures
        are byte-identical to the pre-FEV implementation.
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
        _HARVEST_PER_LAP = 0.35  # MODEL_ASSUMPTION: gentle recovery on non-attack laps

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

            if soc is not None:
                soc = max(0.0, soc - max(0.0, dyn.energy_cost_mj) + (_HARVEST_PER_LAP if not is_attack else 0.0))
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
