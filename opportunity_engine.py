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


class HorizonResult(BaseModel):
    recommended_strategy: str
    ranked_strategies: list[StrategyOutcome]
    foregone_strategy: str
    foregone_value_gap_s: float = Field(
        ...,
        description="How much better the chosen strategy is vs runner-up (s)",
    )
    uncertainty_note: str


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
    ) -> HorizonResult:
        """
        Evaluate candidate multi-lap strategies and return ranked outcomes.

        `window_strength_by_delay` (optional): {delay_laps: strength} where strength
        scales the laptime advantage of that strategy's ARM/USE_OVERTAKE_BONUS laps.
        1.0 = neutral. A caller that believes the window will be stronger in 2 laps
        than now passes e.g. {0: 1.0, 2: 1.3}. Omitted -> all strengths 1.0 and
        behaviour is byte-identical to before this parameter existed.
        """
        cfg = self.config
        self._window_strength = window_strength_by_delay or {}

        strategies: list[tuple[str, int]] = [(HOLD_STRATEGY, -1)]
        for d in sorted(cfg.delay_laps):
            name = ATTACK_NOW_STRATEGY if d == 0 else f"WAIT_{d}"
            strategies.append((name, d))

        outcomes: list[StrategyOutcome] = []

        for strategy_name, delay in strategies:
            mean_total, std_total = self._simulate_strategy(
                strategy_name, delay, gate_result, rival_estimate
            )
            ci_lower = float(mean_total - 1.96 * std_total / np.sqrt(cfg.n_iterations))
            outcomes.append(
                StrategyOutcome(
                    strategy_name=strategy_name,
                    delay_laps=delay,
                    mean_horizon_delta_s=round(mean_total, 4),
                    std_horizon_delta_s=round(std_total, 4),
                    confidence_ci_lower_s=round(ci_lower, 4),
                )
            )

        # Rank by mean delta (lowest = fastest/best)
        outcomes.sort(key=lambda o: o.mean_horizon_delta_s)

        best = outcomes[0]
        runner_up = outcomes[1] if len(outcomes) > 1 else outcomes[0]
        value_gap = float(abs(runner_up.mean_horizon_delta_s - best.mean_horizon_delta_s))

        return HorizonResult(
            recommended_strategy=best.strategy_name,
            ranked_strategies=outcomes,
            foregone_strategy=runner_up.strategy_name,
            foregone_value_gap_s=round(value_gap, 4),
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
    ) -> tuple[float, float]:
        """Simulate strategy over horizon. Returns (mean_total_delta, std_total_delta)."""
        cfg = self.config
        pc = self._planner_config
        n = cfg.n_iterations
        horizon = cfg.horizon_laps

        total_samples = np.zeros(n)

        strength = getattr(self, "_window_strength", {}).get(max(delay, 0), 1.0)

        for lap_offset in range(horizon):
            mode = self._strategy_lap_mode(strategy_name, delay, lap_offset, gate_result)
            uncertainty_scale = 1.0 + pc.horizon_uncertainty_growth * lap_offset
            dyn = DEFAULT_MODE_DYNAMICS[mode]

            lap_samples = self._rng.normal(
                dyn.mean_laptime_delta_s,
                dyn.std_laptime_delta_s * uncertainty_scale,
                n,
            )
            # Scale the attack laps' advantage by how strong the window is at this
            # strategy's chosen moment (1.0 when the caller supplies nothing).
            if strength != 1.0 and mode in (
                DeploymentMode.ARM_OVERTAKE_MODE,
                DeploymentMode.USE_OVERTAKE_BONUS_MODE,
            ):
                lap_samples = lap_samples * strength

            total_samples += lap_samples

        return float(np.mean(total_samples)), float(np.std(total_samples))

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
