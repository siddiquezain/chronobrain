"""
config.py — the single configuration object for the whole decision pipeline.

`DecisionConfig` is a frozen dataclass. It carries the seed (part of the config,
per the determinism contract), the thresholds the fusion step uses, and it
constructs the Stack A per-stage configs so every stage is seeded consistently.

Nothing in here is a claimed FIA regulation — those live in `rule_gate.GateConfig`
and are catalogued by provenance in `app/regulation/` and `docs/regulation.md`.
The numbers here are ChronoPace decision-policy choices (MODEL_ASSUMPTION).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from confidence_gate import ConfidenceGateConfig
from opportunity_engine import HorizonConfig
from planner import PlannerConfig
from rival_estimator import RivalEstimatorConfig
from rule_gate import GateConfig


@dataclass(frozen=True)
class DecisionConfig:
    # --- determinism ---
    seed: int = 42
    n_iterations: int = 10_000  # fixed Monte Carlo rollout count (see planner docstring)

    # --- opportunity horizon ---
    horizon_laps: int = 10
    delay_laps: tuple[int, ...] = (0, 2, 5)
    # A WAIT_N strategy only overrides an aggressive single-lap call if it beats
    # ATTACK_NOW's aggregated horizon value by at least this margin (seconds).
    # MODEL_ASSUMPTION — tune against calibrated dynamics.
    horizon_decisive_margin_s: float = 0.15

    # --- energy policy (MODEL_ASSUMPTION) ---
    low_reserve_mj: float = 2.0          # below this, prefer CONSERVE over BALANCED when holding
    reserve_floor_mj: float = 1.0        # a strategy may not deploy below this in the horizon
    rival_low_soc_mj: float = 3.0        # bucket edges for rival LOW / MEDIUM / HIGH
    rival_high_soc_mj: float = 6.0

    # --- event-time window / trends ---
    window_laps: int = 5

    # --- future energy value (opportunity horizon) ---
    overtake_reward_s: float = 0.3       # laptime-equivalent value of taking a window
    future_window_bias_s: float = 0.12   # bonus for waiting when the window is IMPROVING

    # --- data quality -> confidence ---
    data_quality_floor: float = 0.6      # below this the confidence gate abstains

    # --- ML overtake-probability salience thresholds (MODEL_ASSUMPTION) ---
    ml_prob_high: float = 0.60
    ml_prob_low: float = 0.35

    def gate_config(self) -> GateConfig:
        return GateConfig()

    def planner_config(self) -> PlannerConfig:
        return PlannerConfig(n_iterations=self.n_iterations, default_seed=self.seed)

    def rival_config(self) -> RivalEstimatorConfig:
        return RivalEstimatorConfig(default_seed=self.seed)

    def confidence_config(self) -> ConfidenceGateConfig:
        return ConfidenceGateConfig(data_quality_pass_threshold=self.data_quality_floor)

    def horizon_config(self) -> HorizonConfig:
        return HorizonConfig(
            delay_laps=list(self.delay_laps),
            horizon_laps=self.horizon_laps,
            n_iterations=self.n_iterations,
            default_seed=self.seed,
        )
