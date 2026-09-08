"""
planner.py — Stage 2: Monte Carlo Planner

Ranks legal deployment modes by simulated laptime outcomes using Monte Carlo sampling.
This stage evaluates VIABILITY — it does not re-check legality (that was Stage 1).

Key design decisions:
- n_iterations is FIXED at 10,000 and echoed into PlannerResult. A variable rollout
  count would let the confidence gate manufacture statistical significance by running
  longer — fixing it is what makes Stage 3's test honest.
- Seeded determinism via SeedSequence.spawn(5) with FIXED canonical mode order,
  NOT legal_modes order. Without this, a mode's samples would silently depend on
  which other modes happened to be legal that call, breaking replay determinism.
  Must be .spawn(5) — a leftover 4 would silently corrupt the newest mode's stream.
- ModeDynamics priors are ENGINEERED, NOT MEASURED from real car data. Never present
  numeric output as validated without repeating this caveat.
- The prior is only the STARTING point of each draw. `_simulate_mode` then adjusts
  it by live race state from `PlanningContext` (gap to the car ahead/behind, own
  modelled SoC, opportunity strength, rival energy estimate) so the same planner
  gives different outcomes as the situation changes — and so a mode cannot rank
  first on a favourable static prior when there is nothing to gain from it. Those
  adjustment coefficients (`PlannerConfig`) are ENGINEERED laptime-equivalents too.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from rule_gate import DeploymentMode, GateResult, GateConfig
from rival_estimator import RivalSocEstimate

_DEFAULT_GATE_CONFIG = GateConfig()

# Canonical mode order — FIXED, never use legal_modes order for RNG stream assignment.
# This guarantees a mode's Monte Carlo samples are independent of which other modes
# are legal in a given call — required for seeded-determinism replay ("Time Machine").
_CANONICAL_MODE_ORDER = [
    DeploymentMode.CONSERVE_MODE,
    DeploymentMode.BALANCED_MODE,
    DeploymentMode.ARM_OVERTAKE_MODE,
    DeploymentMode.USE_OVERTAKE_BONUS_MODE,
    DeploymentMode.PUSH_MODE,
]


@dataclass(frozen=True)
class ModeDynamics:
    """
    Racecraft priors for a single deployment mode.

    mean_laptime_delta_s: negative = faster/better. BALANCED_MODE = 0.0 baseline.
    overtake_success_prob semantics differ by mode:
        - ARM_OVERTAKE_MODE: probability of *qualifying* (closing to gap threshold)
        - USE_OVERTAKE_BONUS_MODE, PUSH_MODE: probability of *completing* an overtake
        - CONSERVE_MODE, BALANCED_MODE: 0 (not attempting)
    All values ENGINEERED — not measured from real car data.
    """

    mean_laptime_delta_s: float
    std_laptime_delta_s: float
    overtake_success_prob: float
    energy_cost_mj: float


# ponytail: these numbers need calibration against real race data before production use
DEFAULT_MODE_DYNAMICS: dict[DeploymentMode, ModeDynamics] = {
    DeploymentMode.CONSERVE_MODE: ModeDynamics(
        mean_laptime_delta_s=+0.8,
        std_laptime_delta_s=0.3,
        overtake_success_prob=0.0,
        energy_cost_mj=-0.5,
    ),
    DeploymentMode.BALANCED_MODE: ModeDynamics(
        mean_laptime_delta_s=0.0,
        std_laptime_delta_s=0.3,
        overtake_success_prob=0.0,
        energy_cost_mj=0.0,
    ),
    DeploymentMode.ARM_OVERTAKE_MODE: ModeDynamics(
        mean_laptime_delta_s=-0.4,
        std_laptime_delta_s=0.4,
        overtake_success_prob=0.45,
        energy_cost_mj=0.8,
    ),
    DeploymentMode.USE_OVERTAKE_BONUS_MODE: ModeDynamics(
        mean_laptime_delta_s=-0.9,
        std_laptime_delta_s=0.5,
        overtake_success_prob=0.62,
        energy_cost_mj=1.5,
    ),
    DeploymentMode.PUSH_MODE: ModeDynamics(
        mean_laptime_delta_s=-0.6,
        std_laptime_delta_s=0.4,
        overtake_success_prob=0.35,
        energy_cost_mj=1.2,
    ),
}


@dataclass(frozen=True)
class PlannerConfig:
    n_iterations: int = 10_000  # FIXED — never vary; see module docstring
    default_seed: int = 42
    overtake_gap_max_bonus: float = 0.3
    failed_overtake_penalty_s: float = 0.4
    # Deliberately modest — single noisy rival estimate shouldn't swing outcomes much
    defense_penalty_weight: float = 0.3
    # Uncertainty growth per lap of lookahead — makes WAIT_5 less confident than WAIT_2
    horizon_uncertainty_growth: float = 0.15
    taper_normal_start_kmh: float = 290.0
    taper_normal_end_kmh: float = 355.0
    taper_overtake_full_power_end_kmh: float = 337.0

    # --- live-state responsiveness (all ENGINEERED laptime-equivalents, MODEL_ASSUMPTION) ---
    # A car within `in_range_plateau_s` is fully attackable/defendable (matches the
    # 1.0 s overtake-proximity rule); the mode-specific benefit then fades linearly
    # to zero by `target_range_s`, and is zero when there is no car on that side.
    in_range_plateau_s: float = 1.0
    target_range_s: float = 3.0
    # Strategic laptime-equivalent price (s per MJ) of spending ERS with NO
    # positional payoff (no target in range, not defending). Not a physical lap
    # delta — it is what stops an attack/PUSH mode ranking first on raw deployment
    # pace when there is nothing to gain, so an infeasible-in-context mode cannot
    # dominate the final decision on a static prior alone.
    unrewarded_energy_price_s_per_mj: float = 0.7
    # How much a low own-SoC erodes a deployment-heavy mode's pace (s per MJ of the
    # mode's own energy cost, scaled by the SoC deficit fraction).
    low_soc_pace_erosion_s_per_mj: float = 0.18
    # Value (s) of PUSH_MODE when it is actually defending a car within range.
    defensive_push_value_s: float = 0.3


@dataclass
class PlanningContext:
    """
    Context fed into Stage 2 alongside the legal modes.
    Does NOT carry overtake_qualified_last_lap — that belongs exclusively to
    Stage 1's legal_modes output; duplicating here creates two sources of truth.

    Every field below is a LIVE race-state input; supplying them makes each mode's
    simulated outcome respond to the current situation (see `_simulate_mode`).
    With no fields set the planner treats it as "no car in range, own energy
    unknown" — an attack/PUSH mode then earns no overtake payoff, so it cannot
    rank first on its static prior alone. Output is deterministic either way.
    """

    gap_to_car_ahead_s: Optional[float] = None
    rival_soc_estimate: Optional[RivalSocEstimate] = None
    # Optional scalar probability the rival will actively defend (0..1), estimated
    # deterministically upstream from observable behaviour. None -> not modelled,
    # behaviour byte-identical to before this field existed.
    p_defend: Optional[float] = None
    # rearward gap — PUSH_MODE's defensive rationale
    gap_to_car_behind_s: Optional[float] = None
    # our own modelled SoC and the policy "comfortable" reserve
    own_soc_mj: Optional[float] = None
    low_reserve_mj: float = 2.0
    # current single-lap overtake-opportunity strength (0..1), e.g. ml_overtake_prob
    opportunity_strength: Optional[float] = None


class ModeProjection(BaseModel):
    mode: DeploymentMode
    mean_laptime_delta_s: float = Field(
        ..., description="Mean laptime delta vs BALANCED baseline (s). Negative = faster."
    )
    std_laptime_delta_s: float
    overtake_probability: float = Field(..., ge=0.0, le=1.0)
    sharpe_ratio: float = Field(
        ..., description="Risk-adjusted metric. Capped at ±999 when variance near zero."
    )
    energy_cost_mj: float


class PlannerResult(BaseModel):
    run_id: int = Field(..., description="Monotonic counter — key for get_raw_samples()")
    ranked_modes: list[ModeProjection] = Field(
        ..., description="Legal modes ranked best-first (lowest mean delta)"
    )
    recommended_mode: DeploymentMode
    n_iterations: int


class MonteCarloPlanner:
    """
    Stage 2: Ranks legal deployment modes by simulated laptime outcomes.

    Architecture:
        Legal modes (from Stage 1) + PlanningContext
            → Monte Carlo simulation (n_iterations draws per mode)
            → PlannerResult (ranked ModeProjections, recommended_mode)
            → Stage 3 (ConfidenceGate)
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        dynamics: Optional[dict[DeploymentMode, ModeDynamics]] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or PlannerConfig()
        self.dynamics = dynamics or DEFAULT_MODE_DYNAMICS
        self._seed = seed if seed is not None else self.config.default_seed
        self._run_id_counter = itertools.count(1)
        # Keyed by (run_id, mode) — prevents stale-result corruption
        self._raw_samples: dict[tuple[int, DeploymentMode], np.ndarray] = {}

    def plan(
        self,
        gate_result: GateResult,
        context: Optional[PlanningContext] = None,
    ) -> PlannerResult:
        """
        Run Monte Carlo simulation over legal modes and return ranked projections.
        """
        ctx = context or PlanningContext()
        cfg = self.config
        run_id = next(self._run_id_counter)
        # Evict samples from previous runs — stale run_ids then raise in get_raw_samples
        self._raw_samples.clear()

        # Spawn 5 child RNG streams in CANONICAL mode order (not legal_modes order)
        seed_seq = np.random.SeedSequence(self._seed + run_id)
        streams = {
            mode: np.random.default_rng(child)
            for mode, child in zip(_CANONICAL_MODE_ORDER, seed_seq.spawn(5))
        }

        projections: list[ModeProjection] = []

        for mode in gate_result.legal_modes:
            dyn = self.dynamics[mode]
            rng = streams[mode]

            samples = self._simulate_mode(mode, dyn, ctx, rng)
            self._raw_samples[(run_id, mode)] = samples

            mean = float(np.mean(samples))
            std = float(np.std(samples))

            # Sharpe: -mean/std — capped at ±999 to prevent inf/NaN when std ≈ 0
            if std < 1e-9:
                sharpe = 999.0 if mean < 0 else -999.0
            else:
                sharpe = float(np.clip(-mean / std, -999.0, 999.0))

            projections.append(
                ModeProjection(
                    mode=mode,
                    mean_laptime_delta_s=round(mean, 6),
                    std_laptime_delta_s=round(std, 6),
                    overtake_probability=round(float(np.mean(samples < 0)), 4),
                    sharpe_ratio=round(sharpe, 4),
                    energy_cost_mj=dyn.energy_cost_mj,
                )
            )

        # Rank best-first (lowest mean delta = fastest)
        projections.sort(key=lambda p: p.mean_laptime_delta_s)
        recommended = (
            projections[0].mode if projections else DeploymentMode.BALANCED_MODE
        )

        return PlannerResult(
            run_id=run_id,
            ranked_modes=projections,
            recommended_mode=recommended,
            n_iterations=cfg.n_iterations,
        )

    @staticmethod
    def _range_factor(gap: Optional[float], plateau_s: float, range_s: float) -> float:
        """1.0 for a car within `plateau_s`, fading linearly to 0.0 by `range_s`,
        and 0.0 when there is no car on that side (`gap is None`)."""
        if gap is None:
            return 0.0
        if gap <= plateau_s:
            return 1.0
        if gap >= range_s:
            return 0.0
        return 1.0 - (gap - plateau_s) / (range_s - plateau_s)

    def _simulate_mode(
        self,
        mode: DeploymentMode,
        dyn: ModeDynamics,
        ctx: PlanningContext,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate n_iterations laptime draws for one mode.

        The base draw is the mode's engineered prior. It is then adjusted by LIVE
        race state so that the SAME planner produces DIFFERENT outcomes as the
        situation changes:
          * an attack/PUSH mode only realises its overtake payoff if there is a
            car within range (ahead to attack, or behind for PUSH to defend);
          * opportunity strength and the rival energy estimate scale the realised
            success rate;
          * ERS spent with no positional payoff carries a strategic laptime price
            (this is what stops a static-prior mode from dominating out of context);
          * a low own-SoC erodes a deployment-heavy mode's pace;
          * PUSH gains real value when it is genuinely defending.
        Deterministic given the mode's RNG stream regardless of context.
        """
        cfg = self.config
        n = cfg.n_iterations

        samples = rng.normal(dyn.mean_laptime_delta_s, dyn.std_laptime_delta_s, n)

        target_ahead = self._range_factor(
            ctx.gap_to_car_ahead_s, cfg.in_range_plateau_s, cfg.target_range_s
        )
        defending = (
            self._range_factor(ctx.gap_to_car_behind_s, cfg.in_range_plateau_s, cfg.target_range_s)
            if mode == DeploymentMode.PUSH_MODE
            else 0.0
        )
        payoff = max(target_ahead, defending)  # how much of this mode's rationale applies now

        if dyn.overtake_success_prob > 0:
            eff_prob = self._effective_overtake_prob(mode, dyn, ctx, rng, n)
            if ctx.opportunity_strength is not None:
                eff_prob = eff_prob * (0.5 + 0.5 * float(np.clip(ctx.opportunity_strength, 0.0, 1.0)))
            eff_prob = eff_prob * payoff  # no car in range -> payoff cannot be realised
            success = rng.random(n) < eff_prob
            samples -= success * cfg.overtake_gap_max_bonus
            samples += (~success) * cfg.failed_overtake_penalty_s * (eff_prob > 0).astype(float)

        # ERS spent with no positional payoff has a strategic laptime-equivalent price
        if dyn.energy_cost_mj > 0:
            samples = samples + (1.0 - payoff) * dyn.energy_cost_mj * cfg.unrewarded_energy_price_s_per_mj

        # a low own-SoC erodes a deployment-heavy mode's realisable pace (lift & coast)
        if dyn.energy_cost_mj > 0 and ctx.own_soc_mj is not None and ctx.low_reserve_mj > 0:
            deficit = float(
                np.clip((ctx.low_reserve_mj - ctx.own_soc_mj) / ctx.low_reserve_mj, 0.0, 1.0)
            )
            samples = samples + deficit * dyn.energy_cost_mj * cfg.low_soc_pace_erosion_s_per_mj

        # PUSH is genuinely worth its energy when it is defending a car within range
        if mode == DeploymentMode.PUSH_MODE and defending > 0.0:
            samples = samples - defending * cfg.defensive_push_value_s

        return samples

    def _effective_overtake_prob(
        self,
        mode: DeploymentMode,
        dyn: ModeDynamics,
        ctx: PlanningContext,
        rng: np.random.Generator,
        n: int,
    ) -> np.ndarray:
        """
        Per-iteration effective overtake probability, optionally modulated by rival SoC.
        Applies only to ARM and USE_BONUS modes when rival estimate is provided.
        """
        base_prob = dyn.overtake_success_prob
        cfg = self.config

        if ctx.rival_soc_estimate is not None and mode in (
            DeploymentMode.ARM_OVERTAKE_MODE,
            DeploymentMode.USE_OVERTAKE_BONUS_MODE,
            DeploymentMode.PUSH_MODE,
        ):
            rival_soc = rng.normal(
                ctx.rival_soc_estimate.mean_soc_mj,
                ctx.rival_soc_estimate.std_soc_mj,
                n,
            )
            rival_soc = np.clip(rival_soc, 0.0, _DEFAULT_GATE_CONFIG.max_deployment_per_lap_mj)
            defense_factor = rival_soc / _DEFAULT_GATE_CONFIG.max_deployment_per_lap_mj
            eff_prob = base_prob - cfg.defense_penalty_weight * defense_factor
            if ctx.p_defend is not None:
                # explicit behavioural signal, on top of the SoC-derived defense factor
                eff_prob = eff_prob - cfg.defense_penalty_weight * float(ctx.p_defend)
            return np.clip(eff_prob, 0.0, 1.0)

        return np.full(n, base_prob)

    def get_raw_samples(
        self, result: PlannerResult, mode: DeploymentMode
    ) -> np.ndarray:
        """
        Return raw simulation samples for a specific mode from a specific run.
        Raises ValueError on unknown/stale run_id — prevents silent corruption.
        """
        key = (result.run_id, mode)
        if key not in self._raw_samples:
            raise ValueError(
                f"No samples for run_id={result.run_id}, mode={mode.value}. "
                "This run_id may be stale or the mode was not legal in that run."
            )
        return self._raw_samples[key]
