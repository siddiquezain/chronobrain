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


@dataclass
class PlanningContext:
    """
    Context fed into Stage 2 alongside the legal modes.
    Does NOT carry overtake_qualified_last_lap — that belongs exclusively to
    Stage 1's legal_modes output; duplicating here creates two sources of truth.
    """

    gap_to_car_ahead_s: Optional[float] = None
    rival_soc_estimate: Optional[RivalSocEstimate] = None


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

    def _simulate_mode(
        self,
        mode: DeploymentMode,
        dyn: ModeDynamics,
        ctx: PlanningContext,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate n_iterations laptime draws for one mode."""
        cfg = self.config
        n = cfg.n_iterations

        samples = rng.normal(dyn.mean_laptime_delta_s, dyn.std_laptime_delta_s, n)

        if dyn.overtake_success_prob > 0:
            eff_prob = self._effective_overtake_prob(mode, dyn, ctx, rng, n)
            success = rng.random(n) < eff_prob
            samples -= success * cfg.overtake_gap_max_bonus
            samples += (~success) * cfg.failed_overtake_penalty_s * (eff_prob > 0).astype(float)

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
        ):
            rival_soc = rng.normal(
                ctx.rival_soc_estimate.mean_soc_mj,
                ctx.rival_soc_estimate.std_soc_mj,
                n,
            )
            rival_soc = np.clip(rival_soc, 0.0, _DEFAULT_GATE_CONFIG.max_deployment_per_lap_mj)
            defense_factor = rival_soc / _DEFAULT_GATE_CONFIG.max_deployment_per_lap_mj
            eff_prob = base_prob - cfg.defense_penalty_weight * defense_factor
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
