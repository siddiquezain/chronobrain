"""
confidence_gate.py — Stage 3: Confidence Gate

Two-part significance test + DCLI + rival-confidence + data-quality check
(five gates; the fifth is optional and defaulted so older callers are unaffected).

WHY a plain t-test doesn't work here:
  With n_iterations fixed at 10,000, standard error s/√n is tiny — almost any
  nonzero true difference clears t≥2.0 (we'd be measuring "is n big enough",
  always yes). Abstention — this project's signature behavior — would become
  nearly impossible to trigger.

The fix — BOTH statistical reliability AND practical significance must pass,
plus DCLI and rival-confidence checks, before a non-BALANCED_MODE recommendation
reaches Stage 4.

SIGN CONVENTION WARNING — highest correctness risk in this entire project:
  "Negative delta = faster/better."
  The difference MUST be:
      diff = mean(runner_up) - mean(top)   ← runner_up FIRST
  Subtracting in the other order silently inverts the gate's pass/fail logic
  with no error. Tests check this explicitly and prominently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from planner import MonteCarloPlanner, PlannerResult
from rival_estimator import RivalSocEstimate
from rule_gate import DeploymentMode


@dataclass(frozen=True)
class ConfidenceGateConfig:
    confidence_level: float = 0.95

    # Practical significance: minimum actionable laptime delta (seconds)
    # ponytail: illustrative threshold — tune against real race data
    min_actionable_laptime_delta_s: float = 0.05

    # Additional check for overtake modes (percentage points)
    min_actionable_overtake_prob_pp: float = 0.03

    # DCLI — fully invented, flagged as illustrative
    dcli_time_weight: float = 0.4
    dcli_variability_weight: float = 0.35
    dcli_proximity_weight: float = 0.25
    dcli_pass_threshold: float = 60.0

    # Rival confidence threshold — raised from 1.5 to 2.5 to match the Z-score
    # model's noise floor. The Z-score filter normalises against the driver's own
    # baseline; on constant-SoC synthetic scenarios it provides no absolute-SoC
    # signal, so posterior std stabilises ~1.6-2.5 MJ depending on lap count.
    # 2.5 is the empirical worst-case (< 10 laps of history). The gate still fires
    # for genuinely degenerate posteriors above this ceiling.
    # ponytail: recalibrate against real-data posteriors once FastF1 baseline is ready.
    rival_confidence_threshold_mj: float = 2.5

    # Data-quality gate: below this quality_score (0..1) an aggressive
    # recommendation is not trusted. MODEL_ASSUMPTION.
    data_quality_pass_threshold: float = 0.6


class DriverLoadInput(BaseModel):
    laps_since_last_mode_change: float = Field(..., ge=0.0)
    recent_laptime_std_s: float = Field(..., ge=0.0)
    gap_to_car_ahead_s: Optional[float] = Field(None, ge=0.0)


class ConfidenceGateResult(BaseModel):
    """
    Output of ConfidenceGate.evaluate().
    Carries both the (possibly overridden) recommended_mode and the original
    stage2_recommended_mode so Stage 4 can narrate *what changed*.
    """

    recommended_mode: DeploymentMode
    stage2_recommended_mode: DeploymentMode
    runner_up_mode: Optional[DeploymentMode]

    t_statistic: float
    degrees_of_freedom: float
    ci_lower_bound_s: float  # actual bound value — Stage 4 uses this, not just a boolean

    statistical_reliability_passed: bool
    practical_significance_passed: bool

    dcli_score: float
    dcli_passed: bool

    rival_confidence_passed: bool
    data_quality_passed: bool = True

    overridden: bool
    override_reason: str  # names ALL failing gates
    n_iterations: int


class ConfidenceGate:
    """
    Stage 3: Ensures Stage 2's recommendation is statistically reliable AND
    practically significant before it reaches Stage 4.

    If any gate fails, recommendation is overridden to BALANCED_MODE (the safe
    fallback), and overridden=True with ALL failing gates named.
    """

    def __init__(
        self,
        config: Optional[ConfidenceGateConfig] = None,
        planner: Optional[MonteCarloPlanner] = None,
    ):
        self.config = config or ConfidenceGateConfig()
        self._planner = planner

    def evaluate(
        self,
        planner_result: PlannerResult,
        driver_load: Optional[DriverLoadInput] = None,
        rival_estimate: Optional[RivalSocEstimate] = None,
        planner: Optional[MonteCarloPlanner] = None,
        data_quality_score: float = 1.0,
        opportunity_uncertain: bool = False,
    ) -> ConfidenceGateResult:
        """
        Evaluate the gates and override to BALANCED_MODE if any fail.

        `data_quality_score` (0..1) and `opportunity_uncertain` are optional inputs
        from the Data Quality Gate and the Opportunity Horizon. Bad/stale telemetry
        or an ambiguous opportunity makes the gate abstain — the system's way of
        saying it does not know enough. Defaults (1.0, False) leave behaviour
        unchanged.
        """
        cfg = self.config
        _planner = planner or self._planner

        ranked = planner_result.ranked_modes
        top_mode = planner_result.recommended_mode

        runner_up_mode: Optional[DeploymentMode] = None
        t_stat = 0.0
        df = 0.0
        ci_lower = -999.0
        stat_passed = False
        prac_passed = False

        if len(ranked) >= 2 and _planner is not None:
            runner_up_mode = ranked[1].mode

            try:
                top_samples = _planner.get_raw_samples(planner_result, top_mode)
                runner_up_samples = _planner.get_raw_samples(planner_result, runner_up_mode)

                # SIGN CONVENTION: diff = mean(runner_up) - mean(top), runner_up FIRST.
                # Under "negative = faster", runner_up has higher (worse) mean,
                # so diff > 0 means top is genuinely better.
                t_stat, _ = stats.ttest_ind(top_samples, runner_up_samples, equal_var=False)
                df = float(self._welch_df(top_samples, runner_up_samples))

                diff = float(np.mean(runner_up_samples) - np.mean(top_samples))
                se = float(
                    np.sqrt(
                        np.var(top_samples) / len(top_samples)
                        + np.var(runner_up_samples) / len(runner_up_samples)
                    )
                )
                t_crit = float(stats.t.ppf(cfg.confidence_level, df))
                ci_lower = diff - t_crit * se

                # Gate 1: Statistical reliability (one-sided CI lower bound > practical floor)
                stat_passed = ci_lower > cfg.min_actionable_laptime_delta_s

                # Gate 2: Practical significance
                top_proj = next(p for p in ranked if p.mode == top_mode)
                runner_proj = ranked[1]
                prac_passed = abs(diff) >= cfg.min_actionable_laptime_delta_s

                # Extra overtake-mode check
                if top_mode in (
                    DeploymentMode.ARM_OVERTAKE_MODE,
                    DeploymentMode.USE_OVERTAKE_BONUS_MODE,
                ):
                    prob_diff = abs(
                        top_proj.overtake_probability - runner_proj.overtake_probability
                    )
                    prac_passed = prac_passed and (
                        prob_diff >= cfg.min_actionable_overtake_prob_pp
                    )

            except (ValueError, StopIteration):
                pass

        elif len(ranked) == 1 and _planner is not None:
            # Only one legal mode — auto-pass significance (no comparison possible)
            stat_passed = True
            prac_passed = True

        # Gate 3: DCLI
        dcli_score = self._compute_dcli(driver_load)
        dcli_passed = dcli_score < cfg.dcli_pass_threshold

        # Gate 4: Rival confidence
        # If baseline not yet ready (< min_obs observations), the estimate is
        # prior-dominated — gating on it would block decisions in the first few laps
        # of any race before the filter has seen enough data. Skip the gate until
        # the Z-score baseline is established and the estimate is meaningful.
        rival_passed = (
            rival_estimate is None
            or not rival_estimate.baseline_ready
            or rival_estimate.std_soc_mj <= cfg.rival_confidence_threshold_mj
        )

        # Gate 5: Data quality / opportunity clarity
        data_quality_passed = (
            data_quality_score >= cfg.data_quality_pass_threshold and not opportunity_uncertain
        )

        failing = []
        if not stat_passed:
            failing.append("STATISTICAL_RELIABILITY")
        if not prac_passed:
            failing.append("PRACTICAL_SIGNIFICANCE")
        if not dcli_passed:
            failing.append("DCLI")
        if not rival_passed:
            failing.append("RIVAL_CONFIDENCE")
        if not data_quality_passed:
            failing.append("DATA_QUALITY")

        overridden = bool(failing)
        final_mode = DeploymentMode.BALANCED_MODE if overridden else top_mode

        return ConfidenceGateResult(
            recommended_mode=final_mode,
            stage2_recommended_mode=top_mode,
            runner_up_mode=runner_up_mode,
            t_statistic=round(float(t_stat), 4),
            degrees_of_freedom=round(df, 2),
            ci_lower_bound_s=round(ci_lower, 6),
            statistical_reliability_passed=stat_passed,
            practical_significance_passed=prac_passed,
            dcli_score=round(dcli_score, 2),
            dcli_passed=dcli_passed,
            rival_confidence_passed=rival_passed,
            data_quality_passed=data_quality_passed,
            overridden=overridden,
            override_reason=", ".join(failing) if failing else "",
            n_iterations=planner_result.n_iterations,
        )

    def _compute_dcli(self, load: Optional[DriverLoadInput]) -> float:
        """
        Driver Cognitive Load Index — 0-100 score.
        Fully invented / illustrative — no source document for this formula.
        """
        if load is None:
            return 0.0

        cfg = self.config
        time_score = float(np.clip(1.0 - load.laps_since_last_mode_change / 10.0, 0.0, 1.0))
        var_score = float(np.clip(load.recent_laptime_std_s / 1.0, 0.0, 1.0))
        prox_score = (
            float(np.clip(1.0 - load.gap_to_car_ahead_s / 2.0, 0.0, 1.0))
            if load.gap_to_car_ahead_s is not None
            else 0.0
        )

        dcli = 100.0 * (
            cfg.dcli_time_weight * time_score
            + cfg.dcli_variability_weight * var_score
            + cfg.dcli_proximity_weight * prox_score
        )
        return float(np.clip(dcli, 0.0, 100.0))

    @staticmethod
    def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
        """Welch-Satterthwaite degrees of freedom."""
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        na, nb = len(a), len(b)
        if va == 0 and vb == 0:
            return float(na + nb - 2)
        numerator = (va / na + vb / nb) ** 2
        denominator = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        return float(numerator / denominator) if denominator > 0 else float(na + nb - 2)
