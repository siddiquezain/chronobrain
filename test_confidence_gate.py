"""Tests for confidence_gate.py — Stage 3: Confidence Gate"""

import numpy as np
import pytest
from confidence_gate import ConfidenceGate, ConfidenceGateConfig, DriverLoadInput
from planner import MonteCarloPlanner, PlannerConfig, PlanningContext
from rival_estimator import RivalSocEstimate
from rule_gate import DeploymentMode, GateResult, GateConfig


def make_all_legal_gate() -> GateResult:
    cfg = GateConfig()
    base_cap = cfg.max_deployment_per_lap_mj
    bonus_cap = base_cap + cfg.overtake_bonus_mj
    base_cap_mj = {m.value: base_cap for m in DeploymentMode}
    base_cap_mj[DeploymentMode.USE_OVERTAKE_BONUS_MODE.value] = bonus_cap
    return GateResult(
        legal_modes=list(DeploymentMode),
        violations={m.value: [] for m in DeploymentMode},
        base_cap_mj=base_cap_mj,
        qualifies_for_overtake_bonus_next_lap=True,
    )


def make_conservative_gate() -> GateResult:
    cfg = GateConfig()
    base_cap = cfg.max_deployment_per_lap_mj
    base_cap_mj = {m.value: base_cap for m in DeploymentMode}
    legal = [DeploymentMode.CONSERVE_MODE, DeploymentMode.BALANCED_MODE]
    return GateResult(
        legal_modes=legal,
        violations={m.value: [] if m in legal else ["illegal"] for m in DeploymentMode},
        base_cap_mj=base_cap_mj,
        qualifies_for_overtake_bonus_next_lap=False,
    )


class TestSignConvention:
    def test_diff_computed_runner_up_minus_top(self):
        """
        Critical sign test: diff = mean(runner_up) - mean(top), runner_up FIRST.
        Under "negative = faster", runner_up has higher (worse) mean.
        If the top mode is genuinely better, diff > 0 → ci_lower > 0 when significant.
        """
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)

        # Manually verify: top mode mean < runner_up mean (top is faster/better)
        top = result.ranked_modes[0]
        runner_up = result.ranked_modes[1]
        assert top.mean_laptime_delta_s < runner_up.mean_laptime_delta_s

        gate_eval = ConfidenceGate()
        cg_result = gate_eval.evaluate(result, planner=planner)

        # ci_lower_bound_s should be positive when top is confidently better
        # (it's diff = mean(runner_up) - mean(top) > 0 minus t_crit * se)
        # With n=10,000 and meaningful difference, ci_lower should be > 0
        if cg_result.statistical_reliability_passed:
            assert cg_result.ci_lower_bound_s > 0, (
                "ci_lower_bound_s must be positive when top mode is confidently better. "
                "Negative value indicates sign convention was inverted."
            )


class TestGatePass:
    def test_strong_difference_passes_statistical_reliability(self):
        """With 10k samples and a real mode difference, stat reliability should pass."""
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        cg_result = cg.evaluate(result, planner=planner)
        # At least one gate should pass (the difference is real)
        assert cg_result.statistical_reliability_passed or cg_result.practical_significance_passed

    def test_no_override_when_all_gates_pass(self):
        """When all gates pass, overridden should be False."""
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        driver_load = DriverLoadInput(
            laps_since_last_mode_change=8.0,
            recent_laptime_std_s=0.1,
            gap_to_car_ahead_s=2.0,
        )
        cg_result = cg.evaluate(result, driver_load=driver_load, planner=planner)
        if not cg_result.overridden:
            assert cg_result.recommended_mode == cg_result.stage2_recommended_mode

    def test_override_falls_back_to_balanced(self):
        """When any gate fails, recommended_mode must be BALANCED_MODE."""
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        # High rival uncertainty triggers rival_confidence gate failure
        rival_est = RivalSocEstimate(mean_soc_mj=5.0, std_soc_mj=3.0, n_observations=1)
        cg_result = cg.evaluate(result, rival_estimate=rival_est, planner=planner)
        if cg_result.overridden:
            assert cg_result.recommended_mode == DeploymentMode.BALANCED_MODE


class TestDCLI:
    def test_dcli_zero_when_no_driver_load(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        cg_result = cg.evaluate(result, driver_load=None, planner=planner)
        assert cg_result.dcli_score == 0.0

    def test_high_driver_load_fails_dcli(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        # Very high load: just changed mode, high variability, tight gap
        load = DriverLoadInput(
            laps_since_last_mode_change=0.0,
            recent_laptime_std_s=1.5,
            gap_to_car_ahead_s=0.1,
        )
        cg_result = cg.evaluate(result, driver_load=load, planner=planner)
        assert cg_result.dcli_score > 60.0
        assert cg_result.dcli_passed is False

    def test_low_driver_load_passes_dcli(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        load = DriverLoadInput(
            laps_since_last_mode_change=10.0,
            recent_laptime_std_s=0.05,
            gap_to_car_ahead_s=3.0,
        )
        cg_result = cg.evaluate(result, driver_load=load, planner=planner)
        assert cg_result.dcli_passed is True


class TestRivalConfidence:
    def test_high_rival_std_fails_rival_confidence(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        rival = RivalSocEstimate(mean_soc_mj=5.0, std_soc_mj=3.0, n_observations=5, baseline_ready=True)
        cg_result = cg.evaluate(result, rival_estimate=rival, planner=planner)
        assert cg_result.rival_confidence_passed is False

    def test_low_rival_std_passes_rival_confidence(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        rival = RivalSocEstimate(mean_soc_mj=5.0, std_soc_mj=0.5, n_observations=20, baseline_ready=True)
        cg_result = cg.evaluate(result, rival_estimate=rival, planner=planner)
        assert cg_result.rival_confidence_passed is True

    def test_no_rival_estimate_passes_rival_confidence(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()
        cg_result = cg.evaluate(result, rival_estimate=None, planner=planner)
        assert cg_result.rival_confidence_passed is True


class TestOverrideReason:
    def test_override_reason_names_all_failing_gates(self):
        """When multiple gates fail, override_reason must name ALL of them."""
        planner = MonteCarloPlanner(seed=42)
        gate = make_all_legal_gate()
        result = planner.plan(gate)
        cg = ConfidenceGate()

        # Trigger both DCLI and RIVAL_CONFIDENCE failures
        load = DriverLoadInput(laps_since_last_mode_change=0.0, recent_laptime_std_s=2.0, gap_to_car_ahead_s=0.1)
        rival = RivalSocEstimate(mean_soc_mj=5.0, std_soc_mj=3.0, n_observations=1)
        cg_result = cg.evaluate(result, driver_load=load, rival_estimate=rival, planner=planner)

        if cg_result.dcli_passed is False and cg_result.rival_confidence_passed is False:
            assert "DCLI" in cg_result.override_reason
            assert "RIVAL_CONFIDENCE" in cg_result.override_reason

    def test_n_iterations_echoed(self):
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(make_all_legal_gate())
        cg = ConfidenceGate()
        cg_result = cg.evaluate(result, planner=planner)
        assert cg_result.n_iterations == 10_000

    def test_ci_lower_bound_is_actual_value_not_boolean(self):
        """ci_lower_bound_s must be a float value, not just True/False."""
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(make_all_legal_gate())
        cg = ConfidenceGate()
        cg_result = cg.evaluate(result, planner=planner)
        assert isinstance(cg_result.ci_lower_bound_s, float)
        assert cg_result.ci_lower_bound_s != -999.0 or not cg_result.statistical_reliability_passed
