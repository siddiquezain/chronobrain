"""Tests for planner.py — Stage 2: Monte Carlo Planner"""

import numpy as np
import pytest
from planner import (
    DEFAULT_MODE_DYNAMICS,
    MonteCarloPlanner,
    PlannerConfig,
    PlannerResult,
    PlanningContext,
)
from rival_estimator import RivalSocEstimate, RivalStateEstimator
from rule_gate import DeploymentMode, GateResult, RegulatoryGate
from telemetry_simulator import TelemetryInput


def make_gate_result_all_legal() -> GateResult:
    """All five modes legal — for testing the full mode set."""
    from rule_gate import GateConfig
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


def make_gate_result_conservative() -> GateResult:
    """Only CONSERVE and BALANCED legal — for conservative scenario tests."""
    from rule_gate import GateConfig
    cfg = GateConfig()
    base_cap = cfg.max_deployment_per_lap_mj
    base_cap_mj = {m.value: base_cap for m in DeploymentMode}
    legal = [DeploymentMode.CONSERVE_MODE, DeploymentMode.BALANCED_MODE]
    violations = {m.value: ["illegal"] if m not in legal else [] for m in DeploymentMode}
    return GateResult(
        legal_modes=legal,
        violations=violations,
        base_cap_mj=base_cap_mj,
        qualifies_for_overtake_bonus_next_lap=False,
    )


class TestPlannerBasics:
    def test_returns_only_legal_modes(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_gate_result_conservative()
        result = planner.plan(gate)
        returned_modes = {p.mode for p in result.ranked_modes}
        assert returned_modes == {DeploymentMode.CONSERVE_MODE, DeploymentMode.BALANCED_MODE}

    def test_balanced_mean_near_zero(self):
        """BALANCED_MODE mean should be near 0.0 (it's the baseline)."""
        planner = MonteCarloPlanner(seed=42)
        gate = make_gate_result_all_legal()
        result = planner.plan(gate)
        balanced_proj = next(p for p in result.ranked_modes if p.mode == DeploymentMode.BALANCED_MODE)
        assert abs(balanced_proj.mean_laptime_delta_s) < 0.1

    def test_use_bonus_better_than_balanced(self):
        """USE_OVERTAKE_BONUS_MODE should have a more negative (better) mean than BALANCED."""
        planner = MonteCarloPlanner(seed=42)
        gate = make_gate_result_all_legal()
        result = planner.plan(gate)
        bonus = next(p for p in result.ranked_modes if p.mode == DeploymentMode.USE_OVERTAKE_BONUS_MODE)
        balanced = next(p for p in result.ranked_modes if p.mode == DeploymentMode.BALANCED_MODE)
        assert bonus.mean_laptime_delta_s < balanced.mean_laptime_delta_s

    def test_n_iterations_echoed(self):
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(make_gate_result_conservative())
        assert result.n_iterations == 10_000


class TestSeededDeterminism:
    def test_same_seed_same_result(self):
        gate = make_gate_result_all_legal()
        p1 = MonteCarloPlanner(seed=7)
        p2 = MonteCarloPlanner(seed=7)
        r1 = p1.plan(gate)
        r2 = p2.plan(gate)
        assert r1.recommended_mode == r2.recommended_mode
        for proj1, proj2 in zip(r1.ranked_modes, r2.ranked_modes):
            assert proj1.mean_laptime_delta_s == proj2.mean_laptime_delta_s

    def test_different_seeds_differ(self):
        gate = make_gate_result_all_legal()
        p1 = MonteCarloPlanner(seed=1)
        p2 = MonteCarloPlanner(seed=9999)
        r1 = p1.plan(gate)
        r2 = p2.plan(gate)
        balanced1 = next(p for p in r1.ranked_modes if p.mode == DeploymentMode.BALANCED_MODE)
        balanced2 = next(p for p in r2.ranked_modes if p.mode == DeploymentMode.BALANCED_MODE)
        # Samples differ between seeds — std should differ
        assert balanced1.std_laptime_delta_s != balanced2.std_laptime_delta_s


class TestRNGStreamIndependence:
    def test_removing_a_mode_does_not_change_other_modes_samples(self):
        """
        Critical: a mode's samples must not depend on which other modes are legal.
        Verified by running with all modes vs. conservative subset and checking
        that BALANCED_MODE samples are byte-identical.
        """
        gate_all = make_gate_result_all_legal()
        gate_conserv = make_gate_result_conservative()

        planner_all = MonteCarloPlanner(seed=42)
        planner_conserv = MonteCarloPlanner(seed=42)

        result_all = planner_all.plan(gate_all)
        result_conserv = planner_conserv.plan(gate_conserv)

        samples_all = planner_all.get_raw_samples(result_all, DeploymentMode.BALANCED_MODE)
        samples_conserv = planner_conserv.get_raw_samples(result_conserv, DeploymentMode.BALANCED_MODE)

        np.testing.assert_array_equal(samples_all, samples_conserv)


class TestRawSamples:
    def test_get_raw_samples_stale_run_id_raises(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_gate_result_conservative()
        r1 = planner.plan(gate)
        r2 = planner.plan(gate)  # run_id advances
        with pytest.raises(ValueError, match="stale"):
            planner.get_raw_samples(r1, DeploymentMode.BALANCED_MODE)

    def test_samples_have_correct_shape(self):
        planner = MonteCarloPlanner(seed=42)
        gate = make_gate_result_conservative()
        result = planner.plan(gate)
        samples = planner.get_raw_samples(result, DeploymentMode.BALANCED_MODE)
        assert samples.shape == (10_000,)


class TestNumericalStability:
    def test_sharpe_ratio_never_inf_or_nan(self):
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(make_gate_result_all_legal())
        for proj in result.ranked_modes:
            assert not np.isinf(proj.sharpe_ratio)
            assert not np.isnan(proj.sharpe_ratio)

    def test_probabilities_in_range(self):
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(make_gate_result_all_legal())
        for proj in result.ranked_modes:
            assert 0.0 <= proj.overtake_probability <= 1.0


class TestRivalModulation:
    def test_high_rival_soc_reduces_overtake_probability(self):
        """When rival has high SoC (strong defense), ARM/USE_BONUS modes should have lower overtake prob."""
        gate = make_gate_result_all_legal()

        p_high_rival = MonteCarloPlanner(seed=42)
        p_low_rival = MonteCarloPlanner(seed=42)

        ctx_high = PlanningContext(
            rival_soc_estimate=RivalSocEstimate(mean_soc_mj=8.5, std_soc_mj=0.3, n_observations=10)
        )
        ctx_low = PlanningContext(
            rival_soc_estimate=RivalSocEstimate(mean_soc_mj=1.0, std_soc_mj=0.3, n_observations=10)
        )

        r_high = p_high_rival.plan(gate, ctx_high)
        r_low = p_low_rival.plan(gate, ctx_low)

        bonus_high = next(p for p in r_high.ranked_modes if p.mode == DeploymentMode.USE_OVERTAKE_BONUS_MODE)
        bonus_low = next(p for p in r_low.ranked_modes if p.mode == DeploymentMode.USE_OVERTAKE_BONUS_MODE)

        # Higher rival SoC → stronger defense → lower effective overtake prob
        assert bonus_high.overtake_probability < bonus_low.overtake_probability
