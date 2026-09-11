"""
Regression tests for the Monte Carlo Planner / Opportunity Horizon strategic
hardening pass: rival-uncertainty propagation into the Horizon, the corrected
(non-double-counting) energy-opportunity-cost formula, risk exposure
(utility_std / downside_probability), and the Stage-2 counterfactual
(runner_up_mode / mode_value_gap_s / attack_completion_probability).

Architectural boundary this file exists to protect: Stage 2 (planner.py) and
the Opportunity Horizon (opportunity_engine.py) stay separate simulations —
neither is merged into the other — but both read rival uncertainty from the
SAME RivalSocEstimate via the SAME defense_penalty_weight/9.0 MJ formula, and
both read energy-economics constants from the SAME PlannerConfig fields
(harvest_per_lap_mj, attack_cost_mj).
"""

from __future__ import annotations

import numpy as np
import pytest

from opportunity_engine import HorizonConfig, OpportunityEngine
from planner import DEFAULT_MODE_DYNAMICS, MonteCarloPlanner, PlannerConfig, PlanningContext
from rival_estimator import RivalSocEstimate
from rule_gate import DeploymentMode, GateConfig, GateResult
from telemetry_simulator import TelemetryInput


def _all_legal_gate() -> GateResult:
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


def _telemetry(soc=7.0) -> TelemetryInput:
    return TelemetryInput(
        lap_number=25, current_soc_mj=soc, lap_start_soc_mj=soc + 0.5,
        lap_energy_deployed_mj=1.0, gap_to_car_ahead_s=0.7,
        overtake_qualified_last_lap=True, speed_kmh=290.0, total_laps=50,
    )


# ---------------------------------------------------------------------------
# Test C — rival uncertainty widens Horizon risk without moving the mean
# ---------------------------------------------------------------------------
class TestHorizonRivalUncertaintyIsRisk:
    def _wait2(self, std_mj):
        engine = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2, 5]), seed=42)
        rival = RivalSocEstimate(mean_soc_mj=4.5, std_soc_mj=std_mj, n_observations=10)
        result = engine.evaluate(
            _telemetry(), _all_legal_gate(), rival_estimate=rival,
            window_strength_by_delay={0: 1.0, 2: 1.0, 5: 1.0},
            start_soc_mj=7.0, reserve_floor_mj=1.0, overtake_reward_s=0.3,
        )
        return next(s for s in result.ranked_strategies if s.strategy_name == "WAIT_2")

    def test_same_mean_low_vs_high_uncertainty_widens_the_undiluted_signal(self):
        """The UNDILUTED signal (attack_completion_probability_std) must widen
        materially and monotonically with rival std — this is where 'different
        plausible rival states produce different attack outcomes' is honestly
        visible, at full strength, unmixed with the horizon's own pace noise."""
        low = self._wait2(0.5)
        high = self._wait2(2.5)
        assert low.attack_completion_probability is not None
        assert high.attack_completion_probability is not None
        # mean completion probability ~invariant to std (same rival MEAN)
        assert abs(low.attack_completion_probability - high.attack_completion_probability) < 0.05
        # the undiluted spread scales with rival std (~linear, pre-clipping) — a
        # 5x rival-std increase should produce a materially larger spread here
        assert high.attack_completion_probability_std > low.attack_completion_probability_std * 3.0, (
            f"high-uncertainty completion-probability std ({high.attack_completion_probability_std}) "
            f"should be materially larger than low-uncertainty "
            f"({low.attack_completion_probability_std})"
        )

    def test_utility_std_is_monotonic_but_may_be_diluted_by_pace_noise(self):
        """utility_std folds the (small) opportunity-value spread into the much
        larger horizon pace-noise budget. The mathematically correct claim is
        monotonic non-decrease, NOT a specific magnitude — forcing a larger
        threshold here would mean tuning the test to a desired answer rather
        than checking the model's actual (honest, currently modest) behaviour."""
        low = self._wait2(0.5)
        high = self._wait2(2.5)
        assert high.utility_std >= low.utility_std

    def test_downside_probability_does_not_fall_with_more_uncertainty(self):
        low = self._wait2(0.5)
        high = self._wait2(2.5)
        assert high.downside_probability >= low.downside_probability

    def test_no_rival_estimate_is_byte_identical_to_deterministic_fallback(self):
        """Omitting rival_estimate must reproduce the pre-hardening deterministic
        credit exactly — additive change, zero behavioural drift when unused."""
        engine = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2, 5]), seed=42)
        result = engine.evaluate(
            _telemetry(), _all_legal_gate(), rival_estimate=None,
            start_soc_mj=7.0, reserve_floor_mj=1.0, overtake_reward_s=0.3,
        )
        attack = next(s for s in result.ranked_strategies if s.strategy_name == "ATTACK_NOW")
        assert attack.attack_completion_probability is None
        assert attack.attack_completion_probability_std is None
        assert attack.utility_std == attack.std_horizon_delta_s


# ---------------------------------------------------------------------------
# Test D — energy opportunity cost: spend now vs preserve
# ---------------------------------------------------------------------------
class TestEnergyOpportunityCost:
    def test_scarcer_reserve_raises_opportunity_cost(self):
        """Same attack, less energy left in the tank afterward -> higher cost per MJ spent."""
        engine = OpportunityEngine(seed=42)
        gate = _all_legal_gate()
        comfortable = engine.evaluate(
            _telemetry(), gate, start_soc_mj=8.0, reserve_floor_mj=1.0, overtake_reward_s=0.3,
        )
        scarce = engine.evaluate(
            _telemetry(), gate, start_soc_mj=2.2, reserve_floor_mj=1.0, overtake_reward_s=0.3,
        )
        a_comfortable = next(s for s in comfortable.ranked_strategies if s.strategy_name == "ATTACK_NOW")
        a_scarce = next(s for s in scarce.ranked_strategies if s.strategy_name == "ATTACK_NOW")
        if a_comfortable.energy_spent_mj > 0 and a_scarce.energy_spent_mj > 0:
            density_comfortable = a_comfortable.energy_opportunity_cost / a_comfortable.energy_spent_mj
            density_scarce = a_scarce.energy_opportunity_cost / a_scarce.energy_spent_mj
            assert density_scarce >= density_comfortable

    def test_preserving_energy_for_a_stronger_future_window_can_beat_attacking_now(self):
        """CURRENT OPPORTUNITY = weak, FUTURE OPPORTUNITY = much stronger -> WAIT wins
        on strategic_value even though ATTACK_NOW's raw pace looks fine."""
        engine = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2]), seed=42)
        gate = _all_legal_gate()
        result = engine.evaluate(
            _telemetry(), gate,
            window_strength_by_delay={0: 0.3, 2: 1.4},  # weak now, strong later
            start_soc_mj=6.0, reserve_floor_mj=1.0, overtake_reward_s=0.3,
        )
        by_name = {s.strategy_name: s for s in result.ranked_strategies}
        assert by_name["WAIT_2"].future_opportunity_value > by_name["ATTACK_NOW"].current_opportunity_value

    def test_current_stronger_than_future_favours_attack_now(self):
        """Mirror case: CURRENT stronger than FUTURE -> attacking now should not be
        penalised relative to a materially worse future window."""
        engine = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2]), seed=42)
        gate = _all_legal_gate()
        result = engine.evaluate(
            _telemetry(), gate,
            window_strength_by_delay={0: 1.4, 2: 0.3},  # strong now, weak later
            start_soc_mj=6.0, reserve_floor_mj=1.0, overtake_reward_s=0.3,
        )
        by_name = {s.strategy_name: s for s in result.ranked_strategies}
        assert by_name["ATTACK_NOW"].current_opportunity_value > by_name["WAIT_2"].future_opportunity_value


# ---------------------------------------------------------------------------
# No-double-counting proof, verified numerically (not just structurally)
# ---------------------------------------------------------------------------
class TestNoDoubleCounting:
    def test_current_and_future_value_are_mutually_exclusive_per_strategy(self):
        engine = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2, 5]), seed=42)
        result = engine.evaluate(
            _telemetry(), _all_legal_gate(),
            start_soc_mj=7.0, reserve_floor_mj=1.0, overtake_reward_s=0.3,
        )
        for s in result.ranked_strategies:
            assert s.current_opportunity_value == 0.0 or s.future_opportunity_value == 0.0

    def test_energy_opportunity_cost_is_reconstructible_from_own_ledger_alone(self):
        """energy_opportunity_cost(s) must equal spent(s) * value_density * (1+scarcity(s)),
        where value_density/scarcity are built ONLY from s's own end_soc/spent and the
        shared (not strategy-specific) window_strength input — never from another
        strategy's current/future_opportunity_value. Reconstructing it from the
        public ledger alone proves no cross-strategy term leaked in."""
        overtake_reward_s = 0.3
        reserve_floor_mj = 1.0
        window_strength = {0: 1.0, 2: 1.1, 5: 0.9}
        engine = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2, 5]), seed=42)
        result = engine.evaluate(
            _telemetry(), _all_legal_gate(),
            window_strength_by_delay=window_strength,
            start_soc_mj=7.0, reserve_floor_mj=reserve_floor_mj, overtake_reward_s=overtake_reward_s,
        )
        attack_cost_mj = PlannerConfig().attack_cost_mj
        for s in result.ranked_strategies:
            if s.energy_spent_mj <= 0:
                assert s.energy_opportunity_cost == 0.0
                continue
            best_future_strength = max(
                [v for d, v in window_strength.items() if d > max(s.delay_laps, 0)],
                default=window_strength.get(max(s.delay_laps, 0), 1.0),
            )
            value_density = overtake_reward_s * best_future_strength / attack_cost_mj
            scarcity = max(0.0, (2.0 * reserve_floor_mj - s.end_soc_mj) / reserve_floor_mj)
            expected = s.energy_spent_mj * value_density * (1.0 + scarcity)
            assert abs(s.energy_opportunity_cost - round(expected, 4)) < 1e-3, s.strategy_name


# ---------------------------------------------------------------------------
# Stage-2 counterfactual + attack_completion_probability + shared constants
# ---------------------------------------------------------------------------
class TestStage2Counterfactual:
    def test_runner_up_and_value_gap_present_and_consistent(self):
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(_all_legal_gate(), PlanningContext(gap_to_car_ahead_s=0.5))
        assert result.runner_up_mode is not None
        assert result.runner_up_mode != result.recommended_mode
        assert result.mode_value_gap_s >= 0.0
        top = next(p for p in result.ranked_modes if p.mode == result.recommended_mode)
        runner_up = next(p for p in result.ranked_modes if p.mode == result.runner_up_mode)
        assert abs(
            (runner_up.mean_laptime_delta_s - top.mean_laptime_delta_s) - result.mode_value_gap_s
        ) < 1e-6

    def test_attack_completion_probability_only_on_attack_modes(self):
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(_all_legal_gate(), PlanningContext(gap_to_car_ahead_s=0.5))
        for p in result.ranked_modes:
            if DEFAULT_MODE_DYNAMICS[p.mode].overtake_success_prob > 0:
                assert p.attack_completion_probability is not None
                assert 0.0 <= p.attack_completion_probability <= 1.0
            else:
                assert p.attack_completion_probability is None

    def test_overtake_probability_field_unchanged_semantics(self):
        """Regression guard: overtake_probability must still be exactly
        mean(samples < 0) — the legacy field's behaviour must not silently drift."""
        planner = MonteCarloPlanner(seed=42)
        result = planner.plan(_all_legal_gate())
        for mode in result.ranked_modes:
            samples = planner.get_raw_samples(result, mode.mode)
            assert abs(mode.overtake_probability - round(float(np.mean(samples < 0)), 4)) < 1e-9


class TestSharedConstants:
    def test_harvest_and_attack_cost_are_single_sourced(self):
        """actions.FeasibilityInput's illustrative defaults must match
        PlannerConfig's canonical values — the audit's harvest-per-lap
        triplication is now down to one canonical number, cross-referenced."""
        from app.decision.actions import FeasibilityInput

        pc = PlannerConfig()
        fin_defaults = FeasibilityInput(
            gate_result=_all_legal_gate(), soc_mj=5.0, projected_reserve_mj=5.0,
            can_afford_aggressive=True, reserve_floor_mj=1.0, laps_remaining=10,
        )
        assert fin_defaults.harvest_per_lap_mj == pc.harvest_per_lap_mj
        assert fin_defaults.attack_cost_mj == pc.attack_cost_mj


# ---------------------------------------------------------------------------
# Determinism of every new field
# ---------------------------------------------------------------------------
def test_new_fields_are_deterministic():
    engine1 = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2, 5]), seed=7)
    engine2 = OpportunityEngine(config=HorizonConfig(delay_laps=[0, 2, 5]), seed=7)
    rival = RivalSocEstimate(mean_soc_mj=4.5, std_soc_mj=2.0, n_observations=12)
    kwargs = dict(
        rival_estimate=rival, start_soc_mj=6.5, reserve_floor_mj=1.0,
        overtake_reward_s=0.3, p_defend=0.35,
    )
    r1 = engine1.evaluate(_telemetry(), _all_legal_gate(), **kwargs)
    r2 = engine2.evaluate(_telemetry(), _all_legal_gate(), **kwargs)
    for s1, s2 in zip(r1.ranked_strategies, r2.ranked_strategies):
        assert s1.utility_std == s2.utility_std
        assert s1.downside_probability == s2.downside_probability
        assert s1.attack_completion_probability == s2.attack_completion_probability
        assert s1.attack_completion_probability_std == s2.attack_completion_probability_std
